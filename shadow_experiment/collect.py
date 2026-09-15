"""Dedicated shadow collector: subscribes to state, never imports a controller."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import queue
import select
import signal
import socket
import sys
import threading
import time

import numpy as np

from .core import GroundTruth, Pipelines, StateBuffer, TrialWriter, load_config, preprocess
from .camera_tap import recv_packet
from common.remote_controller import RemoteController, KeyMap


class Collector:
    def __init__(self, config):
        self.config = config
        self.pipelines = Pipelines(config)
        self.start_ns = time.monotonic_ns()
        self.truth = GroundTruth(config, self.start_ns)
        self.writer = TrialWriter(config, self.pipelines.metadata, self.start_ns)
        self.states = StateBuffer(config['runtime']['state_buffer_size'])
        self.positions = StateBuffer(config['runtime']['state_buffer_size'])
        self.queue = queue.Queue(config['runtime']['queue_size'])
        self.commands = queue.Queue(32)
        self.drop_events = queue.Queue(64)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.counters = Counter()
        self.source_error = None
        self.last_id, self.last_capture, self.last_update = None, None, None
        self.last_sensor_stamp = None
        self.last_state_tick = None
        self.temporal_segment = 0
        self.continuity_broken = False
        self.previous_queue_drops = 0
        self.previous_tap_drops = None
        self.termination = 'operator_stop'
        self.gamepad = RemoteController()
        self.gamepad_initialized = False

    def offer(self, metadata, raw, rgb):
        self.counters['capture_received'] += 1
        if 'tap_dropped' in metadata:
            self.counters['tap_dropped_last_observed_cumulative'] = metadata['tap_dropped']
            self.counters['tap_offered_last_observed_cumulative'] = metadata['tap_offered']
        try:
            self.queue.put_nowait((metadata, raw, rgb))
        except queue.Full:
            self.counters['capture_queue_drop_new'] += 1
            try:
                self.drop_events.put_nowait({'frame_id': metadata.get('frame_id'), 'receipt_ns': metadata.get('receipt_ns')})
            except queue.Full:
                self.counters['drop_event_queue_overflow'] += 1

    def state(self, msg):
        if self.stop.is_set():
            return
        tick = int(msg.tick)
        if self.last_state_tick is not None and tick <= self.last_state_tick:
            self.counters['duplicate_or_reordered_state_tick'] += 1
            return
        self.last_state_tick = tick
        sample = {'receipt_ns': time.monotonic_ns(), 'sensor_tick': int(msg.tick),
                  'rpy': list(msg.imu_state.rpy), 'omega': list(msg.imu_state.gyroscope)}
        with self.lock:
            self.states.add(sample)
            self.gamepad_marker(msg, sample['receipt_ns'])
        self.counters['state_received'] += 1

    def gamepad_marker(self, msg, receipt_ns):
        settings = self.config['transition'].get('gamepad', {})
        if not settings.get('enabled', False):
            return
        packet = getattr(msg, 'wireless_remote', None)
        try:
            if packet is None or len(packet) != 40:
                raise ValueError('Expected 40-byte wireless_remote')
            self.gamepad.set(bytes(packet))
        except (ValueError, TypeError, OverflowError):
            self.counters['invalid_gamepad_packets'] += 1
            self.gamepad = RemoteController()
            self.gamepad_initialized = False
            return
        if not self.gamepad_initialized:
            # A held button at startup/recovery is not a new operator marker.
            self.gamepad_initialized = True
            return
        modifier = getattr(KeyMap, settings['modifier']) if settings.get('modifier') else None
        button = getattr(KeyMap, settings['button'])
        buttons = self.gamepad.button
        if not (buttons[button].on_press and (modifier is None or buttons[modifier].pressed)):
            return
        if any(b.pressed for i, b in enumerate(buttons) if i not in (modifier, button)):
            return
        event = self.truth.observe(receipt_ns, marker=self.config['transition']['marker'])
        details = {'input_source': 'gamepad', 'buttons': [name for name in (settings.get('modifier'), settings['button']) if name],
                   'sensor_tick': int(msg.tick), 'timestamp_ns': receipt_ns}
        self.writer.event('gamepad_marker', latched=event is not None, **details)
        if event:
            self.truth.event.update(details)
            self.writer.event('transition', **self.truth.event)
            print('Transition annotated via gamepad: {} at {:.3f} s'.format(
                ' + '.join(details['buttons']), (receipt_ns-self.start_ns)/1e9), flush=True)

    def position(self, msg):
        if self.stop.is_set():
            return
        sample = {'receipt_ns': time.monotonic_ns(), 'xyz': list(msg.position),
                  'source': 'rt/sportmodestate.position; reported odometry, not verified physical crossing',
                  'sensor_stamp': {'sec': int(msg.stamp.sec), 'nanosec': int(msg.stamp.nanosec)},
                  'sensor_clock': 'robot sport-state clock; relation to host unknown'}
        if not np.isfinite(sample['xyz']).all():
            self.counters['invalid_position_samples'] += 1
            return
        with self.lock:
            self.positions.add(sample)
            event = self.truth.observe(sample['receipt_ns'], position=sample) if self.config['transition']['type'] == 'position' else None
        if event:
            self.writer.event('transition', **event)

    def _stdin(self):
        while not self.stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], .2)
            if not ready:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            try:
                self.commands.put_nowait((time.monotonic_ns(), line.strip()))
            except queue.Full:
                self.counters['operator_command_drops'] += 1

    def log_drops(self):
        while True:
            try:
                drop = self.drop_events.get_nowait()
            except queue.Empty:
                return
            self.writer.event('dropped_frame', reason='capture_queue_full', **drop)

    def commands_pending(self):
        while True:
            try:
                stamp, command = self.commands.get_nowait()
            except queue.Empty:
                break
            self.writer.event('operator_command', timestamp_ns=stamp, command=command)
            if command == 'stop':
                self.stop.set()
            elif command.startswith('mark '):
                event = self.truth.observe(stamp, marker=command[5:])
                if event:
                    self.writer.event('transition', **event)
            elif command.startswith('crossing '):
                event = self.truth.verify_crossing(stamp, command[9:])
                self.writer.event('crossing_verification', **event)

    def process(self, metadata, raw, rgb):
        now = time.monotonic_ns()
        receipt = metadata['receipt_ns']
        frame_id = metadata['frame_id']
        def reject(reason):
            self.counters[reason] += 1
            if reason != 'rate_decimation':
                self.continuity_broken = True
            self.writer.event('rejected_frame', reason=reason, frame_id=frame_id, receipt_ns=receipt)
        if receipt < self.start_ns or receipt > now:
            reject('invalid_receipt_timestamp')
            return
        if self.last_id is not None and frame_id <= self.last_id:
            reject('duplicate_or_reordered_frame')
            return
        if self.last_sensor_stamp is not None and metadata['sensor_timestamp_ms'] <= self.last_sensor_stamp:
            reject('stale_sensor_timestamp')
            return
        if self.last_id is not None and frame_id > self.last_id + 1:
            self.counters['sensor_frame_number_gaps'] += frame_id - self.last_id - 1
        self.last_id = frame_id
        self.last_sensor_stamp = metadata['sensor_timestamp_ms']
        with self.lock:
            state = self.states.causal(receipt, self.config['runtime']['max_state_age_s'])
            position = self.positions.causal(receipt, self.config['runtime']['max_state_age_s'])
        event = self.truth.observe(receipt, frame_id, position)
        if event:
            self.writer.event('transition', **event)
        if self.truth.event and self.truth.event['first_frame_id'] is None and receipt >= self.truth.event['timestamp_ns']:
            self.truth.event['first_frame_id'] = frame_id
            self.writer.event('transition_first_frame', frame_id=frame_id, receipt_ns=receipt)
        period = 1e9 / self.config['runtime']['update_hz']
        if self.last_capture is not None and receipt - self.last_capture < period:
            reject('rate_decimation')
            return
        if now - receipt > self.config['runtime']['max_input_age_s'] * 1e9:
            reject('stale_host_input')
            return
        if state is None:
            reject('missing_or_stale_state')
            return
        if not np.isfinite(state['rpy'] + state['omega']).all():
            reject('invalid_state')
            return
        if rgb is not None and metadata.get('rgb_sensor_clock') != metadata['sensor_clock']:
            reject('incompatible_rgb_clock')
            return
        if rgb is not None and abs(metadata['rgb_timestamp_ms'] - metadata['sensor_timestamp_ms']) > self.config['camera']['max_rgb_skew_ms']:
            reject('unsynchronized_rgb')
            return
        cycle_start = time.perf_counter_ns()
        try:
            depth = preprocess(raw, metadata['depth_scale_m'], self.config['camera'])
        except ValueError as error:
            self.writer.event('invalid_depth', frame_id=frame_id, error=str(error))
            self.counters['invalid_depth'] += 1
            self.continuity_broken = True
            return
        preprocess_ms = (time.perf_counter_ns() - cycle_start) / 1e6
        try:
            result = self.pipelines.run(depth, state['rpy'], state['omega'])
        except Exception:
            # Do not allow partially advanced filter banks to continue silently.
            self.termination = 'pipeline_error'
            raise
        total_ms = (time.perf_counter_ns() - cycle_start) / 1e6
        elapsed = (receipt - self.start_ns) / 1e9
        valid = not any(a <= elapsed < b for a, b in self.config['analysis']['excluded_intervals_s'])
        end = time.monotonic_ns()
        queue_drops = self.counters['capture_queue_drop_new']
        tap_drops = metadata.get('tap_dropped')
        if self.continuity_broken or queue_drops > self.previous_queue_drops or (tap_drops is not None and self.previous_tap_drops is not None and tap_drops > self.previous_tap_drops):
            self.temporal_segment += 1
        self.continuity_broken = False
        self.previous_queue_drops, self.previous_tap_drops = queue_drops, tap_drops
        record = {**metadata, **result, 'temporal_segment': self.temporal_segment, 'trial_id': self.config['trial_id'],
                  'accepted_index': self.counters['accepted'], 'elapsed_s': elapsed,
                  'state': state, 'position': position,
                  'ground_truth': self.truth.label(receipt), 'configured_target': self.config['target_class'],
                  'annotation_transition': self.truth.event, 'verified_crossing': self.truth.crossing,
                  'valid_for_analysis': valid, 'exclusion_reason': None if valid else 'configured_uncertain_interval',
                  'processing_start_ns': now, 'processing_end_ns': end,
                  'preprocess_ms': preprocess_ms, 'six_pipeline_cycle_ms': total_ms,
                  'update_interval_ms': None if self.last_update is None else (now - self.last_update) / 1e6,
                  'input_age_ms': (now - receipt) / 1e6,
                  'state_alignment_age_ms': (receipt - state['receipt_ns']) / 1e6,
                  'deadline_miss': end - receipt > period,
                  'compute_deadline_miss': total_ms * 1e6 > period,
                  'counters': dict(self.counters),
                  'state_buffer_evictions': self.states.evictions}
        self.writer.add(record, raw, depth, state['rpy'], state['omega'], rgb)
        self.counters['accepted'] += 1
        self.last_capture, self.last_update = receipt, now

    def run(self, source):
        thread = threading.Thread(target=self._capture, args=(source,), daemon=True)
        thread.start()
        threading.Thread(target=self._stdin, daemon=True).start()
        print('Trial:', self.writer.path, flush=True)
        print('Commands: mark <configured marker>; crossing <evidence>; stop', flush=True)
        gamepad = self.config['transition'].get('gamepad', {})
        if gamepad.get('enabled'):
            print('Gamepad annotation: {}; release held buttons first.'.format(' + '.join(name for name in (gamepad.get('modifier'), gamepad['button']) if name)), flush=True)
        reason = 'operator_stop'
        try:
            while not self.stop.is_set():
                self.log_drops()
                self.commands_pending()
                if self.config['transition']['type'] == 'time':
                    event = self.truth.observe(time.monotonic_ns())
                    if event:
                        self.writer.event('transition', **event)
                duration = self.config['runtime'].get('duration_s')
                if duration and (time.monotonic_ns() - self.start_ns) / 1e9 >= duration:
                    reason = 'duration_reached'
                    break
                if self.source_error:
                    raise RuntimeError(self.source_error)
                try:
                    packet = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                self.commands_pending()
                if not self.stop.is_set():
                    self.process(*packet)
        except KeyboardInterrupt:
            reason = 'interrupted'
        except Exception as error:
            reason = self.termination if self.termination == 'pipeline_error' else 'source_or_processing_error'
            self.writer.event('error', message=repr(error))
            raise
        finally:
            self.stop.set()
            thread.join(timeout=3)
            if thread.is_alive():
                self.counters['capture_shutdown_timeout'] += 1
            self.commands_pending()
            self.log_drops()
            self.counters['queued_unprocessed_at_stop'] = self.queue.qsize()
            while not self.queue.empty():
                metadata, _, _ = self.queue.get_nowait()
                self.writer.event('dropped_frame', reason='queued_at_termination', frame_id=metadata.get('frame_id'), receipt_ns=metadata.get('receipt_ns'))
            if self.termination.startswith('signal_'):
                reason = self.termination
            self.writer.close(reason, self.truth, dict(self.counters))

    def _capture(self, source):
        try:
            source(self)
        except Exception as error:
            if not self.stop.is_set():
                self.source_error = repr(error)


def realsense_source(collector):
    import pyrealsense2 as rs
    c = collector.config['camera']
    pipeline, cfg = rs.pipeline(), rs.config()
    h, w = c['resolution']
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, c['fps'])
    if c['rgb']:
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, c['fps'])
    profile = pipeline.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    device = profile.get_device()
    collector.writer.event('camera_identity', serial=device.get_info(rs.camera_info.serial_number),
                           firmware=device.get_info(rs.camera_info.firmware_version),
                           name=device.get_info(rs.camera_info.name), depth_scale_m=scale,
                           intrinsics=str(profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()))
    try:
        while not collector.stop.is_set():
            frames = pipeline.wait_for_frames(2000)
            receipt = time.monotonic_ns()
            depth = frames.get_depth_frame()
            if not depth:
                collector.counters['missing_depth_capture'] += 1
                continue
            color = frames.get_color_frame() if c['rgb'] else None
            if c['rgb'] and not color:
                collector.counters['missing_rgb_capture'] += 1
                continue
            metadata = {'frame_id': int(depth.get_frame_number()), 'receipt_ns': receipt,
                        'sensor_timestamp_ms': depth.get_timestamp(), 'sensor_clock': str(depth.get_frame_timestamp_domain()),
                        'depth_scale_m': scale, 'raw_units': 'uint16 sensor units; multiply depth_scale_m for meters',
                        'rgb_timestamp_ms': color.get_timestamp() if color else None,
                        'rgb_sensor_clock': str(color.get_frame_timestamp_domain()) if color else None,
                        'rgb_frame_id': int(color.get_frame_number()) if color else None,
                        'source': 'realsense', 'rgb_alignment': 'same frameset, unregistered color view'}
            collector.offer(metadata, np.asanyarray(depth.get_data()).copy(),
                            np.asanyarray(color.get_data()).copy() if color else None)
    finally:
        pipeline.stop()


def socket_source(collector):
    path = Path(collector.config['camera']['socket_path'])
    if path.exists():
        raise FileExistsError('Socket path already exists; verify no collector owns it before removing it')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path, 0o600)
    server.listen(1)
    server.settimeout(.2)
    try:
        while not collector.stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            connection.settimeout(2)
            with connection:
                while not collector.stop.is_set():
                    collector.offer(*recv_packet(connection))
    finally:
        server.close()
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--trial-id')
    parser.add_argument('--interface', default='eth0')
    args = parser.parse_args()
    config = load_config(args.config, args.trial_id)
    # Only subscriptions; no controller module, publisher, or mode-switch client.
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
    ChannelFactoryInitialize(1 if args.interface == 'lo' else 0, args.interface)
    collector = Collector(config)
    lowstate = ChannelSubscriber('rt/lowstate', LowState_)
    lowstate.Init(collector.state, 10)
    position = ChannelSubscriber('rt/sportmodestate', SportModeState_)
    position.Init(collector.position, 10)
    def terminate(signum, frame):
        collector.termination = 'signal_' + str(signum)
        collector.stop.set()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    collector.writer.event('hardware_interface', interface=args.interface, robot_identity=config['robot_identity'])
    source = socket_source if config['camera']['source'] == 'publisher_tap' else realsense_source
    if config['camera']['source'] == 'publisher_tap' and config['camera']['rgb']:
        raise ValueError('Publisher tap does not supply RGB; use direct realsense source for optional RGB')
    collector.run(source)


if __name__ == '__main__':
    main()
