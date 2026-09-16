"""Benchmark each classifier/filter on fresh frames from the deployment depth node."""
from common.dds_runtime import configure_dds_runtime

if __name__ == '__main__':
    configure_dds_runtime()

import argparse
from collections import Counter
import csv
import gc
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch
import yaml

from .analyze import KEYS, write_csv
from .camera_tap import recv_packet
from .core import ROOT, Pipelines, StateBuffer, atomic_json, load_config, provenance


def measure(pipeline, depth, rpy, omega):
    pipeline.sync()
    start = time.perf_counter_ns()
    result = pipeline.run(depth, rpy, omega)
    pipeline.sync()
    return result, (time.perf_counter_ns()-start)/1e6


MEMORY_FIELDS = ('process_rss_mib', 'cuda_allocated_mib', 'cuda_reserved_mib')


def sample_memory(device):
    """Current process RSS and optional Torch CUDA allocations, outside timing.

    RSS includes Python/DDS/input buffers and allocator caches, not the separate
    camera process. These are post-call samples, not transient peak measurements.
    """
    resident_pages = int(Path('/proc/self/statm').read_text().split()[1])
    sample = {'process_rss_mib': resident_pages * os.sysconf('SC_PAGE_SIZE') / 2**20,
              'cuda_allocated_mib': None, 'cuda_reserved_mib': None}
    if torch.device(device).type == 'cuda':
        sample.update(cuda_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
                      cuda_reserved_mib=torch.cuda.memory_reserved(device) / 2**20)
    return sample


def summarize_samples(key, values):
    summary = {'pipeline': key, 'samples': len(values)}
    for field in ('total_ms', 'classifier_ms', 'filter_ms', *MEMORY_FIELDS):
        samples = [r[field] for r in values if r[field] is not None]
        summary[field + '_mean'] = float(np.mean(samples)) if samples else None
        summary[field + '_std'] = float(np.std(samples, ddof=1)) if len(samples) > 1 else None
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--interface', default='eth0')
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.iterations < 2 or args.warmup < 0:
        p.error('iterations >= 2 and warmup >= 0 required')
    c = load_config(args.config)
    if c['camera']['rgb'] or not c['camera']['realsense_filters'] or c['camera']['preprocessing'] != 'deployment_tensor_only':
        p.error('Requires deployment preprocessing, filters enabled, RGB disabled')
    c['runtime'].update(device=args.device, torch_threads=1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'resolved.yaml').write_text(yaml.safe_dump(c))
    counters, states, lock = Counter(), StateBuffer(c['runtime']['state_buffer_size']), threading.Lock()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    ChannelFactoryInitialize(1 if args.interface == 'lo' else 0, args.interface)
    def state(msg):
        sample = {'receipt_ns': time.monotonic_ns(), 'sensor_tick': int(msg.tick),
                  'rpy': list(msg.imu_state.rpy), 'omega': list(msg.imu_state.gyroscope)}
        with lock:
            states.add(sample)
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    subscriber.Init(state, 10)
    manifest = {'complete': False, 'iterations_per_pipeline': args.iterations,
                'warmup_per_pipeline': args.warmup, 'pipeline_order': list(KEYS),
                'scope': 'one active classifier/filter with concurrent deployment camera node; fresh normalized input each call; excludes frame wait, camera filtering/resizing, model loading and disk writes; includes tensor transfers, validation and result packaging',
                'std_convention': 'sample standard deviation ddof=1',
                'memory_scope': 'post-call process RSS from /proc/self/statm, MiB (2^20 bytes); excludes separate camera process, includes Python/DDS/buffers and allocator caches; not model-only or transient peak memory. Pipelines share one process, so retained allocations can affect later pipelines. CUDA allocated/reserved bytes are Torch allocator samples, not total device memory; do not add them to RSS on unified-memory hardware. Sampling is outside latency timing.',
                'filter_policy': 'reset after warmup; retain state during measurements',
                'clock': 'perf_counter_ns with CUDA synchronization before/after calls',
                'input_policy': 'fresh live frames; different pipelines see different frames',
                'provenance': provenance(c), 'models': {}}
    atomic_json(output/'manifest.json', manifest)
    summaries = []
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with tempfile.TemporaryDirectory(prefix='live_latency_') as directory:
            path = Path(directory)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path/'camera.sock'))
                server.listen(1)
                server.settimeout(30)
                camera = {'depth_camera': c['camera'], 'depth_image_shape': [48,64],
                          'cnn_rate_hz': c['runtime']['update_hz'], 'torch_num_threads': 1,
                          'torch_num_interop_threads': 1}
                (path/'camera.yaml').write_text(yaml.safe_dump(camera))
                with (output/'camera.log').open('w') as log:
                    publisher = subprocess.Popen([sys.executable, '-u', str(ROOT/'rough_depth_image.py'),
                        '--config', str(path/'camera.yaml'), '--interface', args.interface,
                        '--shadow-socket', str(path/'camera.sock')], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                    try:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(10)
                            last_id, last_sensor = -1, -1
                            def fresh(after_ns):
                                nonlocal last_id, last_sensor
                                deadline = time.monotonic()+30
                                while time.monotonic() < deadline:
                                    m, _, _ = recv_packet(connection)
                                    counters['received'] += 1
                                    if m['frame_id'] <= last_id or m['sensor_timestamp_ms'] <= last_sensor:
                                        counters['duplicate_or_reordered'] += 1
                                        continue
                                    last_id, last_sensor = m['frame_id'], m['sensor_timestamp_ms']
                                    if m['receipt_ns'] <= after_ns or not 0 <= time.monotonic_ns()-m['receipt_ns'] <= c['runtime']['max_input_age_s']*1e9:
                                        counters['old_frame'] += 1
                                        continue
                                    with lock:
                                        s = states.causal(m['receipt_ns'], c['runtime']['max_state_age_s'])
                                    depth = m.pop('_processed_depth', None)
                                    if s is None or depth is None or depth.shape != (48,64) or not all(np.isfinite(v).all() for v in (depth,s['rpy'],s['omega'])):
                                        counters['invalid_or_unaligned'] += 1
                                        continue
                                    return m, depth, s
                                raise TimeoutError('No fresh aligned input for 30 seconds')
                            for key in KEYS:
                                pipeline = Pipelines(c, only=key)
                                name = key.split('/')[0]
                                manifest['models'][name] = pipeline.metadata[name]
                                atomic_json(output/'manifest.json', manifest)
                                for _ in range(args.warmup):
                                    m, depth, s = fresh(time.monotonic_ns())
                                    pipeline.run(depth,s['rpy'],s['omega'])
                                pipeline.reset()
                                values = []
                                fields = ['iteration','frame_id','sensor_timestamp_ms','receipt_ns','state_tick','input_age_ms','total_ms','classifier_ms','filter_ms','tap_dropped', *MEMORY_FIELDS]
                                with (output/(key.replace('/','_')+'_samples.csv')).open('w') as f:
                                    writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                                    for i in range(args.iterations):
                                        m, depth, s = fresh(time.monotonic_ns())
                                        age = (time.monotonic_ns()-m['receipt_ns'])/1e6
                                        result, ms = measure(pipeline,depth,s['rpy'],s['omega'])
                                        row = dict(iteration=i,frame_id=m['frame_id'],sensor_timestamp_ms=m['sensor_timestamp_ms'],receipt_ns=m['receipt_ns'],state_tick=s['sensor_tick'],input_age_ms=age,total_ms=ms,classifier_ms=result['models'][name]['duration_ms'],filter_ms=result['selectors'][key]['duration_ms'],tap_dropped=m.get('tap_dropped'))
                                        row.update(sample_memory(pipeline.device))
                                        values.append(row);writer.writerow(row);f.flush()
                                summary = summarize_samples(key, values)
                                summaries.append(summary);write_csv(output/'summary.csv',summaries)
                                print('{}: {:.3f} +/- {:.3f} ms (n={})'.format(key,summary['total_ms_mean'],summary['total_ms_std'],len(values)),flush=True)
                                print('  process RSS: {:.3f} +/- {:.3f} MiB'.format(summary['process_rss_mib_mean'], summary['process_rss_mib_std']), flush=True)
                                if summary['cuda_allocated_mib_mean'] is not None:
                                    print('  CUDA allocated: {:.3f} +/- {:.3f} MiB; reserved: {:.3f} +/- {:.3f} MiB'.format(summary['cuda_allocated_mib_mean'], summary['cuda_allocated_mib_std'], summary['cuda_reserved_mib_mean'], summary['cuda_reserved_mib_std']), flush=True)
                                del pipeline
                                gc.collect()
                    finally:
                        if publisher.poll() is None:
                            publisher.send_signal(signal.SIGINT)
                            try:
                                publisher.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                publisher.kill();publisher.wait()
        manifest['complete'] = True
    finally:
        manifest.update(counters=dict(counters), completed_pipelines=len(summaries))
        atomic_json(output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
