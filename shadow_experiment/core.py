"""Trial state, exact inputs, shared classifiers, and crash-recoverable records."""
import copy
import hashlib
from importlib import metadata as package_metadata
import json
import os
import platform
import re
import subprocess
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common.depth_processing import preprocess_depth_array
from terrain_selector import TerrainSelector

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('rough', 'gap', 'stairs', 'climb')
MODES = ('instantaneous', 'ema', 'bayes')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git_info(path):
    def run(*args):
        p = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {'path': str(path), 'commit': run('rev-parse', 'HEAD'),
            'status': run('status', '--porcelain'), 'diff': run('diff', 'HEAD')}


def provenance(config):
    versions = {}
    for name in ('pyrealsense2', 'unitree_sdk2py', 'cyclonedds', 'matplotlib', 'PyYAML'):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    files = [*Path(__file__).parent.glob('*.py'), ROOT / 'terrain_selector.py',
             ROOT / 'common/depth_processing.py', ROOT / 'rough_depth_image.py']
    return {
        'packages': versions,
        'inspected_reference': json.loads((Path(__file__).parent / 'reference_snapshot.json').read_text()),
        'deployment': git_info(ROOT), 'reference': git_info(config['reference_repo']),
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'host': platform.node(), 'platform': platform.platform(),
        'machine': platform.machine(), 'device': config['runtime']['device'],
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'clock': {'ordering': 'host monotonic_ns on one host; receipt alignment, not sensor exposure alignment',
                  'wall': 'time_ns Unix UTC anchor only; may jump',
                  'camera': 'RealSense timestamp milliseconds and domain recorded; not subtracted from host time',
                  'state': 'lowstate tick recorded as opaque device counter; no inferred host offset'},
        'state_conventions': 'Unitree lowstate imu_state.rpy radians; gyroscope body rad/s, unscaled',
        'timing_scope': 'batch-size-one concurrent shadow-mode selector cycle; no specialist execution',
    }


def load_config(path, trial_id=None):
    c = yaml.safe_load(Path(path).read_text())
    if trial_id is not None:
        c['trial_id'] = str(trial_id)
    for field in ('experiment', 'configuration', 'difficulty', 'trial_id'):
        c[field] = str(c[field])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', c[field]):
            raise ValueError('Use filesystem-safe explicit ' + field)
    if c['initial_class'] != 'rough' or c['target_class'] not in CLASSES[1:]:
        raise ValueError('Trials must start rough and target gap, stairs, or climb')
    if not c.get('terrain_description') or not isinstance(c.get('obstacle_dimensions_m'), dict):
        raise ValueError('Terrain description and obstacle dimensions are required')
    trigger = c['transition']
    if trigger['type'] not in ('time', 'position', 'operator'):
        raise ValueError('transition.type must be time, position, or operator')
    if trigger['type'] == 'time' and not (np.isfinite(trigger['seconds']) and trigger['seconds'] >= 0):
        raise ValueError('Invalid time trigger')
    if trigger['type'] == 'position':
        if trigger['axis'] not in ('x', 'y', 'z') or trigger['comparison'] not in ('ge', 'le'):
            raise ValueError('Invalid position trigger')
        if not np.isfinite(trigger['threshold_m']):
            raise ValueError('Position threshold must be finite')
    if trigger['type'] == 'operator' and not trigger.get('marker'):
        raise ValueError('Operator trigger needs a marker name')
    gamepad = trigger.setdefault('gamepad', {'enabled': False, 'modifier': None, 'button': 'B'})
    from common.remote_controller import KeyMap
    if not isinstance(gamepad['enabled'], bool):
        raise ValueError('transition.gamepad.enabled must be a boolean')
    if gamepad['enabled']:
        if trigger['type'] != 'operator':
            raise ValueError('Gamepad annotation requires an operator transition')
        names = {k for k, v in vars(KeyMap).items() if not k.startswith('_') and isinstance(v, int)}
        if (gamepad.get('modifier') is not None and gamepad['modifier'] not in names) or gamepad['button'] not in names or gamepad.get('modifier') == gamepad['button']:
            raise ValueError('Gamepad marker needs a valid button and an optional distinct modifier')
    if set(c['models']) != {'feature', 'raw'}:
        raise ValueError('Exactly feature and raw classifiers are required')
    for name, spec in c['models'].items():
        spec['path'] = str((ROOT / spec['path']).resolve())
        if 'seed' not in spec:
            raise ValueError('Record model seed, or explicitly null if unknown')
    if set(c['class_mapping'].values()) != set(CLASSES):
        raise ValueError('Class mapping must cover rough/gap/stairs/climb')
    cam = c['camera']
    if len(cam['resolution']) != 2 or any(not isinstance(v, int) or v <= 0 for v in cam['resolution']):
        raise ValueError('Camera resolution must be two positive integers')
    if len(cam['cropping']) != 4 or any(not isinstance(v, int) or v < 0 for v in cam['cropping']):
        raise ValueError('Camera cropping must be four nonnegative integers')
    if cam['fps'] <= 0 or cam['max_rgb_skew_ms'] < 0:
        raise ValueError('Invalid camera rate or RGB skew')
    if cam['source'] not in ('realsense', 'publisher_tap'):
        raise ValueError('camera.source must be realsense or publisher_tap')
    if cam['source'] == 'publisher_tap' and cam['rgb']:
        raise ValueError('RGB requires direct realsense capture')
    if cam['preprocessing'] not in ('training_bicubic', 'deployment_tensor_only'):
        raise ValueError('Unknown preprocessing')
    if cam['preprocessing'] == 'deployment_tensor_only' and cam.get('realsense_filters', False):
        raise ValueError('This collector records unfiltered raw Z16; hardware RS filters are not available in replay')
    if cam.get('realsense_filters', False):
        raise ValueError('Synthetic training noise and stateful RS filters are not applied to raw hardware captures')
    runtime = c['runtime']
    for k in ('update_hz', 'queue_size', 'state_buffer_size', 'chunk_frames', 'max_state_age_s', 'max_input_age_s'):
        if not np.isfinite(runtime[k]) or runtime[k] <= 0:
            raise ValueError('runtime.' + k + ' must be positive')
    for k in ('queue_size', 'state_buffer_size', 'chunk_frames', 'torch_threads'):
        if isinstance(runtime[k], bool) or int(runtime[k]) != runtime[k] or runtime[k] < 1:
            raise ValueError('runtime.' + k + ' must be a positive integer')
    for interval in c['analysis']['excluded_intervals_s']:
        if len(interval) != 2 or not 0 <= interval[0] < interval[1]:
            raise ValueError('Invalid excluded interval')
    c['output_root'] = str((ROOT / c['output_root']).resolve())
    c['reference_repo'] = str(Path(c['reference_repo']).expanduser().resolve())
    return c


@torch.inference_mode()
def preprocess(raw, scale, camera):
    """Training depth_mixin crop/normalize/bicubic; no synthetic sensor noise.

    Hardware 640x480 crop is scaled from training's 160x120 crop. This reproduces
    tensor operations and field of view, not synthetic rendering/noise/latency.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Invalid raw depth')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid meters per raw unit')
    if list(raw.shape) != camera['resolution']:
        raise ValueError('Raw depth resolution disagrees with resolved configuration')
    if camera['preprocessing'] == 'deployment_tensor_only':
        return preprocess_depth_array(raw, scale, cropping=camera['cropping'],
                                      depth_range_m=camera['depth_range_m'],
                                      rotate_180=camera['rotate_180'])
    near, far = camera['depth_range_m']
    top, bottom, left, right = camera['cropping']
    if not 0 <= near < far or min(top, bottom, left, right) < 0:
        raise ValueError('Invalid clip/crop')
    h, w = raw.shape
    if top + bottom >= h or left + right >= w:
        raise ValueError('Crop removes entire frame')
    if camera['rotate_180']:
        raw = np.rot90(raw, 2)
    depth = torch.from_numpy(raw.astype(np.float32)).mul_(scale)
    depth = (depth.clamp(near, far) - near) / max(far - near, 1e-6)
    depth = depth[top:h-bottom if bottom else h, left:w-right if right else w]
    depth = F.interpolate(depth[None, None], size=(48, 64), mode='bicubic', align_corners=False)
    return depth[0, 0].clamp(0, 1).numpy()


class GroundTruth:
    def __init__(self, config, start_ns):
        self.config, self.start_ns = config, start_ns
        self.event = None
        self.crossing = None

    def observe(self, stamp_ns, frame_id=None, position=None, marker=None):
        t = self.config['transition']
        fired, event_ns, value = False, stamp_ns, None
        if t['type'] == 'time':
            event_ns = self.start_ns + round(t['seconds'] * 1e9)
            fired, value = stamp_ns >= event_ns, t['seconds']
        elif t['type'] == 'operator':
            fired, value = marker == t['marker'], marker
        elif position is not None:
            value = position['xyz'][('x', 'y', 'z').index(t['axis'])]
            fired = value >= t['threshold_m'] if t['comparison'] == 'ge' else value <= t['threshold_m']
            event_ns = position['receipt_ns']
        if fired and self.event is None:
            self.event = {'timestamp_ns': event_ns, 'observed_ns': stamp_ns,
                          'first_frame_id': frame_id, 'source': t['type'], 'value': value,
                          'annotation_kind': 'configured_transition', 'physical_crossing_verified': False}
            return dict(self.event)
        return None

    def label(self, stamp_ns):
        return self.config['target_class'] if self.event and stamp_ns >= self.event['timestamp_ns'] else 'rough'

    def verify_crossing(self, stamp_ns, evidence):
        if not evidence.strip():
            raise ValueError('Crossing verification requires operator evidence')
        if self.crossing is None:
            self.crossing = {'timestamp_ns': stamp_ns, 'source': 'operator_verification', 'evidence': evidence}
        return self.crossing


class StateBuffer:
    def __init__(self, capacity):
        self.samples = deque(maxlen=capacity)
        self.evictions = 0

    def add(self, sample):
        if len(self.samples) == self.samples.maxlen:
            self.evictions += 1
        self.samples.append(sample)

    def causal(self, stamp_ns, max_age_s):
        candidates = [s for s in self.samples if s['receipt_ns'] <= stamp_ns]
        if not candidates:
            return None
        sample = max(candidates, key=lambda s: s['receipt_ns'])
        return sample if stamp_ns - sample['receipt_ns'] <= max_age_s * 1e9 else None


class Pipelines:
    """Two model evaluations, three independently reset filters each."""
    def __init__(self, config, only=None):
        self.config = config
        self.device = torch.device(config['runtime']['device'])
        torch.set_num_threads(config['runtime']['torch_threads'])
        torch.manual_seed(config['runtime']['seed'])
        torch.use_deterministic_algorithms(True)
        self.models, self.filters, self.metadata = {}, {}, {}
        for name, spec in config['models'].items():
            if only and name != only.split('/')[0]:
                continue
            manifest = json.loads(Path(spec['path'] + '.json').read_text())
            if manifest['architecture'] != {'feature': 'feature_nn', 'raw': 'raw_depth_nn'}[name]:
                raise ValueError('Classifier architecture mismatch')
            base = TerrainSelector(spec['path'], label_to_lora={k: CLASSES.index(v) for k, v in config['class_mapping'].items()}, **config['filters'])
            if base.input_shape != (48, 64) or len(base.class_ids) != 4:
                raise ValueError('Expected four-class 48x64 export')
            labels = [config['class_mapping'][x] for x in base.class_ids]
            if set(labels) != set(CLASSES):
                raise ValueError('Class mapping must be one-to-one for each model')
            self.models[name] = base.model.to(self.device)
            self.metadata[name] = {'sha256': digest(spec['path']), 'manifest_sha256': digest(spec['path'] + '.json'),
                                   'seed': spec['seed'], 'manifest': manifest, 'canonical_class_order': labels}
            selection_file = Path(spec['path']).parent / 'best_terrain_selectors.json'
            if selection_file.exists():
                selection = json.loads(selection_file.read_text())
                chosen = selection['models'][manifest['architecture']]
                if Path(chosen['model_path']).name == Path(spec['path']).name and chosen['seed'] != spec['seed']:
                    raise ValueError('Configured seed disagrees with export selection manifest')
                self.metadata[name]['export_selection'] = selection
                self.metadata[name]['export_selection_sha256'] = digest(selection_file)
            for mode in MODES:
                key = name + '/' + mode
                if only and key != only:
                    continue
                selector = copy.copy(base)
                selector.mode = mode
                selector.reset()
                self.filters[key] = selector

    def reset(self):
        for selector in self.filters.values():
            selector.reset()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def run(self, depth, rpy, omega):
        self.sync()
        start = time.perf_counter_ns()
        models, outputs = {}, {}
        for name, model in self.models.items():
            self.sync()
            t = time.perf_counter_ns()
            inputs = [torch.as_tensor(v, dtype=torch.float32, device=self.device).reshape(shape)
                      for v, shape in ((depth, (1, 48, 64)), (rpy, (1, 3)), (omega, (1, 3)))]
            logits = model(*inputs).reshape(-1)
            self.sync()
            logits = logits.cpu()
            if logits.numel() != 4 or not torch.isfinite(logits).all():
                raise ValueError('Classifier emitted invalid logits')
            probs = torch.softmax(logits, dim=0)
            models[name] = {'logits': logits.tolist(), 'probabilities': probs.tolist(),
                            'duration_ms': (time.perf_counter_ns() - t) / 1e6}
            for key, selector in self.filters.items():
                if not key.startswith(name + '/'):
                    continue
                t = time.perf_counter_ns()
                if selector.mode == 'ema':
                    selected = selector._ema(logits)
                    distribution = torch.softmax(selector.ema_logits, dim=0)
                elif selector.mode == 'bayes':
                    selected = selector._bayes(logits, probabilities=probs)
                    distribution = selector.belief
                else:
                    selected, distribution = int(logits.argmax()), probs
                outputs[key] = {'proposed_skill': self.metadata[name]['canonical_class_order'][selected],
                                'selected_index': selected, 'distribution': distribution.tolist(),
                                'ema_logits': selector.ema_logits.tolist() if selector.ema_logits is not None else None,
                                'pending_index': selector.pending_index, 'pending_count': selector.pending_count,
                                'duration_ms': (time.perf_counter_ns() - t) / 1e6}
        return {'models': models, 'selectors': outputs, 'selector_cycle_ms': (time.perf_counter_ns() - start) / 1e6}


class TrialWriter:
    def __init__(self, config, model_metadata, start_ns):
        self.config, self.pending, self.chunks = config, [], []
        parent = Path(config['output_root']) / config['experiment'] / config['configuration']
        parent.mkdir(parents=True, exist_ok=True)
        # Persistent exclusive claim disallows accidental reuse of a trial ID.
        claim = parent / ('.trial_' + str(config['trial_id']) + '.claim')
        with claim.open('x') as f:
            f.write(str(start_ns))
        name = 'rough_to_{}_{}_trial{}_{}'.format(config['target_class'], config['difficulty'], config['trial_id'], time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        self.path = parent / name
        self.path.mkdir(exist_ok=False)
        (self.path / 'resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        snapshot = self.path / 'source_snapshot'
        for source in [*Path(__file__).parent.glob('*.py'), ROOT / 'terrain_selector.py', ROOT / 'common/depth_processing.py', ROOT / 'rough_depth_image.py']:
            target = snapshot / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        self.events = (self.path / 'events.jsonl').open('a', buffering=1)
        self.event_lock = threading.Lock()
        self.manifest = {'schema': 'go2-shadow-v1', 'config': config,
                         'resolved_config_sha256': digest(self.path / 'resolved.yaml'), 'models': model_metadata,
                         'provenance': provenance(config), 'start_monotonic_ns': start_ns,
                         'start_wall_ns': time.time_ns(), 'complete': False,
                         'termination_reason': 'unclosed', 'chunks': [], 'frame_count': 0}
        atomic_json(self.path / 'manifest.json', self.manifest)
        self.event('reset', filters='all six reset', ground_truth='rough', start_ns=start_ns)

    def event(self, kind, **data):
        with self.event_lock:
            self.events.write(canonical({'kind': kind, 'logged_ns': time.monotonic_ns(), **data}) + '\n')
            self.events.flush()
            os.fsync(self.events.fileno())

    def add(self, record, raw, depth, rpy, omega, rgb=None):
        self.pending.append((record, raw.copy(), depth.copy(), np.array(rpy, np.float32), np.array(omega, np.float32), None if rgb is None else rgb.copy()))
        if len(self.pending) >= self.config['runtime']['chunk_frames']:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        name = 'frames_{:06d}.npz'.format(len(self.chunks))
        path = self.path / name
        arrays = {k: np.stack([r[i] for r in self.pending]) for k, i in [('raw_depth', 1), ('depth', 2), ('rpy', 3), ('omega', 4)]}
        arrays['records_utf8'] = np.frombuffer(canonical([r[0] for r in self.pending]).encode(), np.uint8)
        for i, row in enumerate(self.pending):
            if row[5] is not None:
                arrays['rgb_{:04d}'.format(i)] = row[5]
        with path.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path.with_suffix('.tmp'), path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.chunks.append({'file': name, 'sha256': digest(path), 'frames': len(self.pending)})
        self.manifest['chunks'] = self.chunks.copy()
        self.manifest['frame_count'] += len(self.pending)
        atomic_json(self.path / 'manifest.json', self.manifest)
        self.pending.clear()

    def close(self, reason, truth, counters):
        self.flush()
        self.manifest.update(complete=True, termination_reason=reason,
                             transition=truth.event, verified_crossing=truth.crossing,
                             approach_only=truth.event is None, counters=counters,
                             end_monotonic_ns=time.monotonic_ns())
        self.event('termination', reason=reason, counters=counters)
        atomic_json(self.path / 'manifest.json', self.manifest)
        self.events.close()


def read_trial(path):
    """Discover atomic chunks even if interrupted between rename and manifest update."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    indexed = {c['file']: c for c in manifest['chunks']}
    rows, issues = [], []
    if digest(path/'resolved.yaml') != manifest.get('resolved_config_sha256'):
        issues.append('resolved_config_hash_mismatch')
    if yaml.safe_load((path/'resolved.yaml').read_text()) != manifest['config']:
        issues.append('resolved_config_manifest_disagreement')
    if sum(c['frames'] for c in manifest['chunks']) != manifest['frame_count']:
        issues.append('manifest_frame_count_mismatch')
    if not manifest['complete']:
        issues.append('unclosed_trial')
    for item in sorted(path.glob('frames_*.npz')):
        if item.name in indexed and digest(item) != indexed[item.name]['sha256']:
            issues.append('corrupt_chunk:' + item.name)
            continue
        if item.name not in indexed:
            issues.append('recovered_unindexed_chunk:' + item.name)
        try:
            with np.load(item, allow_pickle=False) as data:
                records = json.loads(data['records_utf8'].tobytes())
                if item.name in indexed and len(records) != indexed[item.name]['frames']:
                    issues.append('chunk_frame_count_mismatch:' + item.name)
                if any(len(data[k]) != len(records) for k in ('raw_depth', 'depth', 'rpy', 'omega')):
                    raise ValueError('Chunk array lengths disagree with records')
                for i, record in enumerate(records):
                    rows.append((record, {k: data[k][i].copy() for k in ('raw_depth', 'depth', 'rpy', 'omega')},
                                 data['rgb_{:04d}'.format(i)].copy() if 'rgb_{:04d}'.format(i) in data else None))
        except (ValueError, OSError, KeyError) as error:
            issues.append('unreadable_chunk:' + item.name + ':' + str(error))
    event_path = path / 'events.jsonl'
    if event_path.exists():
        lines = event_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except ValueError:
                issues.append('incomplete_event_tail' if index == len(lines)-1 else 'corrupt_event_record')
                continue
            if event['kind'] == 'transition' and 'transition' not in manifest:
                manifest['transition'] = event
            if event['kind'] == 'transition_first_frame' and manifest.get('transition') and manifest['transition']['first_frame_id'] is None:
                manifest['transition']['first_frame_id'] = event['frame_id']
            if event['kind'] == 'crossing_verification' and 'verified_crossing' not in manifest:
                manifest['verified_crossing'] = event
    else:
        issues.append('missing_event_log')
    for name in indexed:
        if not (path / name).exists():
            issues.append('missing_chunk:' + name)
    if list(path.glob('*.tmp')):
        issues.append('unfinished_temporary_write')
    return manifest, rows, issues
