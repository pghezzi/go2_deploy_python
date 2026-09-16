"""Controlled classifier -> filter latency, one pipeline at a time; no camera/control."""
import argparse
import copy
import gc
from pathlib import Path
import time

import numpy as np
import torch

from .core import Pipelines, atomic_json, digest, provenance, read_trial
from .analyze import KEYS, write_csv


def benchmark(trial, output, iterations=1000, warmup=50, device='cpu', model_root=None):
    if iterations < 2 or warmup < 0:
        raise ValueError('Require iterations >= 2 and warmup >= 0')
    manifest, rows, issues = read_trial(trial)
    serious = [i for i in issues if i not in ('unclosed_trial', 'unfinished_temporary_write') and not i.startswith('recovered_unindexed_chunk:')]
    if serious or not rows:
        raise ValueError('Invalid saved inputs: ' + str(serious or ['no frames']))
    inputs = [(r['frame_id'], a['depth'], a['rpy'], a['omega']) for r, a, _ in rows]
    if any(d.shape != (48,64) or not all(np.isfinite(v).all() for v in (d,r,w)) for _,d,r,w in inputs):
        raise ValueError('Expected finite, saved normalized 48x64 depth and state')
    config = copy.deepcopy(manifest['config'])
    config['runtime'].update(device=device, torch_threads=1)
    if model_root:
        for spec in config['models'].values():
            spec['path'] = str(Path(model_root).resolve()/Path(spec['path']).name)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run = {'complete': False, 'scope': 'isolated sequential classifier -> filter, batch size one; includes tensor conversion, device transfers, output validation and result packaging; excludes camera, RealSense filters, resizing, disk I/O, model loading and specialist execution',
           'iterations_per_pipeline': iterations, 'warmup_per_pipeline': warmup,
           'input_policy': 'cycle through identical saved inputs in original order for every pipeline',
           'filter_policy': 'reset after warmup and between pipelines; retain state across measured calls',
           'std_convention': 'sample standard deviation (ddof=1)',
           'clock': 'perf_counter_ns wall time; CUDA synchronized before and after calls',
           'trial': str(Path(trial).resolve()), 'trial_manifest_sha256': digest(Path(trial)/'manifest.json'),
           'source_issues': issues, 'pipeline_order': list(KEYS), 'models': manifest['models'],
           'provenance': provenance(config)}
    atomic_json(output/'manifest.json', run)
    summaries = []
    for key in KEYS:
        pipeline = Pipelines(config, only=key)
        name = key.split('/')[0]
        for field in ('sha256', 'manifest_sha256'):
            if pipeline.metadata[name][field] != manifest['models'][name][field]:
                raise ValueError('Saved model mismatch: ' + name)
        for i in range(warmup):
            _, depth, rpy, omega = inputs[i % len(inputs)]
            pipeline.run(depth, rpy, omega)
        pipeline.reset()
        samples = []
        for i in range(iterations):
            fid, depth, rpy, omega = inputs[i % len(inputs)]
            pipeline.sync()
            start = time.perf_counter_ns()
            result = pipeline.run(depth, rpy, omega)
            pipeline.sync()
            elapsed = (time.perf_counter_ns()-start)/1e6
            samples.append({'pipeline': key, 'iteration': i, 'frame_id': fid,
                            'total_ms': elapsed, 'classifier_ms': result['models'][name]['duration_ms'],
                            'filter_ms': result['selectors'][key]['duration_ms']})
        summary = {'pipeline': key, 'samples': iterations}
        for field in ('total_ms', 'classifier_ms', 'filter_ms'):
            values = [s[field] for s in samples]
            summary[field+'_mean'] = float(np.mean(values))
            summary[field+'_std'] = float(np.std(values, ddof=1))
            summary[field+'_p95'] = float(np.percentile(values,95))
        summaries.append(summary)
        write_csv(output/(key.replace('/','_')+'_samples.csv'), samples)
        write_csv(output/'summary.csv', summaries)
        print('{}: {:.3f} +/- {:.3f} ms (n={})'.format(key, summary['total_ms_mean'], summary['total_ms_std'], iterations), flush=True)
        del pipeline
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    run.update(complete=True, provenance=provenance(config))
    atomic_json(output/'manifest.json', run)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial')
    p.add_argument('--output', required=True)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--device', default='cpu')
    p.add_argument('--model-root')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    benchmark(args.trial, args.output, args.iterations, args.warmup, args.device, args.model_root)


if __name__ == '__main__':
    main()
