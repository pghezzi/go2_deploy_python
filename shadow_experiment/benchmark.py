"""Isolated batch-one selector benchmark on saved inputs; no hardware control."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import Pipelines, atomic_json, preprocess, provenance, read_trial
from .analyze import KEYS, write_csv


def benchmark(trial, output, repeats=3, warmup=10, device=None, model_root=None):
    manifest, rows, issues = read_trial(trial)
    if not rows:
        raise ValueError('Trial has no saved inputs')
    if repeats < 1 or warmup < 0:
        raise ValueError('Invalid repetition count')
    config = json.loads(json.dumps(manifest['config']))
    if device:
        config['runtime']['device'] = device
    if model_root:
        for spec in config['models'].values():
            spec['path'] = str(Path(model_root).resolve() / Path(spec['path']).name)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = []
    for key in KEYS:
        pipelines = Pipelines(config, only=key)
        name = key.split('/')[0]
        if pipelines.metadata[name]['sha256'] != manifest['models'][name]['sha256']:
            raise ValueError('Saved model hash mismatch')
        for i in range(warmup):
            r, arrays, _ = rows[i % len(rows)]
            pipelines.run(preprocess(arrays['raw_depth'], r['depth_scale_m'], config['camera']), arrays['rpy'], arrays['omega'])
        for repeat in range(repeats):
            pipelines.reset()
            for r, arrays, _ in rows:
                pipelines.sync()
                start = time.perf_counter_ns()
                depth = preprocess(arrays['raw_depth'], r['depth_scale_m'], config['camera'])
                pre_ms = (time.perf_counter_ns() - start)/1e6
                result = pipelines.run(depth, arrays['rpy'], arrays['omega'])
                pipelines.sync()
                records.append({'pipeline': key, 'repeat': repeat, 'frame_id': r['frame_id'],
                                'preprocessing_ms': pre_ms, 'classifier_ms': result['models'][name]['duration_ms'],
                                'filter_ms': result['selectors'][key]['duration_ms'],
                                'total_ms': (time.perf_counter_ns()-start)/1e6})
    write_csv(output/'isolated_samples.csv', records)
    summary = []
    for key in KEYS:
        v = [r['total_ms'] for r in records if r['pipeline'] == key]
        summary.append({'pipeline':key, 'samples':len(v), 'mean_ms':float(np.mean(v)),
                        'p50_ms':float(np.percentile(v,50)), 'p95_ms':float(np.percentile(v,95)),
                        'std_ms':float(np.std(v))})
    write_csv(output/'isolated_summary.csv', summary)
    atomic_json(output/'manifest.json', {'scope':'isolated sequential pipeline, batch size one, saved input arrays already in RAM; no acquisition, disk reads, or specialist execution',
                'trial':str(trial), 'repeats':repeats, 'warmup_frames':warmup, 'provenance':provenance(config), 'trial_issues':issues,
                'gpu_timing':'synchronize before/after each model; CPU wall clock includes transfers', 'models':manifest['models']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial'); p.add_argument('--output',required=True); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmup',type=int,default=10); p.add_argument('--device'); p.add_argument('--model-root')
    a=p.parse_args(); benchmark(a.trial,a.output,a.repeats,a.warmup,a.device,a.model_root)


if __name__ == '__main__':
    main()
