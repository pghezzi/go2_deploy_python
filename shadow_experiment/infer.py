"""Generate a separate offline-inference trial from a sensor-only recording."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .core import Pipelines, atomic_json, canonical, digest, preprocess, provenance, read_trial, replay_depth_input


def infer(source, output, device='cpu', model_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest, rows, issues = read_trial(source)
    if not manifest['config']['runtime'].get('record_only', False):
        raise ValueError('Expected a recording-only trial')
    serious = [x for x in issues if x not in ('unclosed_trial', 'unfinished_temporary_write') and not x.startswith('recovered_unindexed_chunk:')]
    if serious or not rows:
        raise ValueError('Cannot infer trial: ' + str(serious or ['no frames']))
    config = copy.deepcopy(manifest['config'])
    config['runtime']['device'] = device
    if model_root:
        for spec in config['models'].values():
            spec['path'] = str(Path(model_root).resolve()/Path(spec['path']).name)
    pipelines = Pipelines(config)
    for name, metadata in pipelines.metadata.items():
        for field in ('sha256', 'manifest_sha256'):
            if metadata[field] != manifest['models'][name][field]:
                raise ValueError('Model mismatch: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('resolved.yaml', 'events.jsonl'):
        if (source/name).exists():
            shutil.copy2(source/name, output/name)
    derived = copy.deepcopy(manifest)
    derived.update(models=pipelines.metadata, chunks=[], frame_count=0, complete=False,
                   execution_mode='offline_inference', timing_scope='offline host inference; not robot selector latency',
                   source_trial=str(source), source_manifest_sha256=digest(source/'manifest.json'),
                   source_issues=issues, inference_provenance=provenance(config))
    atomic_json(output/'manifest.json', derived)
    for index, (record, arrays, rgb) in enumerate(rows):
        start = time.perf_counter_ns()
        depth = arrays['depth'] # Exact normalized deployment input saved by the publisher
        preprocessing_ms = (time.perf_counter_ns()-start)/1e6
        result = pipelines.run(depth, arrays['rpy'], arrays['omega'])
        record = dict(record, **result)
        record.update(execution_mode='offline_inference', preprocess_ms=preprocessing_ms,
                      six_pipeline_cycle_ms=(time.perf_counter_ns()-start)/1e6)
        arrays = dict(arrays, depth=depth)
        data = {k: np.expand_dims(v, 0) for k,v in arrays.items()}
        data['records_utf8'] = np.frombuffer(canonical([record]).encode(), np.uint8)
        if rgb is not None:
            data['rgb_0000'] = rgb
        name = 'frames_{:06d}.npz'.format(index)
        tmp = output/(name+'.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **data)
        tmp.replace(output/name)
        derived['chunks'].append({'file': name, 'frames': 1, 'sha256': digest(output/name)})
        derived['frame_count'] += 1
        atomic_json(output/'manifest.json', derived)
    derived['complete'] = manifest['complete']
    atomic_json(output/'manifest.json', derived)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-root')
    args = parser.parse_args()
    print(infer(args.trial, args.output, args.device, args.model_root))


if __name__ == '__main__':
    main()
