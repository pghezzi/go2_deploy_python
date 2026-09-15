"""Offline validation, deterministic replay, trial metrics, and paper artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import CLASSES, MODES, Pipelines, atomic_json, canonical, digest, preprocess, provenance, read_trial
from .reference_metrics import evaluate_transition_accounting, _false_transition_rate

COLORS = {'rough': '#777777', 'gap': '#0072B2', 'stairs': '#E69F00', 'climb': '#009E73'}
KEYS = [name + '/' + mode for name in ('feature', 'raw') for mode in MODES]


def validate(manifest, rows, issues):
    issues = list(issues)
    last_id, last_receipt, previous_label = None, None, 'rough'
    changes = 0
    c = manifest['config']
    event = manifest.get('transition')
    if event is None:
        events = [r[0].get('annotation_transition') for r in rows if r[0].get('annotation_transition')]
        event = events[0] if events else None
    for record, arrays, rgb in rows:
        fid, stamp = record['frame_id'], record['receipt_ns']
        if last_id is not None and (fid <= last_id or stamp <= last_receipt):
            issues.append('nonmonotonic_ids_or_timestamps')
        last_id, last_receipt = fid, stamp
        expected = c['target_class'] if event and stamp >= event['timestamp_ns'] else 'rough'
        if record['configured_target'] != c['target_class']:
            issues.append('configured_target_mismatch')
        if record['ground_truth'] != expected:
            issues.append('ground_truth_mismatch')
        if previous_label != record['ground_truth']:
            changes += 1
        previous_label = record['ground_truth']
        if set(record['selectors']) != set(KEYS) or set(record['models']) != {'feature', 'raw'}:
            issues.append('incomplete_six_pipeline_frame')
        if arrays['depth'].shape != (48, 64) or not np.isfinite(arrays['depth']).all() or np.any((arrays['depth'] < 0) | (arrays['depth'] > 1)):
            issues.append('invalid_processed_depth')
        if not np.array_equal(arrays['rpy'], np.asarray(record['state']['rpy'], np.float32)) or not np.array_equal(arrays['omega'], np.asarray(record['state']['omega'], np.float32)):
            issues.append('state_input_mismatch')
        if not 0 <= stamp - record['state']['receipt_ns'] <= c['runtime']['max_state_age_s'] * 1e9:
            issues.append('invalid_state_alignment')
        for key, output in record['selectors'].items():
            dist = np.asarray(output['distribution'])
            if dist.shape != (4,) or not np.isfinite(dist).all() or (dist < 0).any() or not np.isclose(dist.sum(), 1, atol=1e-5):
                issues.append('invalid_distribution')
            if output['proposed_skill'] not in CLASSES:
                issues.append('invalid_skill')
    if changes > 1:
        issues.append('multiple_ground_truth_changes')
    if not rows:
        issues.append('no_accepted_frames')
    return sorted(set(issues)), event


def replay(path, device=None, atol=1e-6, model_root=None):
    manifest, rows, issues = read_trial(path)
    issues, _ = validate(manifest, rows, issues)
    config = json.loads(json.dumps(manifest['config']))
    if device:
        config['runtime']['device'] = device
    if model_root:
        for spec in config['models'].values():
            spec['path'] = str(Path(model_root).resolve() / Path(spec['path']).name)
    pipelines = Pipelines(config)
    for name, metadata in pipelines.metadata.items():
        if metadata['sha256'] != manifest['models'][name]['sha256'] or metadata['manifest_sha256'] != manifest['models'][name]['manifest_sha256']:
            raise ValueError('Replay refuses mismatched model or manifest: ' + name)
    discrepancies = []
    max_abs = 0.0
    for record, arrays, _ in rows:
        result = pipelines.run(arrays['depth'], arrays['rpy'], arrays['omega'])
        frame_errors = []
        reconstructed = preprocess(arrays['raw_depth'], record['depth_scale_m'], config['camera'])
        if not np.allclose(reconstructed, arrays['depth'], atol=atol, rtol=0):
            frame_errors.append('preprocessed_depth')
        for name in ('feature', 'raw'):
            for field in ('logits', 'probabilities'):
                a, b = np.array(result['models'][name][field]), np.array(record['models'][name][field])
                max_abs = max(max_abs, float(np.max(np.abs(a-b))))
                if not np.allclose(a, b, atol=atol, rtol=0):
                    frame_errors.append(name + ':' + field)
        for key in KEYS:
            a, b = result['selectors'][key], record['selectors'][key]
            for field in ('distribution', 'ema_logits'):
                if a[field] is not None and not np.allclose(a[field], b[field], atol=atol, rtol=0):
                    frame_errors.append(key + ':' + field)
            for field in ('proposed_skill', 'selected_index', 'pending_index', 'pending_count'):
                if a[field] != b[field]:
                    frame_errors.append(key + ':' + field)
        if frame_errors:
            discrepancies.append({'frame_id': record['frame_id'], 'fields': frame_errors})
    return {'trial': str(path), 'frames': len(rows), 'consistent': bool(rows) and not discrepancies and not [i for i in issues if i not in ('unclosed_trial', 'unfinished_temporary_write') and not i.startswith('recovered_unindexed_chunk:')],
            'atol': atol, 'max_abs_classifier_error': max_abs, 'discrepancies': discrepancies,
            'validation_issues': issues, 'replay_provenance': provenance(config)}


def segments(rows, max_gap, exclusions=()):
    """Do not bridge invalid intervals or missing-frame gaps in temporal metrics."""
    valid, ids, segment, last_time, was_valid = [], [], -1, None, False
    previous_segment = None
    for i, (r, _, _) in enumerate(rows):
        if not r['valid_for_analysis']:
            was_valid = False
            continue
        crossed_exclusion = last_time is not None and any(a < r['elapsed_s'] and b > last_time for a,b in exclusions)
        if not was_valid or last_time is None or r['elapsed_s'] - last_time > max_gap or crossed_exclusion or r.get('temporal_segment') != previous_segment:
            segment += 1
        valid.append(i)
        ids.append(segment)
        last_time, was_valid = r['elapsed_s'], True
        previous_segment = r.get('temporal_segment')
    return valid, ids


def trial_metrics(manifest, rows, event, path):
    cfg = manifest['config']
    valid, ids = segments(rows, cfg['analysis']['max_contiguous_gap_s'], cfg['analysis']['excluded_intervals_s'])
    records = [rows[i][0] for i in valid]
    truth = [r['ground_truth'] for r in records]
    gt = np.array([CLASSES.index(x) for x in truth], dtype=int)
    result = []
    for key in KEYS:
        pred = [r['selectors'][key]['proposed_skill'] for r in records]
        predicted = np.array([CLASSES.index(x) for x in pred], dtype=int)
        confusion = np.zeros((4, 4), dtype=int)
        np.add.at(confusion, (gt, predicted), 1)
        support = confusion.sum(1)
        f1_denom = confusion.sum(1) + confusion.sum(0)
        f1 = np.divide(2*confusion.diagonal(), f1_denom, out=np.zeros(4,dtype=float), where=f1_denom>0)
        recalls = np.divide(confusion.diagonal(), support, out=np.zeros(4, dtype=float), where=support > 0)
        accounting = evaluate_transition_accounting(truth, pred, ids)
        transitions = accounting['transition_records_v2']
        # One configured transition; sequence starts never become new transitions.
        transition = transitions[0] if transitions else None
        delay_s = None
        if transition and transition['matched']:
            match = records[transition['matched_frame']]
            delay_s = (match['receipt_ns'] - event['timestamp_ns']) / 1e9 if event else None
        pairs = sum(ids[i] == ids[i-1] for i in range(1, len(ids)))
        false_count = sum(ids[i] == ids[i-1] and truth[i] == truth[i-1] and pred[i] != pred[i-1] for i in range(1, len(ids)))
        probs = np.array([r['selectors'][key]['distribution'] for r in records])
        name = key.split('/')[0]
        order = manifest['models'][name]['canonical_class_order']
        if len(records):
            probs = probs[:, [order.index(c) for c in CLASSES]]
            nll = float(-np.log(np.clip(probs[np.arange(len(gt)), gt], 1e-8, 1)).mean())
            brier = float(np.square(probs - np.eye(4)[gt]).sum(1).mean())
        else:
            nll = brier = None
        latency = [r['preprocess_ms'] + r['models'][name]['duration_ms'] + r['selectors'][key]['duration_ms'] for r in records]
        values = lambda field: [r[field] for r in records if r[field] is not None]
        percentile = lambda v, q: float(np.percentile(v, q)) if v else None
        result.append({'trial': str(path), 'trial_id': cfg['trial_id'], 'experiment': cfg['experiment'],
                       'configuration': cfg['configuration'], 'target': cfg['target_class'], 'difficulty': cfg['difficulty'],
                       'pipeline': key, 'accepted_frames': len(rows), 'valid_frames': len(records),
                       'capture_queue_drops': manifest.get('counters', {}).get('capture_queue_drop_new', 0),
                       'stale_host_frames': manifest.get('counters', {}).get('stale_host_input', 0),
                       'missing_or_stale_state_frames': manifest.get('counters', {}).get('missing_or_stale_state', 0),
                       'excluded_frames': len(rows)-len(records), 'accuracy': float((gt == predicted).mean()) if len(gt) else None,
                       'balanced_accuracy': float(recalls[support > 0].mean()) if len(gt) else None,
                       'macro_f1_supported_truth': float(f1[support>0].mean()) if len(gt) else None,
                       'rough_frames': int(sum(gt == 0)),
                       'premature_nonrough_frames': int(sum((gt == 0) & (predicted != 0))),
                       'nll': nll, 'brier': brier,
                       'false_transitions': false_count, 'adjacent_opportunities': pairs,
                       'false_transition_rate': _false_transition_rate(truth, pred, ids) if len(truth) >= 2 else None,
                       'annotation_present': event is not None, 'approach_only': event is None,
                       'verified_crossing': manifest.get('verified_crossing') is not None,
                       'transition_eligible': transition is not None,
                       'transition_exclusion': None if transition else ('approach_only' if event is None else 'no_valid_contiguous_pre_and_post_segment'),
                       'transition_missed': transition['missed'] if transition else None,
                       'first_match_delay_frames': transition['delay_classification_frames'] if transition and transition['matched'] else None,
                       'first_match_delay_s': delay_s,
                       'concurrent_component_p50_ms': percentile(latency, 50),
                       'concurrent_component_p95_ms': percentile(latency, 95),
                       'six_pipeline_cycle_p95_ms': percentile(values('six_pipeline_cycle_ms'), 95),
                       'update_interval_p95_ms': percentile(values('update_interval_ms'), 95),
                       'input_age_p95_ms': percentile(values('input_age_ms'), 95),
                       'deadline_misses': sum(r['deadline_miss'] for r in records),
                       'confusion': confusion.tolist()})
    return result


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def group_key(manifest):
    cfg = dict(manifest['config'])
    for key in ('trial_id', 'output_root', 'reference_repo'):
        cfg.pop(key, None)
    cfg['models'] = {name: {'sha256': m['sha256'], 'manifest_sha256': m['manifest_sha256']} for name, m in manifest['models'].items()}
    # Recording identifiers/paths don't change the physical setup; all settings do.
    return hashlib.sha256(canonical(cfg).encode()).hexdigest()[:12]


def aggregate(metrics):
    groups = {}
    for m in metrics:
        groups.setdefault((m['group'], m['pipeline']), []).append(m)
    result = []
    for (group, pipeline), trials in sorted(groups.items()):
        row = {'group': group, 'configuration': trials[0]['configuration'], 'pipeline': pipeline,
               'independent_trials': len(trials), 'approach_only_trials': sum(t['approach_only'] for t in trials),
               'annotated_trials': sum(t['annotation_present'] for t in trials),
               'verified_crossing_trials': sum(t['verified_crossing'] for t in trials),
               'eligible_transitions': sum(t['transition_eligible'] for t in trials),
               'missed_transitions': sum(t['transition_missed'] is True for t in trials),
               'excluded_frames': sum(t['excluded_frames'] for t in trials),
               'verified_eligible_transitions': sum(t['verified_crossing'] and t['transition_eligible'] for t in trials),
               'verified_missed_transitions': sum(t['verified_crossing'] and t['transition_missed'] is True for t in trials)}
        verified_delays = [t['first_match_delay_s'] for t in trials if t['verified_crossing'] and t['first_match_delay_s'] is not None]
        row['verified_annotation_delay_mean_s'] = float(np.mean(verified_delays)) if verified_delays else None
        row['verified_annotation_delay_std_s'] = float(np.std(verified_delays, ddof=1)) if len(verified_delays) > 1 else None
        for field in ('accuracy', 'balanced_accuracy', 'macro_f1_supported_truth', 'nll', 'brier', 'false_transition_rate', 'first_match_delay_s', 'concurrent_component_p95_ms'):
            v = [t[field] for t in trials if t[field] is not None]
            row[field + '_n'] = len(v)
            row[field + '_mean'] = float(np.mean(v)) if v else None
            row[field + '_std'] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        confusion = np.sum([t['confusion'] for t in trials], axis=0)
        row['pooled_confusion'] = confusion.tolist()
        result.append(row)
    return result


def plot_trial(path, manifest, rows, output, stem, failure=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    if not rows:
        return
    output = Path(output)
    records = [r[0] for r in rows]
    times = np.array([r['elapsed_s'] for r in records])
    labels = np.array([[CLASSES.index(r['ground_truth']) for r in records]] +
                      [[CLASSES.index(r['selectors'][k]['proposed_skill']) for r in records] for k in KEYS])
    valid = np.array([r['valid_for_analysis'] for r in records])
    position = np.array([r['position']['xyz'] if r.get('position') else [np.nan]*3 for r in records])
    # Deterministic selection: first accepted post-annotation frame, else midpoint.
    candidates = [i for i, r in enumerate(records) if r['ground_truth'] != 'rough']
    chosen = candidates[0] if candidates else len(rows)//2
    if failure:
        failures = [i for i, (r, _, _) in enumerate(rows) if r['valid_for_analysis'] and any(r['selectors'][k]['proposed_skill'] != r['ground_truth'] for k in KEYS)]
        if failures:
            chosen = failures[0]
    r, arrays, rgb = rows[chosen]
    np.savez_compressed(output / (stem + '_figure_data.npz'), elapsed_s=times, labels=labels,
                        valid=valid, position_m=position, depth=arrays['depth'],
                        rgb=rgb if rgb is not None else np.empty((0,), dtype=np.uint8), selected_frame_id=r['frame_id'])
    fig = plt.figure(figsize=(7.2, 4.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(arrays['depth'], vmin=0, vmax=1, cmap='gray')
    ax.set_title('Depth, frame {} ({:.2f} s)'.format(r['frame_id'], r['elapsed_s'])); ax.axis('off')
    ax = fig.add_subplot(grid[0, 1])
    if rgb is not None:
        ax.imshow(rgb); ax.set_title('Same frameset RGB (unregistered)')
    else:
        ax.text(.5, .5, 'RGB not recorded', ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    ax = fig.add_subplot(grid[1, :])
    cmap = ListedColormap([COLORS[c] for c in CLASSES])
    # Explicit edges retain nonuniform acquisition times; invalid/gap spans shaded.
    edges = np.r_[times, times[-1] + 1/manifest['config']['runtime']['update_hz']]
    ax.pcolormesh(edges, np.arange(8), labels, cmap=cmap, vmin=-.5, vmax=3.5, shading='flat')
    ax.set_yticks(np.arange(7)+.5); ax.set_yticklabels(['ground truth'] + KEYS)
    ax.invert_yaxis(); ax.set_xlabel('Host receipt time since trial start (s)')
    for i in range(len(times)):
        if not valid[i]:
            ax.axvspan(edges[i], edges[i+1], facecolor='white', alpha=.7, hatch='//')
        if i and times[i]-times[i-1] > manifest['config']['analysis']['max_contiguous_gap_s']:
            ax.axvspan(times[i-1]+1/manifest['config']['runtime']['update_hz'], times[i], color='white')
    event = manifest.get('transition')
    if event:
        ax.axvline((event['timestamp_ns']-manifest['start_monotonic_ns'])/1e9, color='black', linestyle='--', linewidth=.8)
    crossing = manifest.get('verified_crossing')
    if crossing:
        ax.axvline((crossing['timestamp_ns']-manifest['start_monotonic_ns'])/1e9, color='black', linestyle=':', linewidth=.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[c], label=c) for c in CLASSES], loc='upper center', bbox_to_anchor=(.5, -.27), ncol=4, frameon=False)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / (stem + '.' + suffix), dpi=220)
    plt.close(fig)
    if np.isfinite(position[:, 0]).any():
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4), constrained_layout=True, sharex=True, sharey=True)
        for ax, key in zip(axes.flat, KEYS):
            pred = [r['selectors'][key]['proposed_skill'] for r in records]
            ax.scatter(position[:, 0], [CLASSES.index(p) for p in pred], c=[COLORS[p] for p in pred], s=8)
            ax.set_title(key); ax.set_xlabel('Reported odometry x (m)'); ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
        for suffix in ('pdf', 'png'):
            fig.savefig(output / (stem + '_position.' + suffix), dpi=220)
        plt.close(fig)
    return {'trial': str(path), 'frame_id': r['frame_id'],
            'rule': 'first valid frame with a classifier error' if failure else 'first lexical eligible trial per group; first accepted post-annotation frame or midpoint if approach-only', 'files_stem': stem}


def report(root, output, appendix=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, validation, trials = [], [], []
    for file in sorted(Path(root).rglob('manifest.json')):
        if file.parent == output:
            continue
        try:
            if json.loads(file.read_text()).get('schema') != 'go2-shadow-v1':
                continue
            manifest, rows, issues = read_trial(file.parent)
            issues, event = validate(manifest, rows, issues)
            serious = [i for i in issues if i not in ('unclosed_trial', 'unfinished_temporary_write') and not i.startswith('recovered_unindexed_chunk:')]
            # Recovered trials remain replayable, but don't masquerade as complete repeats.
            eligible = not serious and manifest['complete'] and not manifest.get('counters', {}).get('capture_shutdown_timeout') and manifest.get('termination_reason') not in ('pipeline_error', 'source_or_processing_error')
            validation.append({'trial': str(file.parent), 'manifest_sha256': digest(file), 'issues': issues, 'aggregate_eligible': eligible})
            if rows and not serious:
                current = trial_metrics(manifest, rows, event, file.parent)
                for m in current:
                    m['group'] = group_key(manifest)
                    m['aggregate_eligible'] = eligible
                metrics.extend(current)
                trials.append((file.parent, manifest, rows, current))
        except (KeyError, ValueError, OSError) as error:
            validation.append({'trial': str(file.parent), 'issues': [repr(error)], 'aggregate_eligible': False})
    aggregated = aggregate([m for m in metrics if m['aggregate_eligible']])
    write_csv(output/'per_trial.csv', metrics)
    write_csv(output/'aggregate.csv', aggregated)
    atomic_json(output/'validation.json', validation)
    def tex_summary(row, field):
        mean, std = row[field + '_mean'], row[field + '_std']
        if mean is None:
            return '--'
        return '${:.3f} {}pm {:.3f}$'.format(mean, chr(92), std) if std is not None else '{:.3f}'.format(mean)
    with (output/'aggregate.tex').open('w') as f:
        f.write(chr(92) + 'begin{tabular}{llrrrrr}\n')
        f.write('Group & Pipeline & Trials & Accuracy & False rate & Delay (s) & Misses ' + chr(92)*2 + '\n')
        f.write(chr(92) + 'hline\n')
        for r in aggregated:
            fields = [r['group'], r['pipeline'].replace('_', chr(92)+'_'), str(r['independent_trials']),
                      tex_summary(r, 'accuracy'), tex_summary(r, 'false_transition_rate'),
                      tex_summary(r, 'first_match_delay_s'), '{}/{}'.format(r['missed_transitions'], r['eligible_transitions'])]
            f.write(' & '.join(fields) + ' ' + chr(92)*2 + '\n')
        f.write(chr(92) + 'end{tabular}\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selections, seen = [], set()
    for path, manifest, rows, current in trials:
        group = current[0]['group']
        if group not in seen and current[0]['aggregate_eligible']:
            selections.append(plot_trial(path, manifest, rows, output, 'representative_'+group))
            seen.add(group)
    for group in sorted({r['group'] for r in aggregated}):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), constrained_layout=True)
        data = {}
        for ax, key in zip(axes.flat, KEYS):
            row = next(r for r in aggregated if r['group'] == group and r['pipeline'] == key)
            matrix = np.asarray(row['pooled_confusion'])
            data[key.replace('/', '_')] = matrix
            norm = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros((4,4)), where=matrix.sum(1, keepdims=True)>0)
            ax.imshow(norm, vmin=0, vmax=1, cmap='Blues'); ax.set_title(key, fontsize=8)
            ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=35, fontsize=7)
            ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES, fontsize=7)
            ax.set_xlabel('Proposed skill', fontsize=7); ax.set_ylabel('Ground truth', fontsize=7)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=7, color='white' if norm[i,j]>.5 else 'black')
        np.savez_compressed(output/('confusion_'+group+'_data.npz'), **data)
        for suffix in ('pdf','png'):
            fig.savefig(output/('confusion_'+group+'.'+suffix), dpi=220)
        plt.close(fig)
    if appendix and trials:
        candidates = [t for t in trials if t[3][0]['aggregate_eligible'] and any(m['accuracy'] is not None and m['accuracy'] < 1 for m in t[3])]
        if candidates:
            chosen = min(candidates, key=lambda t: (np.mean([m['accuracy'] for m in t[3] if m['accuracy'] is not None]), str(t[0])))
            selection = plot_trial(*chosen[:3], output, 'appendix_failure', failure=True)
            selection['trial_selection_rule'] = 'lowest mean accuracy across six pipelines among eligible trials with errors; lexical path tie-break'
            selections.append(selection)
    atomic_json(output/'figure_selection.json', selections)
    atomic_json(output/'report_manifest.json', {'schema':'go2-shadow-report-v1', 'primary_record':'recorded online outputs',
                'input_trials': validation, 'aggregation': 'equal weight per independent trial; sample std (ddof=1); pooled confusion separately',
                'reference_transition_metric':'segment_bounded_v2', 'classes':CLASSES, 'colors':COLORS,
                'analysis_source_sha256':digest(__file__), 'appendix_requested':appendix})
    print('Report:', output, 'Trials:', len(validation), 'eligible:', sum(v['aggregate_eligible'] for v in validation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('report'); p.add_argument('root'); p.add_argument('--output', required=True); p.add_argument('--appendix-failure', action='store_true')
    p = sub.add_parser('replay'); p.add_argument('trial'); p.add_argument('--output', required=True); p.add_argument('--device'); p.add_argument('--atol', type=float, default=1e-6); p.add_argument('--model-root')
    args = parser.parse_args()
    if args.command == 'report':
        report(args.root, args.output, args.appendix_failure)
    else:
        result = replay(args.trial, args.device, args.atol, args.model_root)
        atomic_json(args.output, result)
        print('Replay consistent:', result['consistent'])
        if not result['consistent']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
