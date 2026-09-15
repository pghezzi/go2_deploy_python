"""Vendored reporting-only metrics, unchanged from Legged_Gym_EX.
Source: legged_gym/utils/depth_terrain_classifier/
terrain_classifier_bayes_streaming_prototype_rbf.py
Commit: d41b0e6103d0386f91de53dece47080f922fc781
Kept import-independent for offline machines without IsaacGym.
"""
from typing import Sequence, Hashable, Optional, Any
TRANSITION_ACCOUNTING_VERSION = "segment_bounded_v2"

def evaluate_transition_accounting(
    truth: Sequence[Hashable],
    prediction: Sequence[Hashable],
    sequence_ids: Optional[Sequence[Any]] = None,
) -> dict:
    """Reporting-only v2: first match in [transition, target segment end).

    Frame indices are offsets in the supplied classification stream. Sequence
    starts are not transitions. Misses remain explicit, with unavailable delay.
    Legacy delay/search objectives intentionally retain their original behavior.
    """
    truth, prediction = list(truth), list(prediction)
    ids = list(sequence_ids) if sequence_ids is not None else [0] * len(truth)
    if len(prediction) != len(truth) or len(ids) != len(truth):
        raise ValueError("truth, prediction and sequence_ids must have equal lengths")
    records = []
    start, sequence_start = 0, 0
    while start < len(truth):
        end = start + 1
        while end < len(truth) and ids[end] == ids[start] and truth[end] == truth[start]:
            end += 1
        if start == 0 or ids[start] != ids[start - 1]:
            sequence_start = start
        else:
            match = next((t for t in range(start, end) if prediction[t] == truth[start]), None)
            records.append({
                "sequence_id": ids[start], "sequence_start_frame": sequence_start,
                "transition_frame": start, "sequence_transition_frame": start - sequence_start,
                "segment_end_frame_exclusive": end,
                "from_label": truth[start - 1], "target_label": truth[start],
                "matched": match is not None, "missed": match is None,
                "matched_frame": match,
                "delay_classification_frames": match - start if match is not None else float("nan"),
            })
        start = end
    delays = [r["delay_classification_frames"] for r in records if r["matched"]]
    total, matched = len(records), len(delays)
    return {
        "transition_metric_version": TRANSITION_ACCOUNTING_VERSION,
        "total_transitions_v2": total,
        "matched_transitions_v2": matched,
        "missed_transitions_v2": total - matched,
        "transition_miss_rate_v2": (total - matched) / total if total else float("nan"),
        "mean_matched_transition_delay_frames_v2": sum(delays) / matched if matched else float("nan"),
        "transition_records_v2": records,
    }

def _false_transition_rate(
    truth: Sequence[Hashable],
    prediction: Sequence[Hashable],
    sequence_ids: Optional[Sequence[Any]],
) -> float:
    if len(truth) < 2:
        return float("nan")
    ids = list(sequence_ids) if sequence_ids is not None else [0] * len(truth)
    false_changes, opportunities = 0, 0
    for t in range(1, len(truth)):
        if ids[t] != ids[t - 1]:
            continue
        opportunities += 1
        if prediction[t] != prediction[t - 1] and truth[t] == truth[t - 1]:
            false_changes += 1
    return false_changes / max(opportunities, 1)
