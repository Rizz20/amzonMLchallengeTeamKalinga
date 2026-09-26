"""
metrics.py — F₀.₅ scorer for Business Entity Resolution.

F₀.₅ is precision-heavy (weights precision 2× over recall).
Computed as macro-average across all Source 1 entities.
"""
from typing import Dict, List, Optional


def f05_per_entity(pred: List[str], truth: List[str]) -> float:
    """
    Compute F₀.₅ for a single Source 1 entity.

    Args:
        pred:  List of predicted matched entity IDs (may be empty).
        truth: List of true matched entity IDs (may be empty).

    Returns:
        F₀.₅ score in [0, 1].
    """
    pred_set = set(pred)
    truth_set = set(truth)

    # Singleton: both empty → perfect score
    if not pred_set and not truth_set:
        return 1.0

    # One empty, other not → 0
    if not pred_set or not truth_set:
        return 0.0

    tp = len(pred_set & truth_set)

    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(truth_set)

    # F_β = (1 + β²) × P × R / (β² × P + R), β = 0.5
    beta_sq = 0.25  # β² = 0.5² = 0.25
    f05 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    return f05


def f05_macro(
    pred_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]],
) -> float:
    """
    Compute macro-average F₀.₅ across all Source 1 entities in truth_dict.

    Args:
        pred_dict:  {s1_entity_id: [matched_ids, ...]} — your predictions.
        truth_dict: {s1_entity_id: [matched_ids, ...]} — ground truth.

    Returns:
        Macro-average F₀.₅ in [0, 1].
    """
    scores = []
    for s1_id, true_matches in truth_dict.items():
        pred_matches = pred_dict.get(s1_id, [])
        scores.append(f05_per_entity(pred_matches, true_matches))

    return sum(scores) / len(scores) if scores else 0.0


def precision_recall_f05(
    pred_dict: Dict[str, List[str]],
    truth_dict: Dict[str, List[str]],
) -> dict:
    """
    Detailed breakdown: per-entity F₀.₅ + aggregate precision/recall/F₀.₅.
    Useful for error analysis and threshold tuning.
    """
    total_tp = 0
    total_pred = 0
    total_truth = 0
    entity_scores = {}

    for s1_id, true_matches in truth_dict.items():
        pred_matches = pred_dict.get(s1_id, [])
        pred_set = set(pred_matches)
        truth_set = set(true_matches)

        tp = len(pred_set & truth_set)
        total_tp += tp
        total_pred += len(pred_set)
        total_truth += len(truth_set)

        entity_scores[s1_id] = f05_per_entity(pred_matches, true_matches)

    micro_precision = total_tp / total_pred if total_pred > 0 else 0.0
    micro_recall = total_tp / total_truth if total_truth > 0 else 0.0
    beta_sq = 0.25
    micro_f05 = (
        (1 + beta_sq) * micro_precision * micro_recall
        / (beta_sq * micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )
    macro_f05 = sum(entity_scores.values()) / len(entity_scores) if entity_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "micro_f05": micro_f05,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "total_tp": total_tp,
        "total_pred": total_pred,
        "total_truth": total_truth,
        "n_entities": len(entity_scores),
        "entity_scores": entity_scores,  # per-entity breakdown for error analysis
    }


def sweep_threshold(
    scores_by_entity: Dict[str, List[tuple]],
    truth_dict: Dict[str, List[str]],
    thresholds: Optional[List[float]] = None,
) -> dict:
    """
    Sweep prediction threshold and return best threshold + F₀.₅.

    Args:
        scores_by_entity: {s1_id: [(candidate_id, score), ...]} — ranked candidates.
        truth_dict: ground truth dict.
        thresholds: list of thresholds to try. Defaults to 0.1..0.95 in steps of 0.05.

    Returns:
        dict with 'best_threshold', 'best_f05', 'threshold_results'.
    """
    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(2, 20)]  # 0.10 to 0.95

    threshold_results = []
    best_threshold = 0.5
    best_f05 = 0.0

    for thresh in thresholds:
        pred_dict = {}
        for s1_id, candidates in scores_by_entity.items():
            pred_dict[s1_id] = [cid for cid, score in candidates if score >= thresh]

        result = f05_macro(pred_dict, truth_dict)
        threshold_results.append({"threshold": thresh, "f05": result})

        if result > best_f05:
            best_f05 = result
            best_threshold = thresh

    return {
        "best_threshold": best_threshold,
        "best_f05": best_f05,
        "threshold_results": threshold_results,
    }
