from typing import Dict, List, Set, Union
import numpy as np
import pandas as pd


def compute_instance_f05(
    true_set: Set[str], pred_set: Set[str], beta: float = 0.5
) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity following the competition specification.
    
    Precision Bias: F_0.5 weights Precision 2x over Recall.
    Formula:
        F_0.5 = ( (1 + beta^2) * Precision * Recall ) / ( beta^2 * Precision + Recall )
        where beta = 0.5:
        F_0.5 = ( 1.25 * Precision * Recall ) / ( 0.25 * Precision + Recall )
        
    The Singleton Rule:
    - If true_set is empty (Source 1 entity has no true matches across S2/S3):
        - If pred_set is empty -> score is 1.0
        - If pred_set is non-empty -> score is 0.0
    - If true_set is non-empty:
        - If pred_set is empty -> score is 0.0
        - If TP == 0 -> score is 0.0
    """
    # Singleton check
    if not true_set:
        return 1.0 if not pred_set else 0.0
    
    if not pred_set:
        return 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0

    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = beta ** 2  # 0.25
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Computes Macro F_0.5 averaged across all Source 1 entities in ground_truth_dict.
    Also returns singleton F_0.5 and non-singleton F_0.5 diagnostics.
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        score = compute_instance_f05(true_set, pred_set)
        scores.append(score)
        if not true_set:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    singleton_f05 = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_f05": singleton_f05,
        "non_singleton_f05": non_singleton_f05,
        "total_evaluated": len(scores),
        "singleton_count": len(singleton_scores),
        "non_singleton_count": len(non_singleton_scores),
    }


def parse_id_list(id_string: Union[str, float, None]) -> Set[str]:
    """
    Parses a comma-separated entity ID string into a set of clean IDs.
    Returns empty set if NaN or empty string.
    """
    if id_string is None or pd.isna(id_string):
        return set()
    s = str(id_string).strip()
    if not s:
        return set()
    return {item.strip() for item in s.split(",") if item.strip()}
