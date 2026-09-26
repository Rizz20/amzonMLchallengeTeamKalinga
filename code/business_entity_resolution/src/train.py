"""
train.py — LightGBM model training for Business Entity Resolution.

Workflow:
  1. Load & split training data (80/20, stratified by country)
  2. Run blocking on train split to get candidates
  3. Build (pair, label) training data
  4. Fit TF-IDF vectorizers
  5. Compute all features
  6. Train LightGBM classifier
  7. Sweep threshold on validation set to maximize F₀.₅
  8. Save model + vectorizers + best threshold to disk
"""

import sys
import os
import pickle
import random
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from utils.io_utils import load_source, load_ground_truth
from utils.metrics import f05_macro, sweep_threshold
from blocking import run_blocking, evaluate_blocking
from features import (
    compute_features_batch, fit_tfidf_vectorizers,
    build_training_pairs, FEATURE_NAMES, N_FEATURES,
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Data paths (relative to student_resource/)
    "train_s1": "dataset/train/train_source1.tsv",
    "train_s2": "dataset/train/train_source2.tsv",
    "train_s3": "dataset/train/train_source3.tsv",
    "train_gt": "dataset/train/train_ground_truth.tsv",

    # Output paths
    "model_dir": "models/",

    # Training
    "val_fraction": 0.20,        # fraction of S1 entities for validation
    "random_seed": 42,
    "neg_per_pos": 3,            # hard negatives per positive pair
    "max_candidates": 500,       # blocking cap per S1 entity
    "batch_size": 200_000,       # pairs per feature-computation batch

    # LightGBM hyperparameters
    "lgbm_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": 3,   # compensate for class imbalance (neg_per_pos)
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    },

    # Threshold sweep
    "threshold_sweep": [round(i * 0.05, 2) for i in range(2, 20)],  # 0.10..0.95
}


# ──────────────────────────────────────────────────────────────────────────────
# Train/validation split
# ──────────────────────────────────────────────────────────────────────────────

def train_val_split(
    s1_records: List[dict],
    ground_truth: Dict[str, List[str]],
    val_fraction: float = 0.20,
    random_seed: int = 42,
) -> Tuple[List[dict], List[dict], Dict, Dict]:
    """
    Stratified split of S1 records into train and validation sets.
    Stratify by country to ensure France-like country distribution.
    """
    rng = random.Random(random_seed)

    # Group by country
    by_country: Dict[str, List[dict]] = {}
    for row in s1_records:
        country = row.get("country", "unknown")
        by_country.setdefault(country, []).append(row)

    train_records, val_records = [], []
    for country, rows in by_country.items():
        rng.shuffle(rows)
        split_idx = max(1, int(len(rows) * (1 - val_fraction)))
        train_records.extend(rows[:split_idx])
        val_records.extend(rows[split_idx:])

    train_ids = {r["entity_id"] for r in train_records}
    val_ids = {r["entity_id"] for r in val_records}

    train_gt = {k: v for k, v in ground_truth.items() if k in train_ids}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_ids}

    print(f"Train: {len(train_records):,} entities | Val: {len(val_records):,} entities")
    return train_records, val_records, train_gt, val_gt


# ──────────────────────────────────────────────────────────────────────────────
# Feature matrix construction
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    pairs: List[Tuple[dict, dict]],
    name_vec=None,
    addr_vec=None,
    batch_size: int = 200_000,
    verbose: bool = True,
) -> np.ndarray:
    """Compute feature matrix in batches (memory-safe for large pair sets)."""
    from tqdm import tqdm

    n = len(pairs)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    for start in tqdm(range(0, n, batch_size), desc="Feature batches", unit="batch"):
        end = min(start + batch_size, n)
        batch = pairs[start:end]
        X[start:end] = compute_features_batch(
            batch,
            tfidf_name_vectorizer=name_vec,
            tfidf_addr_vectorizer=addr_vec,
            verbose=False,
        )

    return X


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    model_dir: str,
    config: Optional[dict] = None,
) -> dict:
    """
    Full training pipeline.

    Args:
        data_dir:  Path to student_resource/ directory.
        model_dir: Where to save model artifacts.
        config:    Override config values (merged with DEFAULT_CONFIG).

    Returns:
        dict with training results: best_threshold, val_f05, model path, etc.
    """
    import lightgbm as lgb

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    s1 = load_source(os.path.join(data_dir, cfg["train_s1"]))
    s2 = load_source(os.path.join(data_dir, cfg["train_s2"]))
    s3 = load_source(os.path.join(data_dir, cfg["train_s3"]))
    gt = load_ground_truth(os.path.join(data_dir, cfg["train_gt"]))
    print(f"S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")

    # ── 2. Train / val split ─────────────────────────────────────────────────
    print("\n=== Train/val split ===")
    train_s1, val_s1, train_gt, val_gt = train_val_split(
        s1, gt,
        val_fraction=cfg["val_fraction"],
        random_seed=cfg["random_seed"],
    )

    # ── 3. Blocking on train split ───────────────────────────────────────────
    print("\n=== Blocking (train) ===")
    train_candidates = run_blocking(
        train_s1, s2, s3,
        max_candidates=cfg["max_candidates"],
        use_ngram_fallback=True,
        verbose=True,
    )
    blocking_stats = evaluate_blocking(train_candidates, train_gt)
    print(f"Blocking recall (train): {blocking_stats['blocking_recall']:.4f}")
    print(f"Avg candidates/entity:  {blocking_stats['avg_candidates_per_entity']:.1f}")

    # ── 4. Blocking on val split ─────────────────────────────────────────────
    print("\n=== Blocking (val) ===")
    val_candidates = run_blocking(
        val_s1, s2, s3,
        max_candidates=cfg["max_candidates"],
        use_ngram_fallback=True,
        verbose=True,
    )
    val_blocking_stats = evaluate_blocking(val_candidates, val_gt)
    print(f"Blocking recall (val):  {val_blocking_stats['blocking_recall']:.4f}")

    # ── 5. Fit TF-IDF vectorizers ────────────────────────────────────────────
    print("\n=== Fitting TF-IDF vectorizers ===")
    all_records = s1 + s2 + s3
    name_vec, addr_vec = fit_tfidf_vectorizers(all_records, max_features=50_000)
    print(f"Name vocab size: {len(name_vec.vocabulary_):,}")
    print(f"Addr vocab size: {len(addr_vec.vocabulary_):,}")

    # ── 6. Build training pairs & compute features ───────────────────────────
    print("\n=== Building training pairs ===")
    s23_records = s2 + s3
    train_pairs, train_labels = build_training_pairs(
        train_s1, s2, s3,
        train_candidates, train_gt,
        neg_per_pos=cfg["neg_per_pos"],
        random_seed=cfg["random_seed"],
        verbose=True,
    )
    print(f"Train pairs: {len(train_pairs):,} ({sum(train_labels):,} positive, {len(train_labels)-sum(train_labels):,} negative)")

    print("\n=== Computing training features ===")
    X_train = build_feature_matrix(
        train_pairs, name_vec, addr_vec, batch_size=cfg["batch_size"]
    )
    y_train = np.array(train_labels, dtype=np.int32)

    # ── 7. Build validation pairs & features ─────────────────────────────────
    print("\n=== Building validation pairs ===")
    # For validation: score ALL blocking candidates (not just sampled negatives)
    val_pairs = []
    val_pair_labels = []
    val_pair_meta = []   # (s1_id, cand_id) for score aggregation

    s23_map = {r["entity_id"]: r for r in s23_records}
    s1_map = {r["entity_id"]: r for r in val_s1}

    for s1_row in val_s1:
        s1_id = s1_row["entity_id"]
        true_matches = set(val_gt.get(s1_id, []))
        for cid in val_candidates.get(s1_id, []):
            cand_row = s23_map.get(cid)
            if cand_row:
                val_pairs.append((s1_row, cand_row))
                val_pair_labels.append(1 if cid in true_matches else 0)
                val_pair_meta.append((s1_id, cid))

    print(f"Val pairs: {len(val_pairs):,}")
    print("\n=== Computing validation features ===")
    X_val = build_feature_matrix(
        val_pairs, name_vec, addr_vec, batch_size=cfg["batch_size"]
    )
    y_val = np.array(val_pair_labels, dtype=np.int32)

    # ── 8. Train LightGBM ───────────────────────────────────────────────────
    print("\n=== Training LightGBM ===")
    lgbm_params = cfg["lgbm_params"].copy()
    n_estimators = lgbm_params.pop("n_estimators", 500)
    learning_rate = lgbm_params.pop("learning_rate", 0.05)

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        **lgbm_params,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )
    print(f"Best iteration: {model.best_iteration_}")

    # Feature importance
    importance = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 features:")
    for fname, imp in importance[:10]:
        print(f"  {fname}: {imp:.1f}")

    # ── 9. Threshold sweep on validation set ─────────────────────────────────
    print("\n=== Threshold sweep ===")
    val_scores = model.predict_proba(X_val)[:, 1]

    # Build scores_by_entity: {s1_id: [(cand_id, score), ...]}
    scores_by_entity: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, cand_id), score in zip(val_pair_meta, val_scores):
        scores_by_entity.setdefault(s1_id, []).append((cand_id, float(score)))

    sweep_result = sweep_threshold(scores_by_entity, val_gt, thresholds=cfg["threshold_sweep"])
    best_threshold = sweep_result["best_threshold"]
    best_f05 = sweep_result["best_f05"]
    print(f"Best threshold: {best_threshold:.2f} → Val F₀.₅: {best_f05:.4f}")
    print("\nThreshold sweep results:")
    for r in sweep_result["threshold_results"]:
        print(f"  thresh={r['threshold']:.2f}  F₀.₅={r['f05']:.4f}")

    # ── 10. Save artifacts ───────────────────────────────────────────────────
    print("\n=== Saving model artifacts ===")
    model_path = os.path.join(model_dir, "lgbm_model.pkl")
    name_vec_path = os.path.join(model_dir, "tfidf_name.pkl")
    addr_vec_path = os.path.join(model_dir, "tfidf_addr.pkl")
    config_path = os.path.join(model_dir, "training_config.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(name_vec_path, "wb") as f:
        pickle.dump(name_vec, f)
    with open(addr_vec_path, "wb") as f:
        pickle.dump(addr_vec, f)

    training_results = {
        "best_threshold": best_threshold,
        "val_f05": best_f05,
        "blocking_recall_train": blocking_stats["blocking_recall"],
        "blocking_recall_val": val_blocking_stats["blocking_recall"],
        "avg_candidates_train": blocking_stats["avg_candidates_per_entity"],
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "best_lgbm_iteration": model.best_iteration_,
        "feature_importance": {name: int(imp) for name, imp in importance},
        "threshold_sweep": sweep_result["threshold_results"],
    }
    with open(config_path, "w") as f:
        json.dump(training_results, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Config saved to: {config_path}")
    print(f"\n✅ Final val F₀.₅: {best_f05:.4f} at threshold {best_threshold:.2f}")

    return training_results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train entity resolution model")
    parser.add_argument("--data-dir", default=".", help="Path to student_resource/ directory")
    parser.add_argument("--model-dir", default="models/", help="Directory to save model artifacts")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=500)
    args = parser.parse_args()

    results = train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        config={
            "val_fraction": args.val_fraction,
            "neg_per_pos": args.neg_per_pos,
            "max_candidates": args.max_candidates,
        },
    )
    print(f"\nDone. Val F₀.₅: {results['val_f05']:.4f}")
