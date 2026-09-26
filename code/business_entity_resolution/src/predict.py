"""
predict.py — Inference pipeline for Business Entity Resolution.

Loads the trained model + vectorizers, runs blocking on the test set,
computes features, scores pairs, applies threshold, and writes:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv

Also runs validate_submission.py at the end for local sanity check.
"""

import sys
import os
import pickle
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from utils.io_utils import load_source, write_matching_results, write_candidate_pairs
from blocking import run_blocking
from features import compute_features_batch, N_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# Prediction pipeline
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    data_dir: str,
    model_dir: str,
    output_dir: str,
    threshold: Optional[float] = None,
    max_candidates: int = 500,
    batch_size: int = 200_000,
    verbose: bool = True,
) -> None:
    """
    Full inference pipeline.

    Args:
        data_dir:       Path to student_resource/ directory.
        model_dir:      Directory containing saved model artifacts.
        output_dir:     Directory to write output TSVs.
        threshold:      Match score threshold (loaded from training_config.json if None).
        max_candidates: Blocking cap per S1 entity.
        batch_size:     Pairs per feature-computation batch.
        verbose:        Print progress.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model artifacts ─────────────────────────────────────────────────
    if verbose:
        print("\n=== Loading model artifacts ===")

    model_path = os.path.join(model_dir, "lgbm_model.pkl")
    name_vec_path = os.path.join(model_dir, "tfidf_name.pkl")
    addr_vec_path = os.path.join(model_dir, "tfidf_addr.pkl")
    config_path = os.path.join(model_dir, "training_config.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(name_vec_path, "rb") as f:
        name_vec = pickle.load(f)
    with open(addr_vec_path, "rb") as f:
        addr_vec = pickle.load(f)

    if threshold is None:
        with open(config_path) as f:
            training_cfg = json.load(f)
        threshold = training_cfg["best_threshold"]
        if verbose:
            print(f"Using threshold from training: {threshold:.2f} (val F₀.₅ = {training_cfg['val_f05']:.4f})")
    else:
        if verbose:
            print(f"Using provided threshold: {threshold:.2f}")

    # ── Load test data ───────────────────────────────────────────────────────
    if verbose:
        print("\n=== Loading test data ===")

    test_s1 = load_source(os.path.join(data_dir, "dataset/test/test_source1.tsv"))
    test_s2 = load_source(os.path.join(data_dir, "dataset/test/test_source2.tsv"))
    test_s3 = load_source(os.path.join(data_dir, "dataset/test/test_source3.tsv"))

    if verbose:
        print(f"Test S1: {len(test_s1):,} | S2: {len(test_s2):,} | S3: {len(test_s3):,}")

    all_s1_ids = [r["entity_id"] for r in test_s1]

    # ── Blocking ─────────────────────────────────────────────────────────────
    if verbose:
        print("\n=== Blocking (test) ===")

    candidates = run_blocking(
        test_s1, test_s2, test_s3,
        max_candidates=max_candidates,
        use_ngram_fallback=True,
        verbose=verbose,
    )

    total_candidates = sum(len(v) for v in candidates.values())
    if verbose:
        print(f"Total candidate pairs: {total_candidates:,}")
        print(f"Avg candidates per entity: {total_candidates / max(1, len(test_s1)):.1f}")

    # ── Score all candidate pairs ─────────────────────────────────────────────
    if verbose:
        print("\n=== Scoring candidate pairs ===")

    s23_map = {r["entity_id"]: r for r in test_s2 + test_s3}
    s1_map = {r["entity_id"]: r for r in test_s1}

    # Build flat list of pairs + metadata
    pair_meta = []   # [(s1_id, cand_id), ...]
    pairs = []       # [(s1_row, cand_row), ...]

    for s1_row in test_s1:
        s1_id = s1_row["entity_id"]
        for cid in candidates.get(s1_id, []):
            cand_row = s23_map.get(cid)
            if cand_row:
                pairs.append((s1_row, cand_row))
                pair_meta.append((s1_id, cid))

    if verbose:
        print(f"Pairs to score: {len(pairs):,}")

    # Compute features and score in batches
    all_scores = []
    from tqdm import tqdm
    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring batches"):
        end = min(start + batch_size, len(pairs))
        batch = pairs[start:end]
        X_batch = compute_features_batch(
            batch,
            tfidf_name_vectorizer=name_vec,
            tfidf_addr_vectorizer=addr_vec,
            verbose=False,
        )
        scores = model.predict_proba(X_batch)[:, 1]
        all_scores.extend(scores.tolist())

    # ── Apply threshold and build predictions ─────────────────────────────────
    if verbose:
        print(f"\n=== Applying threshold {threshold:.2f} ===")

    predictions: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

    for (s1_id, cand_id), score in zip(pair_meta, all_scores):
        if score >= threshold:
            predictions[s1_id].append(cand_id)

    n_predicted = sum(len(v) for v in predictions.values())
    n_entities_with_matches = sum(1 for v in predictions.values() if v)
    n_singletons = len(all_s1_ids) - n_entities_with_matches

    if verbose:
        print(f"Entities with ≥1 match predicted: {n_entities_with_matches:,}")
        print(f"Singletons predicted: {n_singletons:,}")
        print(f"Total match predictions: {n_predicted:,}")

    # ── Write output files ────────────────────────────────────────────────────
    if verbose:
        print("\n=== Writing output files ===")

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    write_matching_results(predictions, matching_path, all_s1_ids=all_s1_ids)
    write_candidate_pairs(candidates, candidate_path, all_s1_ids=all_s1_ids)

    if verbose:
        print(f"✅ Matching results: {matching_path}")
        print(f"✅ Candidate pairs:  {candidate_path}")

    # ── Validate submission ───────────────────────────────────────────────────
    if verbose:
        print("\n=== Validating submission ===")

    validator_path = os.path.join(data_dir, "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, validator_path,
                "--matching", matching_path,
                "--candidate", candidate_path,
                "--test-dir", os.path.join(data_dir, "dataset/test"),
            ],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("⚠️  VALIDATION FAILED:")
            print(result.stderr)
        else:
            print("✅ Validation PASSED — safe to submit!")
    else:
        print(f"⚠️  Validator not found at {validator_path}. Skipping validation.")


# ──────────────────────────────────────────────────────────────────────────────
# Training-set self-evaluation (for debugging / overfitting check)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_train_sample(
    data_dir: str,
    model_dir: str,
    threshold: Optional[float] = None,
    sample_size: int = 10_000,
    random_seed: int = 42,
) -> dict:
    """
    Evaluate the trained model on a random sample of training entities.
    Useful for checking if blocking recall is the bottleneck.
    """
    import random
    from utils.io_utils import load_ground_truth
    from utils.metrics import precision_recall_f05
    from blocking import evaluate_blocking

    print("=== Training-set evaluation (sample) ===")

    s1 = load_source(os.path.join(data_dir, "dataset/train/train_source1.tsv"))
    s2 = load_source(os.path.join(data_dir, "dataset/train/train_source2.tsv"))
    s3 = load_source(os.path.join(data_dir, "dataset/train/train_source3.tsv"))
    gt = load_ground_truth(os.path.join(data_dir, "dataset/train/train_ground_truth.tsv"))

    rng = random.Random(random_seed)
    sample_s1 = rng.sample(s1, min(sample_size, len(s1)))

    # Load model
    with open(os.path.join(model_dir, "lgbm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "tfidf_name.pkl"), "rb") as f:
        name_vec = pickle.load(f)
    with open(os.path.join(model_dir, "tfidf_addr.pkl"), "rb") as f:
        addr_vec = pickle.load(f)

    if threshold is None:
        with open(os.path.join(model_dir, "training_config.json")) as f:
            threshold = json.load(f)["best_threshold"]

    # Block
    candidates = run_blocking(sample_s1, s2, s3, verbose=True)
    sample_gt = {r["entity_id"]: gt.get(r["entity_id"], []) for r in sample_s1}
    blocking_stats = evaluate_blocking(candidates, sample_gt)
    print(f"Blocking recall: {blocking_stats['blocking_recall']:.4f}")

    # Score
    s23_map = {r["entity_id"]: r for r in s2 + s3}
    pairs, pair_meta = [], []
    for s1_row in sample_s1:
        s1_id = s1_row["entity_id"]
        for cid in candidates.get(s1_id, []):
            cand_row = s23_map.get(cid)
            if cand_row:
                pairs.append((s1_row, cand_row))
                pair_meta.append((s1_id, cid))

    X = compute_features_batch(pairs, name_vec, addr_vec, verbose=True)
    scores = model.predict_proba(X)[:, 1]

    predictions = {r["entity_id"]: [] for r in sample_s1}
    for (s1_id, cid), score in zip(pair_meta, scores):
        if score >= threshold:
            predictions[s1_id].append(cid)

    result = precision_recall_f05(predictions, sample_gt)
    print(f"\nSample F₀.₅ (macro): {result['macro_f05']:.4f}")
    print(f"Micro Precision:      {result['micro_precision']:.4f}")
    print(f"Micro Recall:         {result['micro_recall']:.4f}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference for entity resolution")
    parser.add_argument("--data-dir", default=".", help="Path to student_resource/")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--output-dir", default="output/")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Match score threshold (default: from training_config.json)")
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--eval-train-sample", action="store_true",
                        help="Also evaluate on a training sample")
    args = parser.parse_args()

    predict(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
    )

    if args.eval_train_sample:
        evaluate_on_train_sample(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            threshold=args.threshold,
        )
