import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.blocking import run_blocking, DenseRetriever
from src.config import (
    DATASET_DIR,
    LGBM_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    THRESHOLD_SWEEP_RANGE,
    TRAIN_DIR,
    VAL_SPLIT_RATIO,
    SPARSE_TOP_K,
    DENSE_TOP_K,
)
from src.features import FEATURE_COLUMNS, extract_pairwise_features
from src.metrics import compute_macro_f05, parse_id_list
from src.preprocess import load_tsv, preprocess_dataframe


def train_pipeline(
    train_dir: Path = TRAIN_DIR,
    models_dir: Path = MODELS_DIR,
    val_split_ratio: float = VAL_SPLIT_RATIO,
) -> Dict:
    """
    Executes the training and validation pipeline:
    1. Loads and preprocesses training sources & ground truth.
    2. Splits S1 entities by country into Train and Validation sets.
    3. Runs dual-channel blocking.
    4. Computes rich pairwise features.
    5. Trains LightGBM classifier with imbalanced pair handling.
    6. Sweeps decision threshold tau to directly maximize Macro F_0.5.
    7. Saves trained model and metadata.
    """
    print("=" * 60)
    print("Starting Business Entity Resolution Training Pipeline")
    print("=" * 60)

    # 1. Load data
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    print(f"Loading datasets from {train_dir}...")
    s1_raw = load_tsv(str(s1_path))
    s2_raw = load_tsv(str(s2_path))
    s3_raw = load_tsv(str(s3_path))
    gt_raw = load_tsv(str(gt_path))

    # Parse ground truth mapping
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_raw.iterrows():
        s1_id = row["source1_entity_id"]
        gt_map[s1_id] = parse_id_list(row.get("matched_entity_ids", ""))

    print(f"Source 1 records: {len(s1_raw)}")
    print(f"Source 2 records: {len(s2_raw)}")
    print(f"Source 3 records: {len(s3_raw)}")
    print(f"Ground Truth records: {len(gt_raw)}")

    # 2. Preprocess text
    print("\nPreprocessing entity names, addresses, and country-specific scripts...")
    s1_df = preprocess_dataframe(s1_raw)
    s2_df = preprocess_dataframe(s2_raw)
    s3_df = preprocess_dataframe(s3_raw)

    # 3. Train/Validation Split (Grouped by S1 entity to avoid data leakage)
    all_s1_ids = s1_df["entity_id"].tolist()
    s1_countries = s1_df["country"].tolist()

    # Stratify by country if possible
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids,
        test_size=val_split_ratio,
        random_state=RANDOM_SEED,
        stratify=s1_countries if len(set(s1_countries)) > 1 else None,
    )
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    train_s1_df = s1_df[s1_df["entity_id"].isin(train_s1_set)].reset_index(drop=True)
    val_s1_df = s1_df[s1_df["entity_id"].isin(val_s1_set)].reset_index(drop=True)

    print(f"Train S1 Entities: {len(train_s1_df)} | Validation S1 Entities: {len(val_s1_df)}")

    # 4. Candidate Generation / Blocking
    print("\nRunning dual-channel blocking (TF-IDF 3-gram + Dense SentenceTransformer)...")
    dense_retriever = DenseRetriever()

    # Train blocking
    print("Generating training candidate pairs...")
    train_pairs_df, s1_embs_train, cand_embs_train = run_blocking(
        train_s1_df, s2_df, s3_df, dense_retriever=dense_retriever
    )
    print(f"Generated {len(train_pairs_df)} training pairs.")

    # Validation blocking
    print("Generating validation candidate pairs...")
    val_pairs_df, s1_embs_val, cand_embs_val = run_blocking(
        val_s1_df, s2_df, s3_df, dense_retriever=dense_retriever
    )
    print(f"Generated {len(val_pairs_df)} validation pairs.")

    # Check candidate recall on validation set
    val_gt_cands = 0
    val_retrieved_gt = 0
    for s1_id in val_s1_ids:
        true_matches = gt_map.get(s1_id, set())
        val_gt_cands += len(true_matches)
        if s1_id in val_pairs_df["source1_entity_id"].values:
            retrieved = set(
                val_pairs_df[val_pairs_df["source1_entity_id"] == s1_id][
                    "candidate_entity_id"
                ].tolist()
            )
            val_retrieved_gt += len(true_matches.intersection(retrieved))
    
    cand_recall = (val_retrieved_gt / val_gt_cands) if val_gt_cands > 0 else 1.0
    print(f"Validation Candidate Recall Ceiling: {cand_recall * 100:.2f}% ({val_retrieved_gt}/{val_gt_cands})")

    # 5. Feature Engineering
    print("\nExtracting pairwise features for training pairs...")
    train_feats_df = extract_pairwise_features(
        train_pairs_df, train_s1_df, s2_df, s3_df, s1_embs_train, cand_embs_train
    )
    # Add target label
    train_labels = [
        1 if row["candidate_entity_id"] in gt_map.get(row["source1_entity_id"], set()) else 0
        for _, row in train_feats_df.iterrows()
    ]
    train_feats_df["target"] = train_labels

    print("Extracting pairwise features for validation pairs...")
    val_feats_df = extract_pairwise_features(
        val_pairs_df, val_s1_df, s2_df, s3_df, s1_embs_val, cand_embs_val
    )
    val_labels = [
        1 if row["candidate_entity_id"] in gt_map.get(row["source1_entity_id"], set()) else 0
        for _, row in val_feats_df.iterrows()
    ]
    val_feats_df["target"] = val_labels

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = (n_neg / max(1, n_pos))
    print(f"Training pairs class balance: {n_pos} positive, {n_neg} negative (scale_pos_weight: {pos_weight:.2f})")

    # 6. Train LightGBM Model
    print("\nTraining LightGBM Classifier...")
    X_train = train_feats_df[FEATURE_COLUMNS]
    y_train = train_feats_df["target"]
    X_val = val_feats_df[FEATURE_COLUMNS]
    y_val = val_feats_df["target"]

    lgbm_params = dict(LGBM_PARAMS)
    lgbm_params["scale_pos_weight"] = min(pos_weight, 10.0)  # Bound to avoid extreme precision drop
    lgbm_params["min_child_samples"] = max(2, min(20, len(X_train) // 4))

    model = lgb.LGBMClassifier(**lgbm_params)
    if len(np.unique(y_val)) > 1 and len(y_val) >= 10:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    # 7. Decision Threshold Optimization for Macro F_0.5
    print("\nOptimizing decision threshold tau directly on Macro F_0.5...")
    val_preds_prob = model.predict_proba(X_val)[:, 1]
    val_feats_df["pred_prob"] = val_preds_prob

    # Prepare ground truth dictionary for validation S1 entities
    val_gt_dict = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}

    best_tau = 0.80
    best_macro_f05 = -1.0
    best_diagnostics = {}

    for tau in THRESHOLD_SWEEP_RANGE:
        pred_dict: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_s1_ids}
        high_prob_rows = val_feats_df[val_feats_df["pred_prob"] >= tau]
        for _, row in high_prob_rows.iterrows():
            pred_dict[row["source1_entity_id"]].add(row["candidate_entity_id"])

        diag = compute_macro_f05(val_gt_dict, pred_dict)
        if diag["macro_f05"] >= best_macro_f05:
            best_macro_f05 = diag["macro_f05"]
            best_tau = tau
            best_diagnostics = diag

    print(f"Optimal Threshold (tau*): {best_tau}")
    print(f"Best Validation Macro F_0.5: {best_macro_f05:.4f}")
    print(f"Singleton F_0.5: {best_diagnostics.get('singleton_f05', 0.0):.4f}")
    print(f"Non-Singleton F_0.5: {best_diagnostics.get('non_singleton_f05', 0.0):.4f}")

    # 8. Save Model and Configuration
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "lgbm_model.pkl"
    joblib.dump(model, model_file)
    print(f"\nSaved trained model to {model_file}")

    config_info = {
        "optimal_threshold": best_tau,
        "best_macro_f05": best_macro_f05,
        "singleton_f05": best_diagnostics.get("singleton_f05", 0.0),
        "non_singleton_f05": best_diagnostics.get("non_singleton_f05", 0.0),
        "feature_columns": FEATURE_COLUMNS,
        "sparse_top_k": SPARSE_TOP_K,
        "dense_top_k": DENSE_TOP_K,
    }
    config_file = models_dir / "model_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    print(f"Saved configuration to {config_file}")

    return config_info


if __name__ == "__main__":
    train_pipeline()
