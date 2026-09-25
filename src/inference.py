import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import joblib
import numpy as np
import pandas as pd

from src.blocking import DenseRetriever, export_candidate_pairs_tsv, run_blocking
from src.config import (
    DEFAULT_THRESHOLD,
    MODELS_DIR,
    OUTPUT_DIR,
    TEST_DIR,
)
from src.features import FEATURE_COLUMNS, extract_pairwise_features
from src.preprocess import load_tsv, preprocess_dataframe


def run_inference(
    test_dir: Path = TEST_DIR,
    models_dir: Path = MODELS_DIR,
    output_dir: Path = OUTPUT_DIR,
    custom_threshold: float = None,
) -> Tuple[Path, Path]:
    """
    Executes the end-to-end inference pipeline on test data:
    1. Ingestion & Preprocessing with multilingual normalization.
    2. Country-isolated dual-channel candidate retrieval (blocking).
    3. Exports output/candidate_pairs.tsv.
    4. Feature extraction.
    5. LightGBM scoring & threshold filtering (Singleton rule handling).
    6. Strict subset validation and export of output/matching_results.tsv.
    """
    print("=" * 60)
    print("Starting Business Entity Resolution Inference Pipeline")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_out_path = output_dir / "candidate_pairs.tsv"
    matching_out_path = output_dir / "matching_results.tsv"

    # 1. Load test files
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print(f"Loading test data from {test_dir}...")
    s1_raw = load_tsv(str(s1_path))
    s2_raw = load_tsv(str(s2_path))
    s3_raw = load_tsv(str(s3_path))

    all_test_s1_ids = s1_raw["entity_id"].tolist()
    print(f"Test Source 1 entities: {len(s1_raw)}")
    print(f"Test Source 2 entities: {len(s2_raw)}")
    print(f"Test Source 3 entities: {len(s3_raw)}")

    # 2. Preprocess text
    print("\nPreprocessing test names and addresses (Unicode NFKD, script maps, French tokens)...")
    s1_df = preprocess_dataframe(s1_raw)
    s2_df = preprocess_dataframe(s2_raw)
    s3_df = preprocess_dataframe(s3_raw)

    # 3. Blocking / Candidate Retrieval
    print("\nRunning dual-channel candidate retrieval (Sparse TF-IDF + Dense Multilingual)...")
    dense_retriever = DenseRetriever()
    candidate_pairs_df, s1_embs, cand_embs = run_blocking(
        s1_df, s2_df, s3_df, dense_retriever=dense_retriever
    )
    print(f"Retrieved {len(candidate_pairs_df)} total candidate pairs.")

    # 4. Export output/candidate_pairs.tsv
    print(f"\nWriting candidate pairs to {candidate_out_path}...")
    export_candidate_pairs_tsv(
        candidate_pairs_df,
        all_s1_ids=all_test_s1_ids,
        output_path=str(candidate_out_path),
    )

    # Fast lookup for candidate validation: s1_id -> set of candidate IDs
    valid_cands_map: Dict[str, Set[str]] = {s1: set() for s1 in all_test_s1_ids}
    if not candidate_pairs_df.empty:
        for _, row in candidate_pairs_df.iterrows():
            valid_cands_map[row["source1_entity_id"]].add(row["candidate_entity_id"])

    # 5. Load model and config
    model_file = models_dir / "lgbm_model.pkl"
    config_file = models_dir / "model_config.json"

    tau = DEFAULT_THRESHOLD
    feature_cols = FEATURE_COLUMNS

    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            tau = cfg.get("optimal_threshold", DEFAULT_THRESHOLD)
            feature_cols = cfg.get("feature_columns", FEATURE_COLUMNS)
            print(f"Loaded optimal threshold tau = {tau:.2f} from {config_file}")

    if custom_threshold is not None:
        tau = custom_threshold
        print(f"Overridden threshold tau = {tau:.2f}")

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Please run training first."
        )

    model = joblib.load(model_file)
    print(f"Loaded trained LightGBM model from {model_file}")

    # 6. Feature Extraction & Scoring
    matched_results_map: Dict[str, List[str]] = {s1: [] for s1 in all_test_s1_ids}

    if not candidate_pairs_df.empty:
        print("\nExtracting pairwise features for test candidate pairs...")
        test_feats_df = extract_pairwise_features(
            candidate_pairs_df, s1_df, s2_df, s3_df, s1_embs, cand_embs
        )

        X_test = test_feats_df[feature_cols]
        print("Scoring candidate pairs with LightGBM...")
        probs = model.predict_proba(X_test)[:, 1]
        test_feats_df["pred_prob"] = probs

        # Filter by threshold tau
        high_prob_pairs = test_feats_df[test_feats_df["pred_prob"] >= tau]

        for _, row in high_prob_pairs.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            
            # Strict subset enforcement: Cand ID MUST be in candidate_pairs for that S1
            if cand_id in valid_cands_map[s1_id]:
                if cand_id not in matched_results_map[s1_id]:
                    matched_results_map[s1_id].append(cand_id)

    # 7. Write output/matching_results.tsv
    print(f"\nWriting matching results to {matching_out_path}...")
    matched_rows = []
    singleton_count = 0
    matched_count = 0

    for s1_id in all_test_s1_ids:
        matches = matched_results_map.get(s1_id, [])
        if not matches:
            singleton_count += 1
            matched_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": "",
            })
        else:
            matched_count += 1
            matched_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matches),
            })

    matching_df = pd.DataFrame(matched_rows)
    matching_df.to_csv(
        str(matching_out_path),
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )

    print("\nInference Summary:")
    print(f"Total Test S1 Entities: {len(all_test_s1_ids)}")
    print(f"Entities with predicted matches: {matched_count} ({matched_count/len(all_test_s1_ids)*100:.1f}%)")
    print(f"Entities predicted as singletons: {singleton_count} ({singleton_count/len(all_test_s1_ids)*100:.1f}%)")
    print(f"Candidate file saved to: {candidate_out_path}")
    print(f"Matching file saved to: {matching_out_path}")

    return candidate_out_path, matching_out_path


if __name__ == "__main__":
    run_inference()
