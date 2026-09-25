import argparse
import csv
from pathlib import Path
import sys
from typing import List, Set
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict Submission Validator for Business Entity Resolution ML Challenge."
    )
    parser.add_argument(
        "--matching",
        type=str,
        required=True,
        help="Path to output/matching_results.tsv",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to output/candidate_pairs.tsv",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Path to dataset/test directory containing test_source1.tsv",
    )
    return parser.parse_args()


def load_tsv_file(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        print(f"[FAIL] File not found: {filepath}")
        sys.exit(1)
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
        return df
    except Exception as e:
        print(f"[FAIL] Error parsing {filepath} with sep='\\t' and quoting=csv.QUOTE_NONE: {e}")
        sys.exit(1)


def parse_id_tokens(id_str: str) -> List[str]:
    if not id_str or not isinstance(id_str, str):
        return []
    s = id_str.strip()
    if not s:
        return []
    return [token.strip() for token in s.split(",") if token.strip()]


def validate_submission(matching_file: str, candidate_file: str, test_dir: str):
    print("=" * 65)
    print("SUBMISSION VERIFICATION & COMPLIANCE CHECK")
    print("=" * 65)

    matching_path = Path(matching_file)
    candidate_path = Path(candidate_file)
    test_s1_path = Path(test_dir) / "test_source1.tsv"

    # 1. Existence and loading
    print(f"Checking test ground truth reference: {test_s1_path}")
    s1_df = load_tsv_file(test_s1_path)
    if "entity_id" not in s1_df.columns:
        print("[FAIL] 'entity_id' column missing from test_source1.tsv")
        sys.exit(1)
    expected_s1_ids = s1_df["entity_id"].tolist()
    total_test_s1 = len(expected_s1_ids)
    print(f"[PASS] Loaded {total_test_s1} test Source 1 entities.")

    print(f"\nChecking candidate file: {candidate_path}")
    cand_df = load_tsv_file(candidate_path)
    print(f"[PASS] Candidate file successfully parsed.")

    print(f"\nChecking matching file: {matching_path}")
    match_df = load_tsv_file(matching_path)
    print(f"[PASS] Matching file successfully parsed.")

    # 2. Header and Column Verification
    expected_cand_cols = ["source1_entity_id", "candidate_entity_ids"]
    if list(cand_df.columns) != expected_cand_cols:
        print(f"[FAIL] candidate_pairs.tsv columns must be {expected_cand_cols}, got {list(cand_df.columns)}")
        sys.exit(1)
    print(f"[PASS] Candidate file headers verified: {expected_cand_cols}")

    expected_match_cols = ["source1_entity_id", "matched_entity_ids"]
    if list(match_df.columns) != expected_match_cols:
        print(f"[FAIL] matching_results.tsv columns must be {expected_match_cols}, got {list(match_df.columns)}")
        sys.exit(1)
    print(f"[PASS] Matching file headers verified: {expected_match_cols}")

    # 3. Row count and 1-to-1 Correspondence Check
    if len(cand_df) != total_test_s1:
        print(f"[FAIL] Candidate file row count ({len(cand_df)}) does not match test_source1.tsv ({total_test_s1})")
        sys.exit(1)
    print(f"[PASS] Candidate file row count matches exactly: {total_test_s1}")

    if len(match_df) != total_test_s1:
        print(f"[FAIL] Matching file row count ({len(match_df)}) does not match test_source1.tsv ({total_test_s1})")
        sys.exit(1)
    print(f"[PASS] Matching file row count matches exactly: {total_test_s1}")

    cand_s1_ids = cand_df["source1_entity_id"].tolist()
    match_s1_ids = match_df["source1_entity_id"].tolist()

    if cand_s1_ids != expected_s1_ids:
        print("[FAIL] Candidate file source1_entity_id list or order differs from test_source1.tsv")
        sys.exit(1)
    print("[PASS] Candidate file entity ID order perfectly aligned with test_source1.tsv")

    if match_s1_ids != expected_s1_ids:
        print("[FAIL] Matching file source1_entity_id list or order differs from test_source1.tsv")
        sys.exit(1)
    print("[PASS] Matching file entity ID order perfectly aligned with test_source1.tsv")

    # 4. Content Validation & Strict Subset Constraint
    print("\nValidating entity ID formatting, prefix constraints, and strict subset rule...")
    
    total_candidates = 0
    total_matches = 0
    singletons = 0
    subset_violations = []
    prefix_violations = []
    duplicate_violations = []

    for i in range(total_test_s1):
        s1_id = expected_s1_ids[i]
        c_str = cand_df.iloc[i]["candidate_entity_ids"]
        m_str = match_df.iloc[i]["matched_entity_ids"]

        cand_tokens = parse_id_tokens(c_str)
        match_tokens = parse_id_tokens(m_str)

        total_candidates += len(cand_tokens)
        total_matches += len(match_tokens)

        if not match_tokens:
            singletons += 1

        # Check duplicates in candidate list
        if len(cand_tokens) != len(set(cand_tokens)):
            duplicate_violations.append((s1_id, "candidate_pairs", cand_tokens))

        # Check duplicates in matched list
        if len(match_tokens) != len(set(match_tokens)):
            duplicate_violations.append((s1_id, "matching_results", match_tokens))

        # Check prefixes: only 'S2-' or 'S3-' allowed
        for tid in cand_tokens:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                prefix_violations.append((s1_id, "candidate", tid))
        for tid in match_tokens:
            if not (tid.startswith("S2-") or tid.startswith("S3-")):
                prefix_violations.append((s1_id, "match", tid))

        # Strict Subset Rule: match_set <= cand_set
        cand_set = set(cand_tokens)
        match_set = set(match_tokens)
        violating_ids = match_set - cand_set
        if violating_ids:
            subset_violations.append((s1_id, list(violating_ids)))

    if duplicate_violations:
        print(f"[FAIL] Found duplicate IDs in output rows:")
        for v in duplicate_violations[:5]:
            print(f"  - S1: {v[0]} in {v[1]}")
        sys.exit(1)
    print("[PASS] No duplicate entity IDs found within any row.")

    if prefix_violations:
        print(f"[FAIL] Found invalid entity ID prefixes (must be S2- or S3-):")
        for v in prefix_violations[:5]:
            print(f"  - S1: {v[0]}, type: {v[1]}, invalid ID: {v[2]}")
        sys.exit(1)
    print("[PASS] All entity IDs have valid 'S2-' or 'S3-' prefixes.")

    if subset_violations:
        print(f"[FAIL] STRICT SUBSET RULE VIOLATION! Matched entity IDs not present in candidate_pairs:")
        for v in subset_violations[:10]:
            print(f"  - S1: {v[0]}, Invalid matched IDs: {v[1]}")
        sys.exit(1)
    print("[PASS] Strict subset rule 100% verified (Every matched ID is in candidate_pairs).")

    # 5. Summary Statistics
    avg_cands = total_candidates / total_test_s1 if total_test_s1 else 0
    avg_matches = total_matches / total_test_s1 if total_test_s1 else 0
    print("\n" + "=" * 65)
    print("SUBMISSION VERIFICATION SUCCESSFUL (EXIT CODE 0)")
    print("=" * 65)
    print(f"Total Test Source 1 Entities : {total_test_s1}")
    print(f"Total Candidate Pairs        : {total_candidates} (Avg {avg_cands:.2f} per entity)")
    print(f"Total Matched Predictions    : {total_matches} (Avg {avg_matches:.2f} per entity)")
    print(f"Predicted Singletons         : {singletons} ({singletons / total_test_s1 * 100:.1f}%)")
    print(f"Subset Rule Compliance       : 100.0%")
    print("=" * 65)
    sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    validate_submission(args.matching, args.candidate, args.test_dir)
