"""
blocking.py — Multi-key inverted-index blocking for Business Entity Resolution.

Goal: Reduce the ~22 trillion naive pair space to a manageable candidate set
while keeping blocking recall ≥ 97% (true matches survive blocking).

Strategy: Build an inverted index (key → list of S2/S3 IDs), then for each
S1 entity, union all candidate IDs from its blocking keys.

Blocking keys (see text_utils.blocking_keys for definitions):
  BK1: country + postal prefix + name prefix
  BK2: country + soundex(name)
  BK3: country + NYSIIS(name)
  BK4: country + street_num + soundex(street)
  BK5: country + exact core name
  BK6: country + name prefix-5
  BK7: country + postal code
"""

import sys
import os
import collections
from typing import Dict, List, Set, Optional
from tqdm import tqdm

# Allow importing from sibling directories
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.text_utils import blocking_keys
from utils.io_utils import load_source, iter_source_chunks


# ──────────────────────────────────────────────────────────────────────────────
# Inverted-index blocking
# ──────────────────────────────────────────────────────────────────────────────

def build_inverted_index(
    records: List[dict],
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Build an inverted index from blocking keys → list of entity_ids.

    Args:
        records: List of record dicts (entity_id, business_name, business_address, country).
        verbose: Show progress bar.

    Returns:
        {blocking_key: [entity_id, ...]}
    """
    index: Dict[str, List[str]] = collections.defaultdict(list)
    iterator = tqdm(records, desc="Building inverted index", unit="rec") if verbose else records

    for row in iterator:
        entity_id = row["entity_id"]
        for key in blocking_keys(row):
            index[key].append(entity_id)

    return dict(index)


def generate_candidates(
    s1_records: List[dict],
    index: Dict[str, List[str]],
    max_candidates_per_entity: int = 500,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    For each S1 entity, look up all blocking keys in the index and collect
    candidate S2/S3 entity IDs.

    Args:
        s1_records:               List of Source 1 record dicts.
        index:                    Inverted index from build_inverted_index().
        max_candidates_per_entity: Cap to prevent explosion on very common keys.
        verbose:                  Show progress bar.

    Returns:
        {s1_entity_id: [candidate_s2_s3_entity_ids]}
    """
    candidates: Dict[str, List[str]] = {}
    iterator = tqdm(s1_records, desc="Generating candidates", unit="entity") if verbose else s1_records

    for row in iterator:
        s1_id = row["entity_id"]
        candidate_set: Set[str] = set()

        for key in blocking_keys(row):
            matches = index.get(key, [])
            for mid in matches:
                # Only add S2 and S3 IDs (not other S1 IDs if index was built jointly)
                if mid.startswith("S2-") or mid.startswith("S3-"):
                    candidate_set.add(mid)

        # Cap to avoid exploding on very generic keys (e.g., very common names)
        cand_list = list(candidate_set)
        if len(cand_list) > max_candidates_per_entity:
            # Keep a deterministic subset (sort for reproducibility)
            cand_list = sorted(cand_list)[:max_candidates_per_entity]

        candidates[s1_id] = cand_list

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# N-gram based blocking (supplemental — catches name variations missed by keys)
# ──────────────────────────────────────────────────────────────────────────────

def build_ngram_index(
    records: List[dict],
    n: int = 3,
    field: str = "name",
    verbose: bool = True,
) -> Dict[str, Set[str]]:
    """
    Build a character n-gram inverted index for supplemental blocking.

    Args:
        records: S2/S3 records.
        n:       n-gram size (3 recommended).
        field:   'name' or 'address'.
        verbose: Progress bar.

    Returns:
        {ngram: {entity_id, ...}}
    """
    from utils.text_utils import name_trigrams, address_trigrams

    index: Dict[str, Set[str]] = collections.defaultdict(set)
    iterator = tqdm(records, desc=f"Building {n}-gram index ({field})", unit="rec") if verbose else records

    for row in iterator:
        entity_id = row["entity_id"]
        if field == "name":
            grams = name_trigrams(row.get("business_name", "") or "")
        else:
            grams = address_trigrams(row.get("business_address", "") or "")

        for gram in grams:
            index[gram].add(entity_id)

    return dict(index)


def ngram_candidates(
    s1_record: dict,
    ngram_index: Dict[str, Set[str]],
    min_overlap: int = 3,
    field: str = "name",
) -> Set[str]:
    """
    Find candidates for a single S1 entity using n-gram overlap.

    Args:
        s1_record:    Single S1 record dict.
        ngram_index:  N-gram inverted index from build_ngram_index().
        min_overlap:  Minimum number of shared n-grams to be a candidate.
        field:        'name' or 'address'.

    Returns:
        Set of candidate entity IDs.
    """
    from utils.text_utils import name_trigrams, address_trigrams

    if field == "name":
        grams = set(name_trigrams(s1_record.get("business_name", "") or ""))
    else:
        grams = set(address_trigrams(s1_record.get("business_address", "") or ""))

    if not grams:
        return set()

    overlap_count: Dict[str, int] = collections.Counter()
    for gram in grams:
        for eid in ngram_index.get(gram, set()):
            overlap_count[eid] += 1

    return {eid for eid, cnt in overlap_count.items() if cnt >= min_overlap}


def add_ngram_candidates(
    s1_records: List[dict],
    existing_candidates: Dict[str, List[str]],
    ngram_index: Dict[str, Set[str]],
    min_overlap: int = 3,
    field: str = "name",
    max_candidates_per_entity: int = 500,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Augment existing candidates with n-gram based candidates.
    Useful as a safety net to improve blocking recall.
    """
    iterator = tqdm(s1_records, desc=f"N-gram blocking ({field})", unit="entity") if verbose else s1_records

    for row in iterator:
        s1_id = row["entity_id"]
        new_cands = ngram_candidates(row, ngram_index, min_overlap=min_overlap, field=field)
        new_cands = {c for c in new_cands if c.startswith("S2-") or c.startswith("S3-")}

        existing = set(existing_candidates.get(s1_id, []))
        combined = existing | new_cands

        if len(combined) > max_candidates_per_entity:
            # Prefer the original key-based candidates, add n-gram extras up to cap
            extra = combined - existing
            extra_list = sorted(extra)[:max_candidates_per_entity - len(existing)]
            combined = existing | set(extra_list)

        existing_candidates[s1_id] = list(combined)

    return existing_candidates


# ──────────────────────────────────────────────────────────────────────────────
# Full blocking pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_blocking(
    s1_records: List[dict],
    s2_records: List[dict],
    s3_records: List[dict],
    max_candidates: int = 500,
    use_ngram_fallback: bool = True,
    ngram_min_overlap: int = 3,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Full blocking pipeline: build index on S2+S3, then generate candidates for each S1.

    Args:
        s1_records:          Source 1 records.
        s2_records:          Source 2 records.
        s3_records:          Source 3 records.
        max_candidates:      Max candidates per S1 entity.
        use_ngram_fallback:  Add n-gram blocking as supplemental recall boost.
        ngram_min_overlap:   Min n-gram overlap for n-gram blocking.
        verbose:             Show progress bars.

    Returns:
        {s1_entity_id: [candidate_ids]}
    """
    s2_s3_records = s2_records + s3_records

    if verbose:
        print(f"Building inverted index over {len(s2_s3_records):,} S2+S3 records...")
    index = build_inverted_index(s2_s3_records, verbose=verbose)

    if verbose:
        print(f"Generating candidates for {len(s1_records):,} S1 entities...")
    candidates = generate_candidates(s1_records, index, max_candidates_per_entity=max_candidates, verbose=verbose)

    if use_ngram_fallback:
        if verbose:
            print("Building name n-gram index for supplemental blocking...")
        ngram_idx = build_ngram_index(s2_s3_records, n=3, field="name", verbose=verbose)
        if verbose:
            print("Adding n-gram candidates...")
        candidates = add_ngram_candidates(
            s1_records, candidates, ngram_idx,
            min_overlap=ngram_min_overlap,
            field="name",
            max_candidates_per_entity=max_candidates,
            verbose=verbose,
        )

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# Blocking quality evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_blocking(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> dict:
    """
    Measure blocking quality on the training set.

    Metrics:
    - Blocking recall: fraction of true matches that appear in candidates
    - Reduction ratio: fraction of pairs eliminated vs. brute force
    - Average candidates per S1 entity
    """
    total_true = 0
    true_in_candidates = 0
    total_candidates = 0

    for s1_id, true_matches in ground_truth.items():
        cands = set(candidates.get(s1_id, []))
        total_true += len(true_matches)
        true_in_candidates += sum(1 for m in true_matches if m in cands)
        total_candidates += len(cands)

    n_s1 = len(ground_truth)
    blocking_recall = true_in_candidates / total_true if total_true > 0 else 0.0
    avg_candidates = total_candidates / n_s1 if n_s1 > 0 else 0.0

    return {
        "blocking_recall": blocking_recall,
        "true_matches_found": true_in_candidates,
        "total_true_matches": total_true,
        "avg_candidates_per_entity": avg_candidates,
        "total_candidates": total_candidates,
    }
