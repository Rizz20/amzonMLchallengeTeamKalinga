"""
features.py — Pairwise feature engineering for Business Entity Resolution.

Computes ~28 features for each (S1, candidate) pair.
Designed for vectorized batch processing for scale.

Feature groups:
  - Name similarity (10 features)
  - Address similarity (7 features)
  - Cross-field / meta (6 features)
  - Phonetic (3 features)
  - Structural (2 features)
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.text_utils import (
    normalize_name, core_name, normalize_address, address_tokens,
    extract_postal_code, extract_street_number,
    soundex, nysiis, full_name_soundex,
    name_completeness, address_completeness,
    has_digits_in_name, is_latin_script,
    name_trigrams,
)

# Feature column names — in order matching the numpy array columns
FEATURE_NAMES = [
    # Name features (0-9)
    "name_jaro_winkler",      # 0
    "name_levenshtein_ratio", # 1
    "name_token_sort_ratio",  # 2
    "name_partial_ratio",     # 3
    "name_jaccard_token",     # 4
    "name_jaccard_trigram",   # 5
    "name_tfidf_cosine",      # 6  (filled in separately via vectorizer)
    "core_name_exact",        # 7
    "core_name_jaro",         # 8
    "core_name_trigram_jaccard", # 9
    # Address features (10-16)
    "addr_jaccard_token",     # 10
    "addr_levenshtein_ratio", # 11
    "addr_tfidf_cosine",      # 12  (filled in separately)
    "postal_code_exact",      # 13
    "postal_code_prefix_match", # 14
    "street_num_exact",       # 15
    "city_token_overlap",     # 16
    # Phonetic features (17-19)
    "soundex_match",          # 17
    "nysiis_match",           # 18
    "full_soundex_match",     # 19
    # Cross-field / meta features (20-27)
    "country_match",          # 20
    "source_is_s2",           # 21  (1 if candidate is S2, 0 if S3)
    "name_addr_product",      # 22  (jaro_winkler × addr_jaccard_token)
    "name_completeness_s1",   # 23
    "name_completeness_cand", # 24
    "addr_completeness_s1",   # 25
    "addr_completeness_cand", # 26
    "name_has_digits",        # 27
]

N_FEATURES = len(FEATURE_NAMES)


# ──────────────────────────────────────────────────────────────────────────────
# String similarity helpers
# ──────────────────────────────────────────────────────────────────────────────

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _jaccard_tokens(text_a: str, text_b: str) -> float:
    return _jaccard(set(text_a.split()), set(text_b.split()))


def _jaccard_trigrams(text_a: str, text_b: str) -> float:
    a_grams = set(name_trigrams(text_a)) if text_a else set()
    b_grams = set(name_trigrams(text_b)) if text_b else set()
    return _jaccard(a_grams, b_grams)


def _token_overlap(toks_a: List[str], toks_b: List[str]) -> float:
    """Fraction of tokens in the smaller set that appear in the larger set."""
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0
    set_a, set_b = set(toks_a), set(toks_b)
    return len(set_a & set_b) / min(len(set_a), len(set_b))


# ──────────────────────────────────────────────────────────────────────────────
# Per-pair feature computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_pair_features(s1_row: dict, cand_row: dict) -> np.ndarray:
    """
    Compute all features for a single (S1, candidate) pair.

    Args:
        s1_row:   Source 1 record dict.
        cand_row: Candidate (S2 or S3) record dict.

    Returns:
        1D numpy array of shape (N_FEATURES,) with float32 values.
    """
    try:
        from rapidfuzz import fuzz, distance as rfuzz_distance
    except ImportError:
        raise ImportError("rapidfuzz is required. Install with: pip install rapidfuzz")

    feat = np.zeros(N_FEATURES, dtype=np.float32)

    # ── Normalize fields ─────────────────────────────────────────────────────
    s1_name_raw = s1_row.get("business_name", "") or ""
    cand_name_raw = cand_row.get("business_name", "") or ""
    s1_addr_raw = s1_row.get("business_address", "") or ""
    cand_addr_raw = cand_row.get("business_address", "") or ""
    s1_country = (s1_row.get("country", "") or "").strip()
    cand_country = (cand_row.get("country", "") or "").strip()
    cand_id = cand_row.get("entity_id", "")

    s1_name_norm = normalize_name(s1_name_raw)
    cand_name_norm = normalize_name(cand_name_raw)
    s1_core = core_name(s1_name_raw)
    cand_core = core_name(cand_name_raw)
    s1_addr_norm = normalize_address(s1_addr_raw)
    cand_addr_norm = normalize_address(cand_addr_raw)

    # ── Name features (0-9) ──────────────────────────────────────────────────
    if s1_name_norm and cand_name_norm:
        feat[0] = fuzz.WRatio(s1_name_norm, cand_name_norm) / 100.0     # jaro_winkler proxy
        feat[1] = fuzz.ratio(s1_name_norm, cand_name_norm) / 100.0      # levenshtein ratio
        feat[2] = fuzz.token_sort_ratio(s1_name_norm, cand_name_norm) / 100.0
        feat[3] = fuzz.partial_ratio(s1_name_norm, cand_name_norm) / 100.0
        feat[4] = _jaccard_tokens(s1_name_norm, cand_name_norm)
        feat[5] = _jaccard_trigrams(s1_name_norm, cand_name_norm)
        # feat[6] = tfidf_cosine filled separately
        feat[7] = 1.0 if s1_core and cand_core and s1_core == cand_core else 0.0
        if s1_core and cand_core:
            feat[8] = fuzz.WRatio(s1_core, cand_core) / 100.0
            feat[9] = _jaccard_trigrams(s1_core, cand_core)
    else:
        # One or both names empty
        feat[0] = feat[1] = feat[2] = feat[3] = 0.0
        feat[4] = feat[5] = 0.0
        feat[7] = feat[8] = feat[9] = 0.0

    # ── Address features (10-16) ─────────────────────────────────────────────
    if s1_addr_norm and cand_addr_norm:
        feat[10] = _jaccard_tokens(s1_addr_norm, cand_addr_norm)
        feat[11] = fuzz.ratio(s1_addr_norm, cand_addr_norm) / 100.0
        # feat[12] = tfidf_cosine filled separately
    elif not s1_addr_norm and not cand_addr_norm:
        feat[10] = 1.0   # both empty = treat as identical
        feat[11] = 1.0
    # else: one empty → 0 (default)

    # Postal code
    s1_postal = extract_postal_code(s1_addr_raw, s1_country)
    cand_postal = extract_postal_code(cand_addr_raw, cand_country)
    if s1_postal and cand_postal:
        feat[13] = 1.0 if s1_postal == cand_postal else 0.0
        feat[14] = 1.0 if s1_postal[:3] == cand_postal[:3] else 0.0

    # Street number
    s1_snum = extract_street_number(s1_addr_raw)
    cand_snum = extract_street_number(cand_addr_raw)
    if s1_snum and cand_snum:
        feat[15] = 1.0 if s1_snum == cand_snum else 0.0

    # City token overlap (last meaningful address component is often the city)
    s1_addr_toks = address_tokens(s1_addr_raw)
    cand_addr_toks = address_tokens(cand_addr_raw)
    if s1_addr_toks and cand_addr_toks:
        # Heuristic: take the last 3 tokens as "city region"
        feat[16] = _token_overlap(s1_addr_toks[-3:], cand_addr_toks[-3:])

    # ── Phonetic features (17-19) ─────────────────────────────────────────────
    s1_sdx = soundex(s1_name_raw)
    cand_sdx = soundex(cand_name_raw)
    feat[17] = 1.0 if s1_sdx and cand_sdx and s1_sdx == cand_sdx else 0.0

    s1_nys = nysiis(s1_name_raw)
    cand_nys = nysiis(cand_name_raw)
    feat[18] = 1.0 if s1_nys and cand_nys and s1_nys == cand_nys else 0.0

    s1_fsdx = full_name_soundex(s1_name_raw)
    cand_fsdx = full_name_soundex(cand_name_raw)
    feat[19] = 1.0 if s1_fsdx and cand_fsdx and s1_fsdx == cand_fsdx else 0.0

    # ── Cross-field / meta features (20-27) ──────────────────────────────────
    feat[20] = 1.0 if s1_country.upper() == cand_country.upper() else 0.0
    feat[21] = 1.0 if cand_id.startswith("S2-") else 0.0
    feat[22] = feat[0] * feat[10]   # name_jaro × addr_token_jaccard
    feat[23] = float(name_completeness(s1_name_raw))
    feat[24] = float(name_completeness(cand_name_raw))
    feat[25] = float(address_completeness(s1_addr_raw))
    feat[26] = float(address_completeness(cand_addr_raw))
    feat[27] = 1.0 if has_digits_in_name(s1_name_raw) or has_digits_in_name(cand_name_raw) else 0.0

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Batch feature computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_features_batch(
    pairs: List[Tuple[dict, dict]],
    tfidf_name_vectorizer=None,
    tfidf_addr_vectorizer=None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute features for a batch of pairs.

    Args:
        pairs:                  List of (s1_row, cand_row) tuples.
        tfidf_name_vectorizer:  Fitted sklearn TfidfVectorizer for names (optional).
        tfidf_addr_vectorizer:  Fitted sklearn TfidfVectorizer for addresses (optional).
        verbose:                Show progress bar.

    Returns:
        np.ndarray of shape (len(pairs), N_FEATURES) with float32 values.
    """
    from tqdm import tqdm as _tqdm

    n = len(pairs)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    iterator = _tqdm(enumerate(pairs), total=n, desc="Computing features") if verbose else enumerate(pairs)
    for i, (s1_row, cand_row) in iterator:
        X[i] = compute_pair_features(s1_row, cand_row)

    # Fill TF-IDF cosine similarities if vectorizers provided
    if tfidf_name_vectorizer is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        s1_names = [normalize_name(p[0].get("business_name", "") or "") for p in pairs]
        cand_names = [normalize_name(p[1].get("business_name", "") or "") for p in pairs]
        s1_vecs = tfidf_name_vectorizer.transform(s1_names)
        cand_vecs = tfidf_name_vectorizer.transform(cand_names)
        # Batch cosine similarity (row-wise)
        cosines = np.array([
            cosine_similarity(s1_vecs[i], cand_vecs[i])[0, 0]
            for i in range(n)
        ], dtype=np.float32)
        X[:, 6] = cosines

    if tfidf_addr_vectorizer is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        s1_addrs = [normalize_address(p[0].get("business_address", "") or "") for p in pairs]
        cand_addrs = [normalize_address(p[1].get("business_address", "") or "") for p in pairs]
        s1_vecs = tfidf_addr_vectorizer.transform(s1_addrs)
        cand_vecs = tfidf_addr_vectorizer.transform(cand_addrs)
        cosines = np.array([
            cosine_similarity(s1_vecs[i], cand_vecs[i])[0, 0]
            for i in range(n)
        ], dtype=np.float32)
        X[:, 12] = cosines

    return X


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF vectorizer fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_tfidf_vectorizers(
    all_records: List[dict],
    max_features: int = 50_000,
    ngram_range: Tuple = (1, 2),
):
    """
    Fit TF-IDF vectorizers on all available records (S1 + S2 + S3).
    Returns (name_vectorizer, address_vectorizer).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = [normalize_name(r.get("business_name", "") or "") for r in all_records]
    addrs = [normalize_address(r.get("business_address", "") or "") for r in all_records]

    name_vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        analyzer="char_wb",   # character n-grams: robust to typos + works for non-Latin
        min_df=2,
        sublinear_tf=True,
    )
    name_vec.fit(names)

    addr_vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        analyzer="word",
        min_df=2,
        sublinear_tf=True,
    )
    addr_vec.fit(addrs)

    return name_vec, addr_vec


# ──────────────────────────────────────────────────────────────────────────────
# Training pair builder
# ──────────────────────────────────────────────────────────────────────────────

def build_training_pairs(
    s1_records: List[dict],
    s2_records: List[dict],
    s3_records: List[dict],
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    neg_per_pos: int = 3,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[Tuple[dict, dict]], List[int]]:
    """
    Build (pair, label) training data from blocking candidates and ground truth.

    Positive pairs: (S1, true_match) from ground truth.
    Negative pairs: (S1, blocking_candidate_not_in_ground_truth), sampled.

    Args:
        s1_records:   Source 1 records.
        s2_records:   Source 2 records.
        s3_records:   Source 3 records.
        candidates:   Blocking candidates {s1_id: [candidate_ids]}.
        ground_truth: {s1_id: [matched_ids]}.
        neg_per_pos:  Number of negative samples per positive pair.
        random_seed:  For reproducibility.
        verbose:      Show progress bar.

    Returns:
        (pairs, labels) — pairs is List[(s1_dict, cand_dict)], labels is List[int].
    """
    import random
    rng = random.Random(random_seed)

    # Build lookup maps
    s1_map = {r["entity_id"]: r for r in s1_records}
    s23_map = {r["entity_id"]: r for r in s2_records + s3_records}

    pairs = []
    labels = []

    iterator = s1_records
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(s1_records, desc="Building training pairs", unit="entity")

    for s1_row in iterator:
        s1_id = s1_row["entity_id"]
        true_matches = set(ground_truth.get(s1_id, []))
        cand_pool = candidates.get(s1_id, [])

        # Positive pairs
        pos_cands = [c for c in cand_pool if c in true_matches]
        # Also add true matches that might not be in candidates (recall < 1.0)
        pos_from_gt = list(true_matches)

        for cid in pos_from_gt:
            cand_row = s23_map.get(cid)
            if cand_row:
                pairs.append((s1_row, cand_row))
                labels.append(1)

        # Negative pairs (hard negatives from same blocking bucket)
        neg_pool = [c for c in cand_pool if c not in true_matches]
        n_neg = min(len(neg_pool), neg_per_pos * max(1, len(pos_from_gt)))
        neg_sample = rng.sample(neg_pool, n_neg) if len(neg_pool) > n_neg else neg_pool

        for cid in neg_sample:
            cand_row = s23_map.get(cid)
            if cand_row:
                pairs.append((s1_row, cand_row))
                labels.append(0)

    return pairs, labels
