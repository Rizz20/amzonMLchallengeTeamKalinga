from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.preprocess import extract_numeric_tokens


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates set of character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / union


def _col(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Returns column if present, else a Series filled with default."""
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def extract_pairwise_features(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    s1_embs_dict: Optional[Dict[str, np.ndarray]] = None,
    cand_embs_dict: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Extracts rich pairwise similarity features for candidate pairs.
    Handles Indic script matches, French accented variants, inverted addresses,
    numeric house/PIN codes, and semantic dense embeddings.

    Fully vectorized — no iterrows() loops.
    Entity data is joined via pandas index-join; string metrics use
    np.fromiter over rapidfuzz C-extensions; dense cosine similarity
    uses batched numpy matmul.
    """
    if candidate_pairs_df.empty:
        return pd.DataFrame()

    # ── Build entity lookup indexed by entity_id ────────────────────────────
    combined = pd.concat([s1_df, s2_df, s3_df], ignore_index=True)
    ent = (
        combined
        .set_index("entity_id")[
            ["clean_business_name", "clean_business_address",
             "combined_text", "business_name", "business_address", "country"]
        ]
        .rename(columns={
            "clean_business_name":    "name",
            "clean_business_address": "addr",
            "combined_text":          "comb",
            "business_name":          "raw_name",
            "business_address":       "raw_addr",
        })
    )

    # ── Join entity data onto pair rows via index-join (no iterrows) ─────────
    df = candidate_pairs_df.reset_index(drop=True).copy()
    df = df.join(ent.add_prefix("s1_"),   on="source1_entity_id")
    df = df.join(ent.add_prefix("cand_"), on="candidate_entity_id")

    for c in ["s1_name", "s1_addr", "cand_name", "cand_addr"]:
        df[c] = df[c].fillna("").astype(str)

    # Extract numpy arrays — rapidfuzz functions are C-extensions,
    # so list-comprehension over them is 10-100x faster than iterrows.
    name1 = df["s1_name"].to_numpy()
    name2 = df["cand_name"].to_numpy()
    addr1 = df["s1_addr"].to_numpy()
    addr2 = df["cand_addr"].to_numpy()
    n = len(df)

    # ── 1. Name features ─────────────────────────────────────────────────────
    name_lev        = np.fromiter((Levenshtein.normalized_similarity(a, b) for a, b in zip(name1, name2)), dtype=np.float32, count=n)
    name_jw         = np.fromiter((JaroWinkler.similarity(a, b)            for a, b in zip(name1, name2)), dtype=np.float32, count=n)
    name_token_sort = np.fromiter((fuzz.token_sort_ratio(a, b)             for a, b in zip(name1, name2)), dtype=np.float32, count=n) / 100.0
    name_token_set  = np.fromiter((fuzz.token_set_ratio(a, b)              for a, b in zip(name1, name2)), dtype=np.float32, count=n) / 100.0

    # Purely pandas-vectorized string signals
    name_exact     = ((df["s1_name"] == df["cand_name"]) & df["s1_name"].str.len().gt(0)).astype(np.float32).to_numpy()
    len1           = df["s1_name"].str.len().to_numpy(dtype=np.float32)
    len2           = df["cand_name"].str.len().to_numpy(dtype=np.float32)
    name_len_ratio = np.minimum(len1, len2) / np.maximum(np.maximum(len1, len2), 1.0)

    tok1_first        = df["s1_name"].str.split().str[0].fillna("")
    tok2_first        = df["cand_name"].str.split().str[0].fillna("")
    first_token_match = ((tok1_first == tok2_first) & tok1_first.ne("")).astype(np.float32).to_numpy()

    # Character 3-gram Jaccard — unavoidably set-based but still fast
    name_char3_jaccard = np.fromiter(
        (jaccard_similarity(char_ngrams(a, 3), char_ngrams(b, 3)) for a, b in zip(name1, name2)),
        dtype=np.float32, count=n,
    )

    # ── 2. Address features ──────────────────────────────────────────────────
    addr_token_sort = np.fromiter((fuzz.token_sort_ratio(a, b) for a, b in zip(addr1, addr2)), dtype=np.float32, count=n) / 100.0
    addr_token_set  = np.fromiter((fuzz.token_set_ratio(a, b)  for a, b in zip(addr1, addr2)), dtype=np.float32, count=n) / 100.0

    addr_word_jaccard = np.fromiter(
        (jaccard_similarity(set(a.split()), set(b.split())) for a, b in zip(addr1, addr2)),
        dtype=np.float32, count=n,
    )

    # Compute numeric token sets once, derive three signals
    digit_pairs         = [(extract_numeric_tokens(a), extract_numeric_tokens(b)) for a, b in zip(addr1, addr2)]
    addr_digit_jaccard  = np.fromiter((jaccard_similarity(d1, d2)        for d1, d2 in digit_pairs), dtype=np.float32, count=n)
    common_digits_count = np.fromiter((float(len(d1 & d2))               for d1, d2 in digit_pairs), dtype=np.float32, count=n)
    both_have_digits    = np.fromiter((1.0 if (d1 and d2) else 0.0       for d1, d2 in digit_pairs), dtype=np.float32, count=n)

    alen1          = df["s1_addr"].str.len().to_numpy(dtype=np.float32)
    alen2          = df["cand_addr"].str.len().to_numpy(dtype=np.float32)
    addr_len_ratio = np.minimum(alen1, alen2) / np.maximum(np.maximum(alen1, alen2), 1.0)

    # ── 3. Dense embedding cosine similarity (vectorized matmul) ─────────────
    s1_ids   = df["source1_entity_id"].to_numpy()
    cand_ids = df["candidate_entity_id"].to_numpy()

    if s1_embs_dict and cand_embs_dict:
        dim      = next(iter(s1_embs_dict.values())).shape[0]
        zeros    = np.zeros(dim, dtype=np.float32)
        s1_mat   = np.stack([s1_embs_dict.get(s, zeros)  for s in s1_ids])
        cand_mat = np.stack([cand_embs_dict.get(c, zeros) for c in cand_ids])
        # Element-wise product + row-sum = paired dot products (no Python loop)
        dense_cos_sim = (s1_mat * cand_mat).sum(axis=1).astype(np.float32)
    else:
        dense_cos_sim = np.zeros(n, dtype=np.float32)

    # ── 4. Retrieval rank & score signals ─────────────────────────────────────
    sparse_rank  = _col(df, "sparse_rank",  -1  ).to_numpy(dtype=float)
    sparse_score = _col(df, "sparse_score",  0.0).to_numpy(dtype=np.float32)
    dense_rank   = _col(df, "dense_rank",   -1  ).to_numpy(dtype=float)
    dense_score  = _col(df, "dense_score",   0.0).to_numpy(dtype=np.float32)
    is_in_sparse = _col(df, "is_in_sparse",  0  ).to_numpy(dtype=np.float32)
    is_in_dense  = _col(df, "is_in_dense",   0  ).to_numpy(dtype=np.float32)

    sparse_recip_rank     = np.where(sparse_rank > 0, 1.0 / sparse_rank, 0.0).astype(np.float32)
    dense_recip_rank      = np.where(dense_rank  > 0, 1.0 / dense_rank,  0.0).astype(np.float32)
    both_sparse_and_dense = ((is_in_sparse == 1) & (is_in_dense == 1)).astype(np.float32)

    # ── 5. Metadata signals ───────────────────────────────────────────────────
    is_source2 = df["candidate_entity_id"].str.startswith("S2-").astype(np.float32).to_numpy()
    is_source3 = df["candidate_entity_id"].str.startswith("S3-").astype(np.float32).to_numpy()

    return pd.DataFrame({
        "source1_entity_id":     s1_ids,
        "candidate_entity_id":   cand_ids,
        "name_levenshtein":      name_lev,
        "name_jaro_winkler":     name_jw,
        "name_token_sort":       name_token_sort,
        "name_token_set":        name_token_set,
        "name_exact":            name_exact,
        "name_char3_jaccard":    name_char3_jaccard,
        "first_token_match":     first_token_match,
        "name_len_ratio":        name_len_ratio,
        "addr_token_sort":       addr_token_sort,
        "addr_token_set":        addr_token_set,
        "addr_word_jaccard":     addr_word_jaccard,
        "addr_digit_jaccard":    addr_digit_jaccard,
        "common_digits_count":   common_digits_count,
        "both_have_digits":      both_have_digits,
        "addr_len_ratio":        addr_len_ratio,
        "dense_cos_sim":         dense_cos_sim,
        "sparse_score":          sparse_score,
        "sparse_recip_rank":     sparse_recip_rank,
        "dense_score":           dense_score,
        "dense_recip_rank":      dense_recip_rank,
        "is_in_sparse":          is_in_sparse,
        "is_in_dense":           is_in_dense,
        "both_sparse_and_dense": both_sparse_and_dense,
        "is_source2":            is_source2,
        "is_source3":            is_source3,
    })


FEATURE_COLUMNS = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_exact",
    "name_char3_jaccard",
    "first_token_match",
    "name_len_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_word_jaccard",
    "addr_digit_jaccard",
    "common_digits_count",
    "both_have_digits",
    "addr_len_ratio",
    "dense_cos_sim",
    "sparse_score",
    "sparse_recip_rank",
    "dense_score",
    "dense_recip_rank",
    "is_in_sparse",
    "is_in_dense",
    "both_sparse_and_dense",
    "is_source2",
    "is_source3",
]
