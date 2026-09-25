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
    """
    if candidate_pairs_df.empty:
        return pd.DataFrame()

    # Create fast entity lookup dictionaries
    combined_entities = pd.concat([s1_df, s2_df, s3_df], ignore_index=True)
    entity_map = {
        row["entity_id"]: {
            "name": row["clean_business_name"],
            "addr": row["clean_business_address"],
            "comb": row["combined_text"],
            "raw_name": row["business_name"],
            "raw_addr": row["business_address"],
            "country": row["country"],
        }
        for _, row in combined_entities.iterrows()
    }

    feature_rows = []

    for _, row in candidate_pairs_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]

        e1 = entity_map.get(s1_id, {})
        e2 = entity_map.get(cand_id, {})

        name1 = e1.get("name", "")
        name2 = e2.get("name", "")
        addr1 = e1.get("addr", "")
        addr2 = e2.get("addr", "")

        # 1. Name Features
        name_lev = Levenshtein.normalized_similarity(name1, name2)
        name_jw = JaroWinkler.similarity(name1, name2)
        name_token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        name_token_set = fuzz.token_set_ratio(name1, name2) / 100.0
        name_exact = 1.0 if name1 == name2 and len(name1) > 0 else 0.0

        # Character 3-gram Jaccard
        grams1 = char_ngrams(name1, 3)
        grams2 = char_ngrams(name2, 3)
        name_char3_jaccard = jaccard_similarity(grams1, grams2)

        # First token match
        tokens1 = name1.split()
        tokens2 = name2.split()
        first_token_match = 1.0 if tokens1 and tokens2 and tokens1[0] == tokens2[0] else 0.0

        # Name length ratio
        len1 = len(name1)
        len2 = len(name2)
        name_len_ratio = min(len1, len2) / max(len1, len2, 1)

        # 2. Address Features
        addr_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        addr_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
        
        words1 = set(addr1.split())
        words2 = set(addr2.split())
        addr_word_jaccard = jaccard_similarity(words1, words2)

        # Numeric digit / PIN code overlap
        digits1 = extract_numeric_tokens(addr1)
        digits2 = extract_numeric_tokens(addr2)
        addr_digit_jaccard = jaccard_similarity(digits1, digits2)
        common_digits_count = len(digits1.intersection(digits2))
        both_have_digits = 1.0 if (len(digits1) > 0 and len(digits2) > 0) else 0.0

        addr_len1 = len(addr1)
        addr_len2 = len(addr2)
        addr_len_ratio = min(addr_len1, addr_len2) / max(addr_len1, addr_len2, 1)

        # 3. Dense Embedding Cosine Similarity
        dense_cos_sim = 0.0
        if (
            s1_embs_dict is not None
            and cand_embs_dict is not None
            and s1_id in s1_embs_dict
            and cand_id in cand_embs_dict
        ):
            v1 = s1_embs_dict[s1_id]
            v2 = cand_embs_dict[cand_id]
            dense_cos_sim = float(np.dot(v1, v2))

        # 4. Retrieval Rank & Score Signals
        sparse_rank = row.get("sparse_rank", -1)
        sparse_score = float(row.get("sparse_score", 0.0))
        sparse_recip_rank = 1.0 / sparse_rank if sparse_rank > 0 else 0.0

        dense_rank = row.get("dense_rank", -1)
        dense_score = float(row.get("dense_score", 0.0))
        dense_recip_rank = 1.0 / dense_rank if dense_rank > 0 else 0.0

        is_in_sparse = int(row.get("is_in_sparse", 0))
        is_in_dense = int(row.get("is_in_dense", 0))
        both_sparse_and_dense = 1.0 if (is_in_sparse == 1 and is_in_dense == 1) else 0.0

        # 5. Metadata signals
        is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
        is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

        feature_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "name_levenshtein": name_lev,
            "name_jaro_winkler": name_jw,
            "name_token_sort": name_token_sort,
            "name_token_set": name_token_set,
            "name_exact": name_exact,
            "name_char3_jaccard": name_char3_jaccard,
            "first_token_match": first_token_match,
            "name_len_ratio": name_len_ratio,
            "addr_token_sort": addr_token_sort,
            "addr_token_set": addr_token_set,
            "addr_word_jaccard": addr_word_jaccard,
            "addr_digit_jaccard": addr_digit_jaccard,
            "common_digits_count": float(common_digits_count),
            "both_have_digits": both_have_digits,
            "addr_len_ratio": addr_len_ratio,
            "dense_cos_sim": dense_cos_sim,
            "sparse_score": sparse_score,
            "sparse_recip_rank": sparse_recip_rank,
            "dense_score": dense_score,
            "dense_recip_rank": dense_recip_rank,
            "is_in_sparse": float(is_in_sparse),
            "is_in_dense": float(is_in_dense),
            "both_sparse_and_dense": both_sparse_and_dense,
            "is_source2": is_source2,
            "is_source3": is_source3,
        })

    return pd.DataFrame(feature_rows)


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
