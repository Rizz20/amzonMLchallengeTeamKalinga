import csv
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer
from src.config import (
    DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    SPARSE_TOP_K,
    DENSE_TOP_K,
    MAX_UNION_CANDIDATES,
)


class DenseRetriever:
    """
    Multilingual dense retrieval using sentence-transformers.
    FAISS-accelerated if faiss is installed, with high-performance PyTorch tensor fallback.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        self._load_model()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def search(
        self, query_embs: np.ndarray, index_embs: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (indices, scores) of shape (n_queries, top_k).
        """
        n_queries = query_embs.shape[0]
        n_index = index_embs.shape[0]
        top_k = min(top_k, n_index)
        
        if top_k == 0:
            return np.empty((n_queries, 0), dtype=int), np.empty((n_queries, 0), dtype=np.float32)

        if HAS_FAISS:
            dim = index_embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(index_embs)
            scores, indices = index.search(query_embs, top_k)
            return indices, scores
        else:
            # High-performance PyTorch tensor batch dot product
            device_torch = torch.device(self.device if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                q_tensor = torch.from_numpy(query_embs).to(device_torch)
                idx_tensor = torch.from_numpy(index_embs).to(device_torch)
                
                # Compute in batches if query set is large to conserve memory
                batch_size = 512
                all_indices = []
                all_scores = []
                for i in range(0, n_queries, batch_size):
                    batch_q = q_tensor[i : i + batch_size]
                    sim = torch.mm(batch_q, idx_tensor.t())
                    batch_scores, batch_idx = torch.topk(sim, k=top_k, dim=1)
                    all_scores.append(batch_scores.cpu().numpy())
                    all_indices.append(batch_idx.cpu().numpy())
                    
                indices = np.vstack(all_indices)
                scores = np.vstack(all_scores)
                return indices, scores


class SparseRetriever:
    """
    Character 3-gram TF-IDF sparse retriever.
    Effective for matching noisy names, subword matches, and minor spelling variations.
    """
    def __init__(self, ngram_range: Tuple[int, int] = (3, 3)):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )

    def search(
        self, query_texts: List[str], index_texts: List[str], top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (indices, scores) of shape (n_queries, top_k).
        Fully vectorized — no Python row loop.
        """
        n_queries = len(query_texts)
        n_index = len(index_texts)
        top_k = min(top_k, n_index)

        if top_k == 0:
            return np.empty((n_queries, 0), dtype=int), np.empty((n_queries, 0), dtype=np.float32)

        # Fit TF-IDF on candidate pool and transform queries
        index_matrix = self.vectorizer.fit_transform(index_texts)  # (n_index, vocab)
        query_matrix = self.vectorizer.transform(query_texts)       # (n_queries, vocab)

        # Cosine similarity via sparse dot product → (n_queries, n_index)
        sim_matrix = query_matrix.dot(index_matrix.T)  # sparse CSR

        # Convert to dense only if small enough; otherwise use efficient sparse approach
        if n_queries * n_index <= 50_000_000:
            # Dense path: fast numpy argpartition over full matrix
            dense = np.asarray(sim_matrix.todense(), dtype=np.float32)  # (n_queries, n_index)
            if top_k >= n_index:
                # Return all indices sorted by score descending
                sort_idx = np.argsort(-dense, axis=1)  # (n_queries, n_index)
                all_indices = sort_idx[:, :top_k]
                all_scores = np.take_along_axis(dense, all_indices, axis=1)
            else:
                # Partial sort via argpartition for speed
                part_idx = np.argpartition(-dense, top_k, axis=1)[:, :top_k]  # (n_queries, top_k)
                top_vals = np.take_along_axis(dense, part_idx, axis=1)
                sort_within = np.argsort(-top_vals, axis=1)
                all_indices = np.take_along_axis(part_idx, sort_within, axis=1)
                all_scores = np.take_along_axis(top_vals, sort_within, axis=1)
            # Mark zero-score slots as -1 (no match)
            all_indices = np.where(all_scores > 0, all_indices, -1)
        else:
            # Sparse path for very large matrices: process in row-chunks to avoid OOM
            sim_csr = sim_matrix.tocsr()
            all_indices = np.full((n_queries, top_k), -1, dtype=np.intp)
            all_scores = np.zeros((n_queries, top_k), dtype=np.float32)
            chunk = 10_000
            for start in range(0, n_queries, chunk):
                end = min(start + chunk, n_queries)
                block = np.asarray(
                    sim_csr[start:end].todense(), dtype=np.float32
                )  # (chunk, n_index)
                k = min(top_k, block.shape[1])
                part = np.argpartition(-block, k, axis=1)[:, :k]
                vals = np.take_along_axis(block, part, axis=1)
                order = np.argsort(-vals, axis=1)
                idx = np.take_along_axis(part, order, axis=1)
                v = np.take_along_axis(vals, order, axis=1)
                all_indices[start:end, :k] = np.where(v > 0, idx, -1)
                all_scores[start:end, :k] = v

        return all_indices, all_scores


def run_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    sparse_k: int = SPARSE_TOP_K,
    dense_k: int = DENSE_TOP_K,
    dense_retriever: Optional[DenseRetriever] = None,
    s1_embeddings: Optional[np.ndarray] = None,
    candidate_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Performs Country-Isolated Dual-Channel Blocking across S1, S2, and S3.
    Returns:
        candidate_pairs_df: DataFrame of candidate pairs with ranks and similarity scores.
        s1_embs_dict: Precomputed embeddings for S1 records (s1_id -> vector).
        cand_embs_dict: Precomputed embeddings for Candidate records (cand_id -> vector).
    """
    # Combine S2 and S3 into unified candidate pool
    s2_df = s2_df.copy()
    s3_df = s3_df.copy()
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)

    if dense_retriever is None:
        dense_retriever = DenseRetriever()

    # Precompute or retrieve dense embeddings
    s1_embs_dict: Dict[str, np.ndarray] = {}
    cand_embs_dict: Dict[str, np.ndarray] = {}

    unique_countries = s1_df["country"].unique()
    all_pairs = []

    for country in unique_countries:
        s1_sub = s1_df[s1_df["country"] == country].reset_index(drop=True)
        cand_sub = cand_df[cand_df["country"] == country].reset_index(drop=True)

        if s1_sub.empty:
            continue

        s1_ids = s1_sub["entity_id"].tolist()
        s1_texts = s1_sub["combined_text"].tolist()

        if cand_sub.empty:
            # No candidates in this country (all S1 are singletons)
            continue

        cand_ids = cand_sub["entity_id"].tolist()
        cand_texts = cand_sub["combined_text"].tolist()

        # 1. Sparse Retrieval
        sparse_retriever = SparseRetriever()
        sparse_indices, sparse_scores = sparse_retriever.search(
            s1_texts, cand_texts, top_k=sparse_k
        )

        # 2. Dense Retrieval
        # Encode S1 subset
        s1_embs = dense_retriever.encode(s1_texts)
        for eid, emb in zip(s1_ids, s1_embs):
            s1_embs_dict[eid] = emb

        # Encode Candidate subset
        cand_embs = dense_retriever.encode(cand_texts)
        for eid, emb in zip(cand_ids, cand_embs):
            cand_embs_dict[eid] = emb

        dense_indices, dense_scores = dense_retriever.search(
            s1_embs, cand_embs, top_k=dense_k
        )

        # 3. Union & Candidate Pair Assembly
        for row_i, s1_id in enumerate(s1_ids):
            seen_cands: Dict[str, Dict] = {}

            # Process sparse candidates
            for rank, (cand_idx, score) in enumerate(
                zip(sparse_indices[row_i], sparse_scores[row_i])
            ):
                if cand_idx < 0:
                    continue
                cid = cand_ids[cand_idx]
                seen_cands[cid] = {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "sparse_rank": rank + 1,
                    "sparse_score": float(score),
                    "dense_rank": -1,
                    "dense_score": 0.0,
                    "is_in_sparse": 1,
                    "is_in_dense": 0,
                }

            # Process dense candidates
            for rank, (cand_idx, score) in enumerate(
                zip(dense_indices[row_i], dense_scores[row_i])
            ):
                if cand_idx < 0:
                    continue
                cid = cand_ids[cand_idx]
                if cid in seen_cands:
                    seen_cands[cid]["dense_rank"] = rank + 1
                    seen_cands[cid]["dense_score"] = float(score)
                    seen_cands[cid]["is_in_dense"] = 1
                else:
                    seen_cands[cid] = {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cid,
                        "sparse_rank": -1,
                        "sparse_score": 0.0,
                        "dense_rank": rank + 1,
                        "dense_score": float(score),
                        "is_in_sparse": 0,
                        "is_in_dense": 1,
                    }

            # Collect pairs for this S1 entity, capped at MAX_UNION_CANDIDATES
            # Sort so that dual-channel matches (in both sparse+dense) are preferred
            cand_list = sorted(
                seen_cands.values(),
                key=lambda x: (-(x["is_in_sparse"] + x["is_in_dense"]),
                               -(x["dense_score"] + x["sparse_score"])),
            )
            for cand_data in cand_list[:MAX_UNION_CANDIDATES]:
                all_pairs.append(cand_data)

    candidate_pairs_df = pd.DataFrame(all_pairs)
    return candidate_pairs_df, s1_embs_dict, cand_embs_dict


def export_candidate_pairs_tsv(
    candidate_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    output_path: str,
) -> None:
    """
    Writes output/candidate_pairs.tsv in exact competition format:
    source1_entity_id<TAB>candidate_entity_ids
    
    Ensures:
    - Every S1 entity in all_s1_ids appears exactly once.
    - Candidate IDs are comma-separated S2- and S3- identifiers.
    - Empty string if no candidates found.
    - Preserves row order matching all_s1_ids.
    """
    cand_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
    
    if not candidate_pairs_df.empty:
        for _, row in candidate_pairs_df.iterrows():
            s1 = row["source1_entity_id"]
            cand = row["candidate_entity_id"]
            if s1 in cand_map:
                cand_map[s1].append(cand)

    rows = []
    for s1_id in all_s1_ids:
        cands = cand_map.get(s1_id, [])
        # Deduplicate while preserving order
        seen = set()
        dedup_cands = [c for c in cands if not (c in seen or seen.add(c))]
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(dedup_cands),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(
        output_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
