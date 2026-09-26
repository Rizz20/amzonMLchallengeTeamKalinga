"""
io_utils.py — TSV I/O helpers for Business Entity Resolution.

All files in this challenge are tab-separated (.tsv).
"""

import csv
import os
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Reading source files
# ──────────────────────────────────────────────────────────────────────────────

def load_source(path: str) -> List[dict]:
    """
    Load a source TSV (source1/source2/source3) into a list of dicts.
    Columns: entity_id, business_name, business_address, country
    """
    records = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Ensure all fields exist (handle missing columns gracefully)
            records.append({
                "entity_id": row.get("entity_id", "").strip(),
                "business_name": row.get("business_name", "") or "",
                "business_address": row.get("business_address", "") or "",
                "country": row.get("country", "") or "",
            })
    return records


def load_source_as_dict(path: str) -> Dict[str, dict]:
    """Load source TSV as {entity_id: row_dict}."""
    return {r["entity_id"]: r for r in load_source(path)}


def load_ground_truth(path: str) -> Dict[str, List[str]]:
    """
    Load ground truth TSV into {source1_entity_id: [matched_ids]}.
    Returns empty list for singletons (entities with no matches).
    """
    gt = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"].strip()
            matched_raw = row.get("matched_entity_ids", "") or ""
            matched_ids = [m.strip() for m in matched_raw.split(",") if m.strip()]
            gt[s1_id] = matched_ids
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Writing output files
# ──────────────────────────────────────────────────────────────────────────────

def write_matching_results(
    predictions: Dict[str, List[str]],
    path: str,
    all_s1_ids: Optional[List[str]] = None,
) -> None:
    """
    Write matching_results.tsv.

    Args:
        predictions:  {source1_entity_id: [matched_ids]} — may be empty list for singletons.
        path:         output file path.
        all_s1_ids:   If provided, ensures every S1 ID appears (even if not in predictions).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Build ordered list of S1 IDs to write
    if all_s1_ids:
        s1_ids = list(all_s1_ids)
        # Add any in predictions but not in all_s1_ids (shouldn't happen, but be safe)
        extra = set(predictions.keys()) - set(all_s1_ids)
        s1_ids.extend(sorted(extra))
    else:
        s1_ids = sorted(predictions.keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in s1_ids:
            matched = predictions.get(s1_id, [])
            # Deduplicate preserving order
            seen = set()
            deduped = []
            for mid in matched:
                if mid not in seen:
                    seen.add(mid)
                    deduped.append(mid)
            writer.writerow([s1_id, ",".join(deduped)])


def write_candidate_pairs(
    candidates: Dict[str, List[str]],
    path: str,
    all_s1_ids: Optional[List[str]] = None,
) -> None:
    """
    Write candidate_pairs.tsv (the blocking output, before final matching).

    Args:
        candidates:  {source1_entity_id: [candidate_ids]} — from blocking stage.
        path:        output file path.
        all_s1_ids:  If provided, ensures every S1 ID appears.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if all_s1_ids:
        s1_ids = list(all_s1_ids)
        extra = set(candidates.keys()) - set(all_s1_ids)
        s1_ids.extend(sorted(extra))
    else:
        s1_ids = sorted(candidates.keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id in s1_ids:
            cands = candidates.get(s1_id, [])
            # Deduplicate
            seen = set()
            deduped = []
            for cid in cands:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            writer.writerow([s1_id, ",".join(deduped)])


# ──────────────────────────────────────────────────────────────────────────────
# Chunked reading for large files
# ──────────────────────────────────────────────────────────────────────────────

def iter_source_chunks(path: str, chunk_size: int = 100_000):
    """
    Yield chunks of records from a source TSV — memory-efficient for large files.

    Yields:
        List[dict] of up to chunk_size records.
    """
    chunk = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            chunk.append({
                "entity_id": row.get("entity_id", "").strip(),
                "business_name": row.get("business_name", "") or "",
                "business_address": row.get("business_address", "") or "",
                "country": row.get("country", "") or "",
            })
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def count_lines(path: str) -> int:
    """Fast line count (excluding header)."""
    with open(path, "rb") as f:
        count = sum(1 for _ in f)
    return max(0, count - 1)  # subtract header
