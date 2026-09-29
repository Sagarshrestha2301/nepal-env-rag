"""
build_index.py — Step 7 of nepal-env-rag.

Loads chunks + their vectors into a local LanceDB table, drops noise chunks,
and builds a vector (ANN) index and a full-text (BM25) index for hybrid search.

Noise filter (no re-embedding needed; it only removes rows):
  - chunks dense with citations (leftover reference lists)
  - documents that are peer-review discussions (referee comments, replies)

Input:  data/chunks/<collection>/<category>.jsonl
        data/embeddings/<collection>/<category>/vectors_*.npy + ids_*.txt
Output: data/lancedb/            (table "chunks", fully offline)
        data/index_dropped_sample.jsonl   (examples of dropped chunks, to review)

Usage:
    python build_index.py --category 01_Floods_GLOFs   # pilot (overwrites table)
    python build_index.py                              # full build (overwrites table)
    python build_index.py --no-filter                  # keep everything
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa

# --- Configuration --------------------------------------------------------

CHUNK_DIR = os.path.join("data", "chunks")
EMBED_DIR = os.path.join("data", "embeddings")
DB_DIR = os.path.join("data", "lancedb")
TABLE = "chunks"
DROPPED_SAMPLE = os.path.join("data", "index_dropped_sample.jsonl")

EMBED_DIM = 1024
ANN_MIN_ROWS = 50_000       # below this, exact (brute-force) search is fast enough
SAMPLE_PER_REASON = 100

# Reference-list signals. Years alone are weak evidence (environmental science
# prose is full of dates), so they count only a quarter.
REF_STRONG = re.compile(
    r"\bdoi\b|https?://|\[\d{1,3}\]|\bpp?\.\s*\d|\bvol\.|\b\d+\s*\(\d+\)\s*[:,]|\bet al\b"
    r"|\b\d+\s*:\s*\d+\s*[-–]\s*\d+",          # "16: 30164-30180" volume: pages
    re.I,
)
REF_INITIAL = re.compile(r"\b[A-Z]\.(?=[\s,])")          # author initials: "P., J. Smith"
REF_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
DEFAULT_CITATION_DENSITY = 0.12   # weighted markers per word; prose ~0.0-0.08, references ~0.2+

# Peer-review discussion pages (referee reports, author replies).
REVIEW_MARKER = re.compile(
    r"interactive comment|printer-friendly version|anonymous referee|"
    r"referee\s*#\s*\d|reviewer\s*#\s*\d|response to (?:the )?reviewers?|"
    r"reply to (?:the )?referees?|authors?'? response to|"
    r"response to the (?:general|specific) comments?|technical corrections|"
    r"we thank the (?:anonymous )?(?:reviewer|referee)s?\b",
    re.I,
)
# Line-number replies: "L418: please comment", "L465-470:".
LINE_REF = re.compile(r"\bL\d{2,4}(?:\s*[-‐–]\s*\d{1,4})?\s*:")

SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("doc_id", pa.string()),
    ("chunk_index", pa.int32()),
    ("collection", pa.string()),
    ("category", pa.string()),
    ("title", pa.string()),
    ("year", pa.string()),
    ("year_int", pa.int32()),       # 0 = unknown; used for range filters
    ("author", pa.string()),
    ("path", pa.string()),
    ("section", pa.string()),
    ("page_start", pa.int32()),
    ("page_end", pa.int32()),
    ("lang", pa.string()),
    ("tokens", pa.int32()),
    ("text", pa.string()),          # shown to the LLM / user
    ("embed_text", pa.string()),    # header + text; used for keyword search
    ("vector", pa.list_(pa.float32(), EMBED_DIM)),
])


# --- Noise filter ---------------------------------------------------------

def citation_density(text: str) -> float:
    """Weighted reference-list markers per word."""
    words = len(text.split())
    score = (len(REF_STRONG.findall(text)) + len(REF_INITIAL.findall(text))
             + 0.25 * len(REF_YEAR.findall(text)))
    return score / max(words, 1)


def is_review_chunk(text: str) -> bool:
    """Referee/author-reply language, or 2+ line-number references (L418:)."""
    return bool(REVIEW_MARKER.search(text)) or len(LINE_REF.findall(text)) >= 2


def review_doc_ids(records: list[dict]) -> set[str]:
    """Docs whose first chunk, or 2+ chunks, look like peer-review discussion."""
    hits = Counter(r["doc_id"] for r in records if is_review_chunk(r["text"]))
    first = {r["doc_id"] for r in records
             if r["chunk_index"] == 0 and is_review_chunk(r["text"])}
    return {doc for doc, n in hits.items() if n >= 2} | first


def drop_reason(rec: dict, review_docs: set[str], max_density: float) -> str | None:
    if rec["doc_id"] in review_docs:
        return "peer_review"
    if citation_density(rec["text"]) > max_density:
        return "citation_dense"
    return None


# --- Loading --------------------------------------------------------------

def chunk_files(category: str | None) -> list[str]:
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*", "*.jsonl")))
    if category:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] == category]
    if not files:
        sys.exit("No chunk files found" + (f" for '{category}'" if category else "") + ".")
    return files


def load_vectors(chunk_file: str) -> tuple[np.ndarray, dict[str, int]]:
    """All saved vectors for one chunk file, plus chunk_id -> row lookup."""
    collection = os.path.basename(os.path.dirname(chunk_file))
    category = os.path.splitext(os.path.basename(chunk_file))[0]
    out_dir = os.path.join(EMBED_DIR, collection, category)

    blocks, ids = [], []
    for vec_path in sorted(glob.glob(os.path.join(out_dir, "vectors_*.npy"))):
        ids_path = vec_path.replace("vectors_", "ids_").replace(".npy", ".txt")
        if not os.path.exists(ids_path):
            continue  # incomplete shard
        blocks.append(np.load(vec_path))
        with open(ids_path, encoding="utf-8") as f:
            ids.extend(line.strip() for line in f if line.strip())

    if not blocks:
        return np.empty((0, EMBED_DIM), dtype=np.float32), {}
    return np.vstack(blocks).astype(np.float32), {cid: i for i, cid in enumerate(ids)}


def to_int(value: str) -> int:
    return int(value) if str(value).isdigit() else 0


def build_batch(records: list[dict], vectors: np.ndarray) -> pa.Table:
    columns = {name: [] for name in SCHEMA.names if name != "vector"}
    for r in records:
        columns["chunk_id"].append(r["chunk_id"])
        columns["doc_id"].append(r["doc_id"])
        columns["chunk_index"].append(r["chunk_index"])
        columns["collection"].append(r["collection"])
        columns["category"].append(r["category"])
        columns["title"].append(r["title"])
        columns["year"].append(r["year"] or "")
        columns["year_int"].append(to_int(r["year"]))
        columns["author"].append(r.get("author", ""))
        columns["path"].append(r["path"])
        columns["section"].append(r["section"])
        columns["page_start"].append(r["page_start"])
        columns["page_end"].append(r["page_end"])
        columns["lang"].append(r["lang"])
        columns["tokens"].append(r["tokens"])
        columns["text"].append(r["text"])
        columns["embed_text"].append(r["embed_text"])

    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    arrays = [pa.array(columns[n], type=SCHEMA.field(n).type) for n in SCHEMA.names if n != "vector"]
    arrays.append(pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


# --- Indexes --------------------------------------------------------------

def build_indexes(table, rows: int) -> None:
    if rows >= ANN_MIN_ROWS:
        partitions = max(64, int(math.sqrt(rows)))
        print(f"Building vector index (IVF_PQ, {partitions} partitions)... takes a few minutes")
        table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_PQ",
            num_partitions=partitions,
            num_sub_vectors=EMBED_DIM // 16,   # 64 sub-vectors
            replace=True,
        )
    else:
        print(f"{rows:,} rows: skipping vector index (exact search is fast at this size)")

    print("Building full-text (BM25) index on embed_text...")
    try:
        from lancedb.index import FTS  # lancedb >= 0.25
        table.create_index("embed_text", config=FTS(), replace=True)
    except ImportError:
        table.create_fts_index("embed_text", replace=True)  # older versions
    except Exception as e:
        sys.exit(f"Full-text index failed: {e}")


# --- Main -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LanceDB search index.")
    parser.add_argument("--category", help="only one folder (pilot)")
    parser.add_argument("--no-filter", action="store_true", help="keep noise chunks")
    parser.add_argument("--citation-density", type=float, default=DEFAULT_CITATION_DENSITY,
                        help="drop chunks above this citation-markers-per-word ratio")
    args = parser.parse_args()

    try:
        import lancedb
    except ImportError:
        sys.exit("LanceDB is not installed. Run: pip install lancedb")

    files = chunk_files(args.category)
    db = lancedb.connect(DB_DIR)
    table = None
    stats: Counter = Counter()
    samples: dict[str, list[dict]] = {"peer_review": [], "citation_dense": []}
    start = time.time()

    for path in files:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        vectors, row_of = load_vectors(path)
        review_docs = set() if args.no_filter else review_doc_ids(records)

        keep, rows = [], []
        for rec in records:
            reason = None if args.no_filter else drop_reason(rec, review_docs, args.citation_density)
            if reason:
                stats[f"dropped_{reason}"] += 1
                if len(samples[reason]) < SAMPLE_PER_REASON:
                    samples[reason].append({"chunk_id": rec["chunk_id"], "title": rec["title"],
                                            "text": rec["text"][:500]})
                continue
            if rec["chunk_id"] not in row_of:
                stats["missing_vector"] += 1
                continue
            keep.append(rec)
            rows.append(row_of[rec["chunk_id"]])

        if keep:
            batch = build_batch(keep, vectors[rows])
            if table is None:
                table = db.create_table(TABLE, data=batch, mode="overwrite")
            else:
                table.add(batch)
            stats["rows"] += len(keep)
        print(f"  {os.path.relpath(path, CHUNK_DIR)}: {len(keep):,} kept of {len(records):,}")

    if table is None:
        sys.exit("Nothing indexed. Did the embedding step finish?")

    try:
        table.optimize()  # merge the many small appends into fewer files
    except Exception:
        pass
    build_indexes(table, stats["rows"])

    with open(DROPPED_SAMPLE, "w", encoding="utf-8") as f:
        for reason, items in samples.items():
            for item in items:
                f.write(json.dumps({"reason": reason, **item}, ensure_ascii=False) + "\n")

    print(f"\nDone in {(time.time() - start) / 60:.1f} min -> {DB_DIR} (table '{TABLE}')")
    print(f"  rows indexed:            {stats['rows']:,}")
    print(f"  dropped (reference-like): {stats['dropped_citation_dense']:,}")
    print(f"  dropped (peer review):    {stats['dropped_peer_review']:,}")
    if stats["missing_vector"]:
        print(f"  missing vectors:         {stats['missing_vector']:,} "
              "(embedding not finished for these; rerun after embed_chunks.py)")
    print(f"  review dropped examples: {DROPPED_SAMPLE}")


if __name__ == "__main__":
    main()