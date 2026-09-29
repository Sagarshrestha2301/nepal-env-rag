"""
retriever.py — hybrid search over the nepal-env-rag index (fully offline).

Combines:
  - vector search (meaning):   bge-m3 query embedding vs chunk vectors
  - keyword search (BM25):     exact terms like "Imja", "GLOF", "Melamchi"
and merges them with Reciprocal Rank Fusion (RRF).

Use from Python (the answer step imports this):
    from retriever import Retriever
    r = Retriever()
    hits = r.search("GLOF risk of Imja lake", k=8, year_min=2010)

Or test from the command line:
    python retriever.py "GLOF risk of Imja lake"
    python retriever.py "Melamchi flood 2021" --mode keyword
    python retriever.py "बाढी पहिरो जोखिम" --collection reports --k 5

Set HF_HUB_OFFLINE=1 to guarantee no network access once models are cached.
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np

DB_DIR = os.path.join("data", "lancedb")
TABLE = "chunks"
MODEL_NAME = "BAAI/bge-m3"

RRF_K = 60              # standard RRF constant; dampens the weight of top ranks
DEFAULT_CANDIDATES = 50 # results taken from each search before fusion
NPROBES = 20            # IVF partitions scanned per query (more = better recall)
REFINE_FACTOR = 5       # re-score top candidates with full vectors (fixes PQ error)

RESULT_COLUMNS = ["chunk_id", "doc_id", "title", "year", "author", "collection", "category",
                  "section", "page_start", "page_end", "lang", "path", "text"]

# Characters that break full-text query parsing.
FTS_SPLIT = re.compile(r"[\s\"'`?:!()\[\]{}^~*\\/+\-.,;|&<>=]+")


def rrf_fuse(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, results in zip(("vector", "keyword"), result_lists):
        for rank, row in enumerate(results, 1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            rows.setdefault(cid, row)
            ranks.setdefault(cid, {})[name] = rank
    fused = sorted(scores, key=scores.get, reverse=True)
    return [{**rows[c], "score": round(scores[c], 5), "ranks": ranks[c]} for c in fused]


def fts_query(text: str) -> str:
    """Plain terms only, so punctuation can't break the query parser."""
    return " ".join(t for t in FTS_SPLIT.split(text) if t)


def build_where(collection: str | None = None, category: str | None = None,
                year_min: int | None = None, year_max: int | None = None,
                lang: str | None = None) -> str | None:
    """SQL filter for LanceDB. Unknown years (0) are excluded by year filters."""
    def quote(v: str) -> str:
        return "'" + v.replace("'", "''") + "'"
    parts = []
    if collection:
        parts.append(f"collection = {quote(collection)}")
    if category:
        parts.append(f"category = {quote(category)}")
    if year_min:
        parts.append(f"year_int >= {int(year_min)}")
    if year_max:
        parts.append(f"year_int <= {int(year_max)} AND year_int > 0")
    if lang:
        parts.append(f"lang = {quote(lang)}")
    return " AND ".join(parts) or None


class Retriever:
    def __init__(self, db_dir: str = DB_DIR, table: str = TABLE, device: str = "cpu"):
        import lancedb
        from sentence_transformers import SentenceTransformer

        self.table = lancedb.connect(db_dir).open_table(table)
        # CPU by default: keeps the GPU free for the local LLM.
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        if device == "cuda":
            self.model.half()

    def embed(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].astype(np.float32)

    @staticmethod
    def _finish(q, limit: int, score_column: str) -> list[dict]:
        # We rank by position (RRF), so raw scores aren't needed. Opt out of
        # score autoprojection where supported; otherwise select the score
        # column explicitly. Either way LanceDB's deprecation warning stops.
        if hasattr(q, "disable_scoring_autoprojection"):
            q = q.disable_scoring_autoprojection()
            columns = RESULT_COLUMNS
        else:
            columns = RESULT_COLUMNS + [score_column]
        return q.select(columns).limit(limit).to_list()

    def vector_search(self, query: str, limit: int, where: str | None) -> list[dict]:
        q = self.table.search(self.embed(query), vector_column_name="vector")
        q = q.distance_type("cosine") if hasattr(q, "distance_type") else q.metric("cosine")
        if hasattr(q, "nprobes"):
            q = q.nprobes(NPROBES).refine_factor(REFINE_FACTOR)
        if where:
            q = q.where(where, prefilter=True)
        return self._finish(q, limit, "_distance")

    def keyword_search(self, query: str, limit: int, where: str | None) -> list[dict]:
        terms = fts_query(query)
        if not terms:
            return []
        q = self.table.search(terms, query_type="fts")
        if where:
            q = q.where(where, prefilter=True)
        return self._finish(q, limit, "_score")

    def search(self, query: str, k: int = 8, mode: str = "hybrid",
               candidates: int = DEFAULT_CANDIDATES, **filters) -> list[dict]:
        """Top-k chunks for a query. mode: hybrid | vector | keyword."""
        where = build_where(**filters)
        vec = self.vector_search(query, candidates, where) if mode in ("hybrid", "vector") else []
        kw = self.keyword_search(query, candidates, where) if mode in ("hybrid", "keyword") else []
        return rrf_fuse([vec, kw])[:k]


# --- Command line ---------------------------------------------------------

def format_pages(hit: dict) -> str:
    if hit["page_end"] != hit["page_start"]:
        return f"p{hit['page_start']}-{hit['page_end']}"
    return f"p{hit['page_start']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search test.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    parser.add_argument("--collection", choices=["papers", "reports"])
    parser.add_argument("--category")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--lang", choices=["en", "ne"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    load_start = time.time()
    retriever = Retriever(device=args.device)
    loaded = time.time() - load_start

    start = time.time()
    hits = retriever.search(args.query, k=args.k, mode=args.mode, collection=args.collection,
                            category=args.category, year_min=args.year_min,
                            year_max=args.year_max, lang=args.lang)
    took = time.time() - start

    print(f"\nQuery: {args.query}  [{args.mode}]  "
          f"({took * 1000:.0f} ms search, {loaded:.1f} s model load)\n")
    for i, h in enumerate(hits, 1):
        found_by = ", ".join(f"{name} #{rank}" for name, rank in h["ranks"].items())
        print(f"{i}. {h['title'][:70]} ({h['year'] or 'n.d.'}) {format_pages(h)}  [{found_by}]")
        if h["section"]:
            print(f"   Section: {h['section'][:80]}")
        print(f"   {h['text'][:280].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()