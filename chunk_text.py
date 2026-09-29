"""
chunk_text.py — Step 5 of nepal-env-rag.

Splits extracted page text into retrieval chunks with context headers.

Strategy (benchmark-backed; see project report):
  - Structure-aware recursive splitting: section -> paragraph -> sentence -> words
  - Target ~512 tokens, hard max 768, min floor 100 (tiny tails are merged)
  - ~64-token sentence overlap between neighbouring chunks in a section
  - Chunks may span pages; page_start / page_end kept for citations
  - Context header (title, year, source, section) prepended for embedding
  - Skips OCR-needed, emptied, foreign-language and legacy-font pages,
    tables of contents and acknowledgement sections
  - Drops exact-duplicate chunks

Input:  data/extraction_report.csv, data/text/<collection>/<doc_id>.json
Output: data/chunks/<collection>/<category>.jsonl

Usage:
    python chunk_text.py --category 01_Floods_GLOFs   # pilot on one folder
    python chunk_text.py                              # everything
    python chunk_text.py --target 400 --overlap 48    # tune sizes

Requires `pip install tokenizers` for exact token counts (falls back to
an estimate if unavailable).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

# --- Configuration --------------------------------------------------------

EXTRACTION_REPORT = os.path.join("data", "extraction_report.csv")
TEXT_DIR = os.path.join("data", "text")
CHUNK_DIR = os.path.join("data", "chunks")

TOKENIZER_NAME = "BAAI/bge-m3"  # must match the embedding model

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Headings: numbered, chapter/annex, ALL CAPS, or common section names.
HEADING = re.compile(
    r"^("
    r"(?i:(chapter|section|part|annex(ure)?|appendix)\s+[\divxlc]+\b.*)"
    r"|\d+(\.\d+){0,3}\.?\s+[A-Z][^.!?]{2,80}"
    r"|[A-Z][A-Z0-9 ,&:()'/\-]{3,80}"
    r"|(?i:abstract|introduction|background|methods?|methodology|"
    r"materials\s+and\s+methods|study\s+area|results?|discussion|"
    r"results\s+and\s+discussion|conclusions?|recommendations?|"
    r"summary|executive\s+summary|key\s+findings)"
    r")\s*:?$"
)
NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
TOP_LEVEL_HEADING = re.compile(r"^(?i:chapter|part|annex|appendix)|^[A-Z0-9 ,&:()'/\-]+$")

# Sections whose content is noise for retrieval (whole-line titles only,
# so "Contents of heavy metals..." is NOT treated as a table of contents).
SKIP_SECTION = re.compile(
    r"^(\d+(\.\d+)*\.?\s*)?(?i:acknowledge?ments?|table\s+of\s+contents|contents|"
    r"list\s+of\s+(tables|figures|maps|boxes|abbreviations|acronyms)|"
    r"abbreviations(\s+and\s+acronyms)?|acronyms(\s+and\s+abbreviations)?)\s*:?\s*$"
)
SKIP_SECTION_MAX_CHARS = 4000  # safety cap: resume if a "skipped" section runs long
TOC_ENTRY = re.compile(r"\.{4,}\s*[\divxlc]+\b", re.I)  # "Introduction ..... 5"

# Sentence boundaries: ., !, ? and the Nepali purna viram (।).
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+(?=[\"'(\[]?[A-Z0-9\u0900-\u097F])")
# A split after these is a false boundary ("et al. 2015", "Fig. 3", "B. Shrestha").
ABBREVIATION = re.compile(
    r"(\b(?i:et al|e\.g|i\.e|fig|figs|eq|no|vol|dr|mr|mrs|ms|prof|approx|vs|cf|"
    r"etc|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)|\b[A-Z])\.$"
)


@dataclass(frozen=True)
class ChunkConfig:
    target: int = 512   # aim for this many tokens per chunk
    max: int = 768      # never exceed (except one unsplittable unit)
    min: int = 100      # smaller tails are merged into a neighbour
    overlap: int = 64   # tokens of trailing sentences repeated in next chunk


@dataclass
class Sentence:
    text: str
    tokens: int
    page: int
    new_paragraph: bool


@dataclass
class Segment:
    """Consecutive sentences under one section heading."""
    section: str
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sentences)


# --- Token counting -------------------------------------------------------

class TokenCounter:
    """Exact counts with the embedding model's tokenizer, else an estimate."""

    def __init__(self, name: str):
        self.name = name
        try:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_pretrained(name)
        except Exception:
            self._tok = None

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def count(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        if self._tok:
            return [len(e.ids) for e in self._tok.encode_batch(texts, add_special_tokens=False)]
        # Rough fallback: ~4 chars/token for Latin text, ~2 for Devanagari.
        return [
            max(1, (len(t) - len(DEVANAGARI.findall(t))) // 4 + len(DEVANAGARI.findall(t)) // 2)
            for t in texts
        ]


_counter: TokenCounter | None = None


def _init_worker(tokenizer_name: str) -> None:
    global _counter
    _counter = TokenCounter(tokenizer_name)


# --- Text structure -------------------------------------------------------

def is_heading(paragraph: str) -> bool:
    if len(paragraph) > 100 or len(paragraph.split()) > 12:
        return False
    if paragraph.rstrip().endswith((".", ",", ";")):
        return False
    return bool(HEADING.match(paragraph.strip()))


def heading_level(heading: str) -> int:
    """1 for chapters / ALL CAPS / '4 Results', 2+ for '4.2', '4.2.1'."""
    numbered = NUMBERED_HEADING.match(heading)
    if numbered:
        return numbered.group(1).count(".") + 1
    return 1 if TOP_LEVEL_HEADING.match(heading) else 2


def split_sentences(paragraph: str) -> list[str]:
    parts = SENTENCE_END.split(paragraph)
    merged: list[str] = []
    for part in parts:
        if merged and ABBREVIATION.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"  # undo a false boundary
        else:
            merged.append(part)
    return [s.strip() for s in merged if s.strip()]


def split_long(text: str, tokens: int, limit: int) -> list[str]:
    """Hard-split an over-long sentence (e.g. a flattened table) by words."""
    words = text.split()
    pieces = max(2, -(-tokens // limit))  # ceil division
    size = -(-len(words) // pieces)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


# --- Document -> segments -------------------------------------------------

def page_skip_reason(page: dict) -> str | None:
    if page.get("needs_ocr"):
        return "needs_ocr"
    if page.get("emptied_by_cleaning") or not page.get("text"):
        return "empty"
    if page.get("lang") == "other":
        return "other_language"
    if page.get("legacy_font"):
        return "legacy_font"
    return None


def build_segments(doc: dict, cfg: ChunkConfig, stats: Counter) -> list[Segment]:
    segments = [Segment(section="")]
    headings: dict[int, str] = {}
    skipping = False
    skipped_chars = 0  # characters dropped in the current skipped section

    for page in doc["pages"]:
        reason = page_skip_reason(page)
        if reason:
            stats[f"pages_skipped_{reason}"] += 1
            continue
        stats["pages_used"] += 1

        for paragraph in page["text"].split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) <= 100 and SKIP_SECTION.match(paragraph):
                skipping, skipped_chars = True, 0  # e.g. "Acknowledgements"
                stats["paragraphs_skipped_section"] += 1
                continue

            if is_heading(paragraph):
                skipping = False
                level = heading_level(paragraph)
                headings = {k: v for k, v in headings.items() if k < level}
                headings[level] = paragraph
                path = " > ".join(headings[k] for k in sorted(headings))
                segments.append(Segment(section=path))
                continue

            if skipping:
                skipped_chars += len(paragraph)
                if skipped_chars <= SKIP_SECTION_MAX_CHARS:
                    stats["paragraphs_skipped_section"] += 1
                    stats["chars_skipped_section"] += len(paragraph)
                    continue
                # Too long for acknowledgements/TOC: a missed heading. Resume.
                skipping = False
                stats["skip_sections_capped"] += 1
            if len(TOC_ENTRY.findall(paragraph)) >= 1 and len(paragraph) < 400:
                stats["paragraphs_skipped_toc"] += 1
                continue

            sentences = split_sentences(paragraph)
            counts = _counter.count(sentences)
            first = True
            for text, n in zip(sentences, counts):
                pieces = [text] if n <= cfg.target else split_long(text, n, cfg.target)
                piece_counts = [n] if len(pieces) == 1 else _counter.count(pieces)
                for piece, pn in zip(pieces, piece_counts):
                    segments[-1].sentences.append(Sentence(piece, pn, page["page"], first))
                    first = False

    return [s for s in segments if s.sentences]


# --- Segments -> chunks ---------------------------------------------------

def overlap_tail(sentences: list[Sentence], budget: int) -> list[Sentence]:
    tail, used = [], 0
    for s in reversed(sentences):
        if used + s.tokens > budget:
            break
        tail.insert(0, s)
        used += s.tokens
    return tail


def pack_segment(segment: Segment, cfg: ChunkConfig) -> list[list[Sentence]]:
    """Greedy packing to ~target tokens with sentence overlap."""
    chunks: list[list[Sentence]] = []
    current: list[Sentence] = []
    n_overlap = 0

    for s in segment.sentences:
        size = sum(x.tokens for x in current)
        if current and len(current) > n_overlap and size + s.tokens > cfg.target:
            chunks.append(current)
            current = overlap_tail(current, cfg.overlap)
            n_overlap = len(current)
        current.append(s)

    if current and len(current) > n_overlap:
        new_part = current[n_overlap:]
        new_tokens = sum(x.tokens for x in new_part)
        last_tokens = sum(x.tokens for x in chunks[-1]) if chunks else 0
        if chunks and new_tokens < cfg.min and last_tokens + new_tokens <= cfg.max:
            chunks[-1].extend(new_part)  # merge a tiny tail
        else:
            chunks.append(current)
    return chunks


def pack_document(segments: list[Segment], cfg: ChunkConfig) -> list[tuple[str, list[Sentence]]]:
    """Pack each section; merge tiny sections into a neighbour."""
    out: list[tuple[str, list[Sentence]]] = []
    carry: Segment | None = None

    for seg in segments:
        if carry:  # a tiny previous section rides along with this one
            seg = Segment(seg.section, carry.sentences + seg.sentences)
            carry = None

        if seg.tokens < cfg.min:
            prev_tokens = sum(s.tokens for s in out[-1][1]) if out else None
            if prev_tokens is not None and prev_tokens + seg.tokens <= cfg.max:
                out[-1][1].extend(seg.sentences)
            else:
                carry = seg
            continue

        out.extend((seg.section, chunk) for chunk in pack_segment(seg, cfg))

    if carry:
        out.append((carry.section, carry.sentences))
    return out


# --- Output records -------------------------------------------------------

def readable(name: str) -> str:
    """'01_Floods_GLOFs' -> 'Floods GLOFs'; 'Some_Report.pdf' -> 'Some Report'."""
    name = os.path.splitext(name)[0]
    name = re.sub(r"^\d+_", "", name)
    return name.replace("_", " ").strip()


def document_context(doc: dict) -> dict:
    is_paper = doc["collection"] == "papers"
    title = doc.get("title") or readable(os.path.basename(doc["path"]))
    source = (f"Research paper, topic: {readable(doc['category'])}" if is_paper
              else f"Report by {readable(doc['category'])}")
    return {"title": title, "source": source, "year": doc.get("year", "")}


def join_sentences(sentences: list[Sentence]) -> str:
    parts: list[str] = []
    for i, s in enumerate(sentences):
        if i and s.new_paragraph:
            parts.append("\n\n")
        elif i:
            parts.append(" ")
        parts.append(s.text)
    return "".join(parts)


def make_record(doc: dict, ctx: dict, index: int, section: str,
                sentences: list[Sentence]) -> dict:
    body = join_sentences(sentences)
    header = f"Document: {ctx['title']}" + (f" ({ctx['year']})" if ctx["year"] else "")
    header += f"\nSource: {ctx['source']}"
    if section:
        header += f"\nSection: {section}"

    visible = [c for c in body if not c.isspace()]
    nepali = sum(1 for c in visible if DEVANAGARI.match(c)) / max(len(visible), 1)
    body_tokens = sum(s.tokens for s in sentences)
    header_tokens = _counter.count([header])[0] + 2  # +2 for the blank line

    return {
        "chunk_id": f"{doc['doc_id']}:{index:04d}",
        "doc_id": doc["doc_id"],
        "chunk_index": index,
        "collection": doc["collection"],
        "category": doc["category"],
        "title": ctx["title"],
        "year": ctx["year"],
        "author": doc.get("author", ""),
        "openalex_id": doc.get("openalex_id", ""),
        "path": doc["path"],
        "section": section,
        "page_start": sentences[0].page,
        "page_end": sentences[-1].page,
        "lang": "ne" if nepali >= 0.3 else "en",
        "tokens": body_tokens,
        "embed_tokens": body_tokens + header_tokens,
        "text": body,                        # shown to the LLM / user
        "embed_text": f"{header}\n\n{body}",  # what gets embedded
    }


def chunk_document(args: tuple[str, ChunkConfig]) -> tuple[list[dict], Counter]:
    path, cfg = args
    stats: Counter = Counter()
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        ctx = document_context(doc)
        packed = pack_document(build_segments(doc, cfg, stats), cfg)
        records = [make_record(doc, ctx, i, sec, sents) for i, (sec, sents) in enumerate(packed)]
        stats["docs_ok"] += 1
        if not records:
            stats["docs_no_chunks"] += 1
        return records, stats
    except Exception as e:
        stats["docs_failed"] += 1
        print(f"  FAILED {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return [], stats


# --- Planning and output --------------------------------------------------

def load_document_paths(category: str | None, limit: int | None) -> list[str]:
    if not os.path.exists(EXTRACTION_REPORT):
        sys.exit(f"{EXTRACTION_REPORT} not found. Run extract_text.py first.")
    with open(EXTRACTION_REPORT, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    if category:
        rows = [r for r in rows if r["category"] == category]
    rows.sort(key=lambda r: r["path"])
    paths = [os.path.join(TEXT_DIR, r["collection"] or "other", f"{r['doc_id']}.json")
             for r in rows]
    return paths[:limit] if limit else paths


def dedupe_key(text: str) -> bytes:
    return hashlib.blake2b(" ".join(text.lower().split()).encode(), digest_size=16).digest()


class ShardWriter:
    """One JSONL file per collection/category, written atomically."""

    def __init__(self, root: str):
        self.root = root
        self.files: dict[str, object] = {}
        self.tmp_paths: dict[str, str] = {}

    def write(self, record: dict) -> None:
        key = os.path.join(record["collection"] or "other", f"{record['category'] or 'root'}.jsonl")
        if key not in self.files:
            final = os.path.join(self.root, key)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            self.tmp_paths[key] = final + ".tmp"
            self.files[key] = open(self.tmp_paths[key], "w", encoding="utf-8")
        self.files[key].write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for key, fh in self.files.items():
            fh.close()
            os.replace(self.tmp_paths[key], os.path.join(self.root, key))


def print_summary(records_tokens: list[int], stats: Counter, embed_tokens: int,
                  exact: bool, seconds: float) -> None:
    n = len(records_tokens)
    print(f"\nDone in {seconds / 60:.1f} min -> {CHUNK_DIR}")
    print(f"  documents:          {stats['docs_ok']:,} ok, {stats['docs_failed']:,} failed, "
          f"{stats['docs_no_chunks']:,} with no usable text")
    print(f"  chunks written:     {n:,}  (duplicates dropped: {stats['duplicates']:,})")
    if n:
        ordered = sorted(records_tokens)
        print(f"  tokens per chunk:   mean {statistics.mean(ordered):.0f}, "
              f"median {ordered[n // 2]}, p95 {ordered[int(n * 0.95)]}, max {ordered[-1]}")
        small = sum(1 for t in ordered if t < 100)
        print(f"  chunks < 100 tok:   {small:,} ({small / n:.1%})")
    print(f"  tokens to embed:    {embed_tokens:,} "
          f"({'exact' if exact else 'ESTIMATED: pip install tokenizers'})")
    print(f"  pages used:         {stats['pages_used']:,}")
    for key in sorted(k for k in stats if k.startswith("pages_skipped_")):
        print(f"  pages skipped ({key.removeprefix('pages_skipped_')}): {stats[key]:,}")
    print(f"  paragraphs skipped: {stats['paragraphs_skipped_section']:,} "
          f"({stats['chars_skipped_section']:,} chars) in acknowledgements/TOC sections, "
          f"{stats['paragraphs_skipped_toc']:,} TOC lines")
    if stats["skip_sections_capped"]:
        print(f"  skip sections capped (resumed early): {stats['skip_sections_capped']:,}")


# --- Entry point ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk extracted text for RAG.")
    parser.add_argument("--category", help="only one folder, e.g. 01_Floods_GLOFs")
    parser.add_argument("--limit", type=int, help="only the first N documents")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--target", type=int, default=ChunkConfig.target)
    parser.add_argument("--max", type=int, default=ChunkConfig.max)
    parser.add_argument("--min", type=int, default=ChunkConfig.min)
    parser.add_argument("--overlap", type=int, default=ChunkConfig.overlap)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    args = parser.parse_args()

    cfg = ChunkConfig(args.target, args.max, args.min, args.overlap)
    counter = TokenCounter(args.tokenizer)  # also downloads it once for workers
    if not counter.exact:
        print("WARNING: tokenizer unavailable, token counts are estimates. "
              "Run: pip install tokenizers")

    paths = load_document_paths(args.category, args.limit)
    print(f"Chunking {len(paths):,} documents with {args.workers} workers "
          f"(target {cfg.target}, max {cfg.max}, min {cfg.min}, overlap {cfg.overlap})...")

    # A full run rebuilds everything; a filtered run only replaces its shards.
    if not args.category and not args.limit and os.path.isdir(CHUNK_DIR):
        shutil.rmtree(CHUNK_DIR)

    start = time.time()
    writer = ShardWriter(CHUNK_DIR)
    seen: set[bytes] = set()
    stats: Counter = Counter()
    token_counts: list[int] = []
    embed_tokens = 0

    jobs = ((p, cfg) for p in paths)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.tokenizer,)) as pool:
        for i, (records, doc_stats) in enumerate(pool.map(chunk_document, jobs, chunksize=8), 1):
            stats.update(doc_stats)
            for rec in records:
                key = dedupe_key(rec["text"])
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                writer.write(rec)
                token_counts.append(rec["tokens"])
                embed_tokens += rec["embed_tokens"]
            if i % 1000 == 0:
                print(f"  {i:,}/{len(paths):,} docs, {len(token_counts):,} chunks")

    writer.close()
    print_summary(token_counts, stats, embed_tokens, counter.exact, time.time() - start)


if __name__ == "__main__":
    main()