"""
extract_text.py — Step 4 of nepal-env-rag.

Extracts and cleans page-level text from PDFs that passed the quality
check. Writes one JSON file per document, ready for chunking.

Input:  data/quality_report.csv, manifest.csv, data/raw/
Output: data/text/<collection>/<file_id>.json
        data/extraction_report.csv

Cleaning:
  - Unicode normalization (fixes ligatures like "ﬁ")
  - Rejoins words hyphenated across line breaks
  - Removes page numbers and repeated headers/footers
  - Drops reference lists (noise for retrieval)
  - Flags empty pages (for OCR later) and legacy Nepali fonts (Preeti etc.)

Usage:
    python extract_text.py --category 01_Floods_GLOFs   # pilot on one folder
    python extract_text.py --limit 200                  # quick test
    python extract_text.py                              # everything
    python extract_text.py --force                      # redo existing files

Resumable: documents already extracted are skipped unless --force.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        sys.exit("PyMuPDF is not installed. Run: pip install pymupdf")


# --- Configuration --------------------------------------------------------

MANIFEST_FILE = "manifest.csv"
QUALITY_FILE = os.path.join("data", "quality_report.csv")
TEXT_DIR = os.path.join("data", "text")
REPORT_FILE = os.path.join("data", "extraction_report.csv")

DEFAULT_STATUSES = ("ok", "low_text")  # needs_ocr handled in a later step

MIN_PAGE_CHARS = 50            # below this a page is treated as empty
HEADER_MIN_PAGES = 4           # need this many pages to detect headers
HEADER_REPEAT_SHARE = 0.5      # edge text on >=50% of pages = header/footer
NEPALI_PAGE_SHARE = 0.3        # Devanagari share to label a page "ne"
LEGACY_TOKEN_SHARE = 0.15      # share of garbled tokens to flag legacy font
PROSE_BLOCKS_TO_RESUME = 3     # long non-citation blocks that end a ref list
PROSE_MIN_CHARS = 300

# Non-Unicode Nepali fonts: text extracts as Latin gibberish ("g]kfn").
LEGACY_FONT_NAMES = ("preeti", "kantipur", "himalb", "sagarmatha", "fontasy")

PAGE_NUMBER = re.compile(
    r"^\s*(page\s*)?(\d+|[ivxlc]{1,6})(\s*(of|/)\s*\d+)?\s*$", re.I
)
REFERENCE_HEADING = re.compile(
    r"^\s*(\d+\.?\s*)?(references?( cited)?|bibliography|literature cited|"
    r"works cited|reference list|citations)\s*:?\s*$",
    re.I,
)
HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")

# Inside a reference list: lines that look like citations.
CITATION_HINT = re.compile(
    r"\b(19|20)\d{2}[a-z]?\b|\bet al\b|doi\.org|\bdoi:|https?://|\bvol\.|\bpp?\.\s*\d",
    re.I,
)
# Headings that start a new section, ending a reference list.
SECTION_HEADING = re.compile(
    r"^\s*("
    r"(?i:(chapter|section|part)\s+[\divxlc]+\b.*)"  # Chapter 4 ...
    r"|(?i:(annex(ure)?|appendix|appendices)\b.*)"   # Annex 1 ...
    r"|\d+(\.\d+)*\.?\s+[A-Z][^.]{2,80}"            # 4.2 Methods
    r"|[A-Z][A-Z0-9 ,&:()\-]{4,80}"                 # INTRODUCTION
    r")\s*$"
)
LEGACY_TOKEN = re.compile(r"[A-Za-z][\]\[;{}|][A-Za-z]")  # e.g. g]kfn
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
CJK = re.compile(r"[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")  # ja/zh/ko

# Years that mean "unknown" in the source filenames (1970 = epoch default).
PLACEHOLDER_YEARS = {"0", "1970"}


# --- Data model -----------------------------------------------------------

@dataclass
class ExtractResult:
    doc_id: str
    path: str
    collection: str
    category: str
    pages: int = 0
    text_pages: int = 0
    empty_pages: int = 0          # no text layer: candidates for OCR
    emptied_pages: int = 0        # had text, all removed by cleaning
    legacy_font_pages: int = 0    # need Preeti -> Unicode conversion
    nepali_pages: int = 0
    other_lang_pages: int = 0     # e.g. Japanese; excluded at chunking
    references_start_page: int | None = None   # first reference list
    reference_sections: int = 0                # lists found (per chapter)
    reference_blocks_removed: int = 0
    chars: int = 0
    status: str = ""              # ok | failed
    error: str = ""


# --- Paths ----------------------------------------------------------------

def long_path(path: str) -> str:
    """Bypass the 260-char path limit on Windows."""
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path)
    return abs_path if abs_path.startswith("\\\\?\\") else "\\\\?\\" + abs_path


def output_path(collection: str, doc_id: str) -> str:
    return os.path.join(TEXT_DIR, collection or "other", f"{doc_id}.json")


# --- Text cleaning --------------------------------------------------------

def clean_block(text: str) -> str:
    """Normalize one text block into a single clean paragraph."""
    text = unicodedata.normalize("NFKC", text)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def edge_key(text: str) -> str:
    """Comparable form of header/footer text (page numbers ignored)."""
    return re.sub(r"\d+", "#", text.lower()).strip()[:80]


def find_repeated_edges(pages_blocks: list[list[tuple]]) -> set[str]:
    """Top/bottom blocks that repeat across many pages = headers/footers."""
    if len(pages_blocks) < HEADER_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for blocks in pages_blocks:
        if not blocks:
            continue
        top = min(blocks, key=lambda b: b[1])
        bottom = max(blocks, key=lambda b: b[3])
        counts.update({edge_key(top[4]), edge_key(bottom[4])})

    threshold = max(3, HEADER_REPEAT_SHARE * len(pages_blocks))
    return {k for k, n in counts.items() if k and n >= threshold}


def has_legacy_font(page, text: str) -> bool:
    """Detect Preeti-style fonts by font name or garbled-token pattern."""
    font_names = " ".join(f[3] for f in page.get_fonts()).lower()
    if any(name in font_names for name in LEGACY_FONT_NAMES):
        return True
    tokens = text.split()
    if len(tokens) < 20:
        return False
    garbled = sum(1 for t in tokens if LEGACY_TOKEN.search(t))
    return garbled / len(tokens) >= LEGACY_TOKEN_SHARE


def page_language(text: str) -> str:
    """'ne' (Nepali), 'en' (Latin script), 'other' (e.g. Japanese), or ''."""
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return ""
    if sum(1 for c in visible if CJK.match(c)) / len(visible) >= 0.2:
        return "other"
    share = sum(1 for c in visible if DEVANAGARI.match(c)) / len(visible)
    return "ne" if share >= NEPALI_PAGE_SHARE else "en"


def is_reference_heading(block_text: str) -> bool:
    """True if the block is, or starts with, a reference-list heading.
    Handles headings merged with the first citation in one block."""
    first_line = block_text.strip().split("\n", 1)[0]
    return bool(REFERENCE_HEADING.match(first_line))


def is_section_heading(block_text: str) -> bool:
    """True if the block starts a new section (ends a reference list)."""
    first_line = block_text.strip().split("\n", 1)[0].strip()
    if len(first_line) > 100 or CITATION_HINT.search(first_line):
        return False
    return bool(SECTION_HEADING.match(first_line))


def is_prose(block_text: str) -> bool:
    """Long paragraph with no citation markers: body text, not a reference."""
    return len(block_text) >= PROSE_MIN_CHARS and not CITATION_HINT.search(block_text)


def clean_year(year: str) -> str:
    return "" if year.strip() in PLACEHOLDER_YEARS else year.strip()


# --- Extraction (runs in worker processes) --------------------------------

def _init_worker() -> None:
    pymupdf.TOOLS.mupdf_display_errors(False)
    if hasattr(pymupdf.TOOLS, "mupdf_display_warnings"):
        pymupdf.TOOLS.mupdf_display_warnings(False)


def extract_document(doc: dict) -> ExtractResult:
    result = ExtractResult(
        doc_id=doc["file_id"],
        path=doc["path"],
        collection=doc["collection"],
        category=doc["category"],
    )

    try:
        with open(long_path(doc["local_path"]), "rb") as f:
            data = f.read()

        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            # Text blocks only (type 0); skip image blocks.
            pages_blocks = [
                [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
                for page in pdf
            ]
            repeated = find_repeated_edges(pages_blocks)

            pages_out = []
            in_references = False
            pending_prose: list[str] = []  # prose seen inside a ref list

            for index, (page, blocks) in enumerate(zip(pdf, pages_blocks)):
                paragraphs = []
                for block in blocks:
                    raw = block[4]
                    if edge_key(raw) in repeated or PAGE_NUMBER.match(raw):
                        continue

                    if is_reference_heading(raw):
                        in_references, pending_prose = True, []
                        result.reference_sections += 1
                        if result.references_start_page is None:
                            result.references_start_page = index + 1
                        continue

                    if in_references:
                        if is_section_heading(raw):
                            paragraphs.extend(pending_prose)  # new section begins
                            in_references, pending_prose = False, []
                        elif is_prose(raw):
                            pending_prose.append(clean_block(raw))
                            if len(pending_prose) < PROSE_BLOCKS_TO_RESUME:
                                continue
                            # Sustained body text: the list ended; keep it all.
                            paragraphs.extend(pending_prose)
                            in_references, pending_prose = False, []
                            continue
                        else:
                            # A citation: any prose buffered so far was noise.
                            result.reference_blocks_removed += 1 + len(pending_prose)
                            pending_prose = []
                            continue

                    cleaned = clean_block(raw)
                    if cleaned:
                        paragraphs.append(cleaned)

                text = "\n\n".join(paragraphs)
                # OCR only if the PDF itself has no text layer on this page.
                # A page emptied by cleaning (e.g. all references) is not a scan.
                raw_chars = sum(len(b[4].strip()) for b in blocks)
                no_text_layer = raw_chars < MIN_PAGE_CHARS
                emptied = not no_text_layer and len(text) < MIN_PAGE_CHARS
                legacy = len(text) >= MIN_PAGE_CHARS and has_legacy_font(page, text)

                pages_out.append({
                    "page": index + 1,
                    "text": text,
                    "chars": len(text),
                    "lang": page_language(text),
                    "needs_ocr": no_text_layer,
                    "emptied_by_cleaning": emptied,
                    "legacy_font": legacy,
                })

            result.pages = pdf.page_count

        record = {
            "doc_id": doc["file_id"],
            "path": doc["path"],
            "collection": doc["collection"],
            "category": doc["category"],
            "year": clean_year(doc.get("year", "")),
            "year_raw": doc.get("year", ""),
            "author": doc.get("author", ""),
            "title": doc.get("title", ""),
            "openalex_id": doc.get("openalex_id", ""),
            "quality_status": doc["status"],
            "page_count": result.pages,
            "references_start_page": result.references_start_page,
            "reference_sections": result.reference_sections,
            "pages": pages_out,
        }
        write_json(output_path(doc["collection"], doc["file_id"]), record)

        result.text_pages = sum(1 for p in pages_out if p["chars"] >= MIN_PAGE_CHARS)
        result.empty_pages = sum(1 for p in pages_out if p["needs_ocr"])
        result.emptied_pages = sum(1 for p in pages_out if p["emptied_by_cleaning"])
        result.legacy_font_pages = sum(1 for p in pages_out if p["legacy_font"])
        result.nepali_pages = sum(1 for p in pages_out if p["lang"] == "ne")
        result.other_lang_pages = sum(1 for p in pages_out if p["lang"] == "other")
        result.chars = sum(p["chars"] for p in pages_out)
        result.status = "ok"

    except Exception as e:
        result.status = "failed"
        result.error = f"{type(e).__name__}: {e}"[:300]

    return result


def write_json(path: str, record: dict) -> None:
    """Atomic write: a crash never leaves a half-written file."""
    os.makedirs(long_path(os.path.dirname(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(long_path(tmp), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(long_path(tmp), long_path(path))


# --- Planning -------------------------------------------------------------

def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run the earlier pipeline steps first.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_documents(statuses: set[str], category: str | None,
                   limit: int | None, force: bool) -> tuple[list[dict], int]:
    """Documents to extract, joined with manifest metadata."""
    manifest = {r["file_id"]: r for r in read_csv(MANIFEST_FILE)}
    docs, skipped = [], 0

    for row in read_csv(QUALITY_FILE):
        if row["status"] not in statuses:
            continue
        if category and row["category"] != category:
            continue
        if not force and os.path.exists(
            long_path(output_path(row["collection"], row["file_id"]))
        ):
            skipped += 1
            continue
        docs.append({**manifest.get(row["file_id"], {}), **row})

    return (docs[:limit] if limit else docs), skipped


# --- Reporting ------------------------------------------------------------

def update_report(results: list[ExtractResult]) -> list[dict]:
    """Merge this run's results into the existing report (keyed by doc_id)."""
    existing = {}
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, newline="", encoding="utf-8-sig") as f:
            existing = {r["doc_id"]: r for r in csv.DictReader(f)}
    for r in results:
        existing[r.doc_id] = asdict(r)

    rows = list(existing.values())
    columns = [f.name for f in fields(ExtractResult)]
    tmp = REPORT_FILE + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, REPORT_FILE)
    return rows


def print_summary(results: list[ExtractResult], skipped: int, seconds: float) -> None:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status == "failed"]
    total_chars = sum(r.chars for r in ok)

    print(f"\nExtracted {len(ok)} docs in {seconds / 60:.1f} min "
          f"({skipped} already done, {len(failed)} failed)")
    print(f"  pages with text:      {sum(r.text_pages for r in ok):,}")
    print(f"  empty pages (OCR):    {sum(r.empty_pages for r in ok):,}")
    print(f"  emptied by cleaning:  {sum(r.emptied_pages for r in ok):,} (ref lists, headers)")
    print(f"  Nepali pages:         {sum(r.nepali_pages for r in ok):,}")
    print(f"  other-language pages: {sum(r.other_lang_pages for r in ok):,}")
    print(f"  legacy-font docs:     {sum(1 for r in ok if r.legacy_font_pages):,}")
    print(f"  references removed:   {sum(1 for r in ok if r.references_start_page):,} docs, "
          f"{sum(r.reference_sections for r in ok):,} lists, "
          f"{sum(r.reference_blocks_removed for r in ok):,} blocks")
    multi = sum(1 for r in ok if r.reference_sections > 1)
    print(f"  docs with per-chapter reference lists: {multi:,}")
    print(f"  text size:            {total_chars / 1e6:,.1f}M chars "
          f"(~{total_chars / 4 / 1e6:,.1f}M tokens)")

    if failed:
        print(f"\nFailures listed in {REPORT_FILE} (status=failed).")


# --- Entry point ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and clean PDF text.")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    parser.add_argument("--statuses", default=",".join(DEFAULT_STATUSES),
                        help="quality statuses to include (default: %(default)s)")
    parser.add_argument("--category", help="only one folder, e.g. 01_Floods_GLOFs")
    parser.add_argument("--limit", type=int, help="only the first N documents")
    parser.add_argument("--force", action="store_true", help="re-extract existing")
    args = parser.parse_args()

    statuses = {s.strip() for s in args.statuses.split(",")}
    docs, skipped = load_documents(statuses, args.category, args.limit, args.force)
    print(f"Extracting {len(docs)} documents with {args.workers} workers "
          f"({skipped} already done)...")

    start = time.time()
    results: list[ExtractResult] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.map(extract_document, docs, chunksize=8), 1):
            results.append(result)
            if i % 500 == 0:
                print(f"  {i}/{len(docs)}  ({i / (time.time() - start):.0f} docs/s)")

    update_report(results)
    print_summary(results, skipped, time.time() - start)


if __name__ == "__main__":
    main()