"""
check_pdf_quality.py — Step 3 of nepal-env-rag.

Inspects every downloaded PDF and flags files that are broken, stubs,
or scans that need OCR before text extraction.

Input:  manifest.csv, data/raw/
Output: data/quality_report.csv

Usage:
    python check_pdf_quality.py               # full run
    python check_pdf_quality.py --limit 300   # quick test on a subset
    python check_pdf_quality.py --workers 8   # more CPU processes

Safe to run while download_drive.py is still going: files not yet
downloaded are reported as "missing". Rerun later to cover them.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf  # older PyMuPDF versions
    except ImportError:
        sys.exit("PyMuPDF is not installed. Run: pip install pymupdf")


# --- Configuration --------------------------------------------------------

MANIFEST_FILE = "manifest.csv"
RAW_DIR = os.path.join("data", "raw")
REPORT_FILE = os.path.join("data", "quality_report.csv")

# Thresholds. Review the first report and tune if needed.
TINY_BYTES = 20_000          # under this: likely an abstract or stub
MIN_CHARS_PER_PAGE = 200     # under this: no real text layer -> OCR
LOW_CHARS_PER_PAGE = 800     # under this: partial text (mixed scan)
DEFAULT_SAMPLE_PAGES = 10    # pages read per PDF, spread across the doc

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WINDOWS_BAD_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')

# Status values, in the order the summary prints them.
STATUSES = ("ok", "low_text", "needs_ocr", "stub", "encrypted", "broken", "missing")


# --- Data model -----------------------------------------------------------

@dataclass
class QualityResult:
    file_id: str
    path: str
    local_path: str
    collection: str
    category: str
    size_bytes: int = 0
    pages: int = 0
    sampled_pages: int = 0
    chars_per_page: int = 0
    devanagari_ratio: float = 0.0   # share of Nepali-script characters
    status: str = ""
    error: str = ""


# --- Paths (must match download_drive.py) ---------------------------------

def local_path_for(drive_path: str) -> str:
    """Map a Drive path to its sanitized location under data/raw/."""
    parts = [p for p in drive_path.strip("/").split("/") if p]
    safe = [WINDOWS_BAD_CHARS.sub("_", p).rstrip(" .") or "_" for p in parts]
    return os.path.join(RAW_DIR, *safe)


def long_path(path: str) -> str:
    """Bypass the 260-char path limit on Windows."""
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path)
    return abs_path if abs_path.startswith("\\\\?\\") else "\\\\?\\" + abs_path


# --- Planning -------------------------------------------------------------

def load_targets(limit: int | None) -> list[QualityResult]:
    """Unique, non-skipped files from the manifest (same rules as download)."""
    if not os.path.exists(MANIFEST_FILE):
        sys.exit(f"{MANIFEST_FILE} not found. Run build_manifest.py first.")

    with open(MANIFEST_FILE, newline="", encoding="utf-8-sig") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["path"])

    targets, seen_md5 = [], set()
    for row in rows:
        if row["skip"] == "True":
            continue
        if row["md5"]:
            if row["md5"] in seen_md5:
                continue  # duplicate content, checked once
            seen_md5.add(row["md5"])

        targets.append(QualityResult(
            file_id=row["file_id"],
            path=row["path"],
            local_path=local_path_for(row["path"]),
            collection=row["collection"],
            category=row["category"],
        ))

    return targets[:limit] if limit else targets


# --- Inspection (runs in worker processes) --------------------------------

def _init_worker() -> None:
    # MuPDF prints noisy warnings for slightly malformed PDFs; they are
    # harmless and would flood the console across 22k files.
    pymupdf.TOOLS.mupdf_display_errors(False)
    if hasattr(pymupdf.TOOLS, "mupdf_display_warnings"):  # newer versions only
        pymupdf.TOOLS.mupdf_display_warnings(False)


def _sample_indices(page_count: int, n: int) -> list[int]:
    """Up to n page indices spread evenly from first to last page."""
    if page_count <= n:
        return list(range(page_count))
    step = (page_count - 1) / (n - 1)
    return sorted({round(i * step) for i in range(n)})


def _classify(r: QualityResult) -> str:
    if r.pages == 0:
        return "broken"
    if r.chars_per_page < MIN_CHARS_PER_PAGE:
        return "needs_ocr"
    if r.size_bytes < TINY_BYTES:
        return "stub"
    if r.chars_per_page < LOW_CHARS_PER_PAGE:
        return "low_text"
    return "ok"


def inspect_pdf(args: tuple[QualityResult, int]) -> QualityResult:
    r, sample_pages = args
    path = long_path(r.local_path)

    if not os.path.exists(path):
        r.status = "missing"
        return r

    r.size_bytes = os.path.getsize(path)

    try:
        # Read via Python so the long-path prefix works on Windows.
        with open(path, "rb") as f:
            data = f.read()

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                r.status = "encrypted"
                return r

            r.pages = doc.page_count
            indices = _sample_indices(r.pages, sample_pages)
            text = "".join(doc[i].get_text() for i in indices)

        r.sampled_pages = len(indices)
        visible = [c for c in text if not c.isspace()]
        r.chars_per_page = len(visible) // max(len(indices), 1)
        if visible:
            nepali = sum(1 for c in visible if DEVANAGARI.match(c))
            r.devanagari_ratio = round(nepali / len(visible), 3)

        r.status = _classify(r)

    except Exception as e:  # corrupt / truncated / not really a PDF
        r.status = "broken"
        r.error = f"{type(e).__name__}: {e}"[:300]

    return r


# --- Reporting ------------------------------------------------------------

def write_report(results: list[QualityResult]) -> None:
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    tmp = REPORT_FILE + ".tmp"
    columns = [f.name for f in fields(QualityResult)]

    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    os.replace(tmp, REPORT_FILE)  # never leave a half-written report


def print_summary(results: list[QualityResult], seconds: float) -> None:
    total = len(results)
    counts = Counter(r.status for r in results)

    print(f"\nChecked {total} files in {seconds / 60:.1f} min -> {REPORT_FILE}\n")
    print(f"{'status':<12}{'files':>8}{'share':>9}")
    for status in STATUSES:
        n = counts.get(status, 0)
        if n:
            print(f"{status:<12}{n:>8}{n / total:>9.1%}")

    # Status by collection: papers and reports usually behave differently.
    print(f"\n{'collection':<12}" + "".join(f"{s:>11}" for s in STATUSES))
    for coll in sorted({r.collection for r in results}):
        by = Counter(r.status for r in results if r.collection == coll)
        print(f"{coll:<12}" + "".join(f"{by.get(s, 0):>11}" for s in STATUSES))

    nepali = sum(1 for r in results if r.devanagari_ratio >= 0.3)
    if nepali:
        print(f"\nMostly Nepali-script documents: {nepali} "
              "(embedding model must support Nepali)")

    hints = {
        "missing": "not downloaded yet: finish download_drive.py, then rerun",
        "needs_ocr": "scanned: route to OCR in the extraction step",
        "broken": "unreadable: see 'error' column; try re-downloading",
        "stub": "very small: review a few, likely drop",
    }
    notes = [f"  {s}: {msg}" for s, msg in hints.items() if counts.get(s)]
    if notes:
        print("\nNext:")
        print("\n".join(notes))


# --- Entry point ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check quality of downloaded PDFs.")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1),
                        help="parallel processes (default: CPU cores - 1)")
    parser.add_argument("--sample-pages", type=int, default=DEFAULT_SAMPLE_PAGES,
                        help="pages read per PDF (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="only check the first N files")
    args = parser.parse_args()

    targets = load_targets(args.limit)
    print(f"Checking {len(targets)} PDFs with {args.workers} workers...")

    start = time.time()
    results: list[QualityResult] = []
    jobs = ((t, args.sample_pages) for t in targets)

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.map(inspect_pdf, jobs, chunksize=16), 1):
            results.append(result)
            if i % 500 == 0:
                rate = i / (time.time() - start)
                print(f"  {i}/{len(targets)}  ({rate:.0f} files/s)")

    write_report(results)
    print_summary(results, time.time() - start)


if __name__ == "__main__":
    main()