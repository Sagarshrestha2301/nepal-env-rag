"""
download_drive.py

STEP 2 of the nepal-env-rag pipeline.

What this script does
---------------------
Reads manifest.csv (made by build_manifest.py) and downloads every file
to data/raw/, keeping the same folder structure as on Google Drive:

    data/raw/papers/01_Floods_GLOFs/2019_....pdf
    data/raw/reports/ICIMOD/....pdf

Key features
------------
- SKIPS rows marked skip=True (the _wrong_pdf folder).
- DEDUPLICATES: files with the same md5 checksum are identical, so each
  unique file is downloaded only once (saves ~538 downloads).
- RESUMES: files already on disk with the right size are skipped, so you
  can stop (Ctrl+C) and rerun any time; it continues where it stopped.
- VERIFIES: each download's md5 is checked against Drive's checksum.
- RETRIES: rate-limit / quota / server errors are retried with backoff.
- PARALLEL: downloads a few files at once (WORKERS setting).
- LOGS: every result goes to download_log.csv; failures to failed.csv.

Run
---
    python download_drive.py

Rerun the same command to retry failures or continue after stopping.
"""

import csv
import hashlib
import os
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

MANIFEST_FILE = "manifest.csv"
OUTPUT_DIR = os.path.join("data", "raw")
LOG_FILE = "download_log.csv"      # one line per file attempted this run
FAILED_FILE = "failed.csv"         # only the failures (rewritten each run)

# Parallel downloads. 4 is safe; up to 8 is usually fine.
# Higher values risk hitting Google's rate limits (more 403/429 errors).
WORKERS = 12

# Retry settings for temporary errors.
MAX_RETRIES = 6          # attempts per file before giving up
BASE_BACKOFF_SEC = 2     # wait 2, 4, 8, 16... seconds (+ random jitter)

# Download in 10 MB pieces (large reports won't use too much memory).
CHUNK_SIZE = 10 * 1024 * 1024

# Keep at least this much free disk space as a safety margin.
DISK_MARGIN_BYTES = 2 * 1024**3   # 2 GB

# HTTP status codes worth retrying (rate limits + server errors).
RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}

# 403 reasons that will NEVER succeed on retry (permission problems).
PERMANENT_403_REASONS = ("cannotDownloadFile", "insufficientFilePermissions")


# ------------------------------------------------------------------
# Login (same as the other scripts)
# ------------------------------------------------------------------

def authenticate():
    """Return valid credentials, re-logging in if the token expired."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            print("Saved login expired. Logging in again...\n")
            os.remove(TOKEN_FILE)
            creds = None
    if not creds or not creds.valid:
        if not os.path.exists(CREDENTIALS_FILE):
            sys.exit(f"{CREDENTIALS_FILE} not found.")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    return creds


# The Google API client is NOT thread-safe, so each download thread
# builds and keeps its own "service" object in thread-local storage.
_thread_local = threading.local()


def get_service(creds):
    if not hasattr(_thread_local, "service"):
        _thread_local.service = build(
            "drive", "v3", credentials=creds, cache_discovery=False
        )
    return _thread_local.service


# ------------------------------------------------------------------
# Paths (Windows-safe)
# ------------------------------------------------------------------

# Characters Windows does not allow in file or folder names.
_BAD_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')


def local_path_for(drive_path):
    """Turn a Drive path like '/papers/01_Floods/x.pdf' into a safe
    local path under data/raw/, replacing characters Windows rejects."""
    parts = [p for p in drive_path.strip("/").split("/") if p]
    safe_parts = [_BAD_CHARS.sub("_", p).rstrip(" .") or "_" for p in parts]
    return os.path.join(OUTPUT_DIR, *safe_parts)


def long_path(path):
    """Windows normally limits paths to 260 characters. Some paper
    filenames are long, so we add the '\\\\?\\' prefix on Windows,
    which lifts that limit. Does nothing on Mac/Linux."""
    if os.name == "nt":
        abs_path = os.path.abspath(path)
        if not abs_path.startswith("\\\\?\\"):
            return "\\\\?\\" + abs_path
        return abs_path
    return path


def file_md5(path):
    """Compute a file's md5 checksum (read in chunks to save memory)."""
    h = hashlib.md5()
    with open(long_path(path), "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------
# Planning: which files to download
# ------------------------------------------------------------------

def load_plan():
    """Read manifest.csv and decide what to download.

    Returns (to_download, already_done, duplicates, skipped) lists.
    """
    if not os.path.exists(MANIFEST_FILE):
        sys.exit(f"{MANIFEST_FILE} not found. Run build_manifest.py first.")

    with open(MANIFEST_FILE, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Sort by path so the choice of which duplicate to keep is stable
    # (the same copy is chosen every run, which keeps resume working).
    rows.sort(key=lambda r: r["path"])

    to_download, already_done, duplicates, skipped = [], [], [], []
    seen_md5 = set()

    for row in rows:
        # 1. Rows marked skip (e.g. in _wrong_pdf) are never downloaded.
        if row["skip"] == "True":
            skipped.append(row)
            continue

        # 2. Same md5 as a file we already planned = identical duplicate.
        md5 = row["md5"]
        if md5 and md5 in seen_md5:
            duplicates.append(row)
            continue
        if md5:
            seen_md5.add(md5)

        # 3. Already on disk with the correct size = done (resume).
        row["local_path"] = local_path_for(row["path"])
        expected = int(row["size_bytes"]) if row["size_bytes"] else None
        lp = long_path(row["local_path"])
        if expected is not None and os.path.exists(lp) and os.path.getsize(lp) == expected:
            already_done.append(row)
            continue

        to_download.append(row)

    return to_download, already_done, duplicates, skipped


def check_disk_space(rows):
    """Stop early if the remaining downloads won't fit on the disk."""
    needed = sum(int(r["size_bytes"] or 0) for r in rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    free = shutil.disk_usage(OUTPUT_DIR).free
    print(f"Space needed: {human_size(needed)}   Free on disk: {human_size(free)}")
    if needed + DISK_MARGIN_BYTES > free:
        sys.exit(
            "Not enough disk space. Free up space, or change OUTPUT_DIR to "
            "a bigger drive (e.g. an external disk), then rerun."
        )


# ------------------------------------------------------------------
# Downloading one file
# ------------------------------------------------------------------

def download_one(creds, row):
    """Download a single file with retries. Returns (status, message).

    status is one of: "ok", "failed".
    The file is written to '<name>.part' first, then renamed when done,
    so an interrupted download never looks like a finished file.
    """
    final_path = row["local_path"]
    part_path = final_path + ".part"
    os.makedirs(long_path(os.path.dirname(final_path)), exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            service = get_service(creds)
            request = service.files().get_media(
                fileId=row["file_id"], supportsAllDrives=True
            )
            with open(long_path(part_path), "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            # Verify content matches what Drive says (catches corruption).
            if row["md5"] and file_md5(part_path) != row["md5"]:
                raise ValueError("md5 mismatch (corrupted download)")

            # Success: move .part to the real filename.
            os.replace(long_path(part_path), long_path(final_path))
            return "ok", ""

        except HttpError as e:
            status = e.resp.status
            reason = str(e)
            # Permission problems: retrying won't help.
            if status == 403 and any(r in reason for r in PERMANENT_403_REASONS):
                return "failed", f"403 not downloadable: {reason[:200]}"
            if status == 404:
                return "failed", "404 file no longer exists on Drive"
            if status not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                return "failed", f"HTTP {status}: {reason[:200]}"
            message = f"HTTP {status}"

        except Exception as e:  # network drops, timeouts, md5 mismatch
            if attempt == MAX_RETRIES:
                return "failed", f"{type(e).__name__}: {e}"
            message = type(e).__name__

        # Exponential backoff with jitter before the next attempt.
        wait = BASE_BACKOFF_SEC * 2 ** (attempt - 1) + random.uniform(0, 1)
        print(f"    retry {attempt}/{MAX_RETRIES - 1} in {wait:.0f}s "
              f"({message}): {row['name'][:60]}")
        time.sleep(wait)

    return "failed", "unknown"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def human_size(num_bytes):
    num_bytes = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def main():
    to_download, already_done, duplicates, skipped = load_plan()

    print("Plan:")
    print(f"  already downloaded:  {len(already_done)}")
    print(f"  to download now:     {len(to_download)}")
    print(f"  duplicates (skipped): {len(duplicates)}")
    print(f"  marked skip:         {len(skipped)}\n")

    if not to_download:
        print("Nothing to download. All files are already in", OUTPUT_DIR)
        return

    check_disk_space(to_download)

    creds = authenticate()
    total_bytes = sum(int(r["size_bytes"] or 0) for r in to_download)
    done_bytes = 0
    ok_count = 0
    failures = []
    start = time.time()

    # Open the log in append mode so history from earlier runs is kept.
    new_log = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as log_f:
        log = csv.writer(log_f)
        if new_log:
            log.writerow(["time", "status", "file_id", "path", "message"])

        print(f"\nDownloading {len(to_download)} files with {WORKERS} workers...")
        print("(Safe to stop with Ctrl+C; rerun to continue.)\n")

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(download_one, creds, r): r for r in to_download}

            for i, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                status, message = future.result()
                log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), status,
                              row["file_id"], row["path"], message])
                log_f.flush()  # write immediately in case of a crash

                if status == "ok":
                    ok_count += 1
                    done_bytes += int(row["size_bytes"] or 0)
                else:
                    failures.append({**row, "error": message})
                    print(f"  FAILED: {row['path']}  ({message[:80]})")

                # Progress line every 50 files (and at the end).
                if i % 50 == 0 or i == len(to_download):
                    elapsed = time.time() - start
                    speed = done_bytes / elapsed if elapsed else 0
                    left = (total_bytes - done_bytes) / speed if speed else 0
                    print(
                        f"  {i}/{len(to_download)} files | "
                        f"{human_size(done_bytes)} / {human_size(total_bytes)} | "
                        f"{human_size(speed)}/s | ~{left / 60:.0f} min left"
                    )

    # Save failures so they're easy to inspect. Rerunning the script
    # automatically retries them (they're not on disk yet).
    with open(FAILED_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "path", "size_bytes", "error"])
        writer.writeheader()
        for r in failures:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    minutes = (time.time() - start) / 60
    print(f"\nFinished in {minutes:.1f} min: {ok_count} ok, {len(failures)} failed.")
    if failures:
        print(f"See {FAILED_FILE}. Rerun this script to retry them.")
    else:
        print("All files downloaded. Next step: PDF quality check.")


if __name__ == "__main__":
    main()