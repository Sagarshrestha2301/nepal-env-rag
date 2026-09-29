"""
build_manifest.py

STEP 1 of the nepal-env-rag pipeline.

What this script does
---------------------
Walks the whole shared "LLM Reports" folder on Google Drive and writes
ONE row per file into manifest.csv. It does NOT download any PDFs.

Why we need a manifest
----------------------
Every later step (download, quality check, text extraction, chunking)
reads this CSV instead of asking Google Drive again. That means:
  - later steps are faster (no repeated API listing)
  - a crashed step can resume (we know exactly which files exist)
  - we have a permanent record of what is in the dataset

Before running
--------------
- check_drive_access.py must already work (credentials.json + token.json
  are in this folder). This script reuses the same login.

Run
---
    python build_manifest.py

Output
------
    manifest.csv   (one row per file, opens fine in Excel)
"""

import csv
import os
import re
import sys
from collections import Counter

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ------------------------------------------------------------------
# Settings (change these if needed)
# ------------------------------------------------------------------

# Read-only: this script can never modify or delete anything on Drive.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ID of the "LLM Reports" folder (from its URL .../folders/<ID>).
FOLDER_ID = "1ew-Pj4ry2vIzYpFMA3h3LiGZT-87YvnP"

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"
MANIFEST_FILE = "manifest.csv"

# Any file inside a folder with one of these names gets skip=True.
# "_wrong_pdf" holds bad / misfiled PDFs, so we don't want to ingest it.
SKIP_FOLDERS = {"_wrong_pdf"}

# How many times the Google client retries a failed API call
# (network hiccups, 5xx errors, rate limits) before giving up.
API_RETRIES = 5

# Google's MIME types for folders and shortcuts.
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

# Paper filenames look like:
#   2019_Bastola_Assessment_of_Soil_Erosion_Loss_..._W2994851782.pdf
#   ^year ^author ^title (underscores)             ^OpenAlex ID
# The regex below captures those four parts.
#
# Limitation: author and title are both separated by "_", so a
# two-part surname (e.g. "Burton_Page") will be split wrongly:
# author="Burton", title starts with "Page ...". Year and ID are
# always reliable; treat author/title as "best guess" metadata.
PAPER_NAME_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<author>[^_]+)_(?P<title>.+)_(?P<openalex_id>W\d+)\.pdf$",
    re.IGNORECASE,
)

# Columns written to manifest.csv, in this order.
CSV_COLUMNS = [
    "file_id",       # Google Drive ID - used later to download the file
    "path",          # full path inside the folder, e.g. /papers/01_Floods_GLOFs/x.pdf
    "name",          # just the filename
    "size_bytes",    # size in bytes (used to resume downloads + check totals)
    "md5",           # checksum from Drive - verifies downloads, finds duplicates
    "mime_type",     # should be application/pdf for everything here
    "modified_time", # last modified on Drive (to detect updated files later)
    "collection",    # "papers" or "reports" (first folder level)
    "category",      # topic (papers) or source organization (reports)
    "year",          # parsed from filename (papers only, if pattern matches)
    "author",        # parsed from filename - best guess, see note above
    "title",         # parsed from filename, underscores turned into spaces
    "openalex_id",   # the W... ID at the end of paper filenames
    "skip",          # True = don't ingest (e.g. inside _wrong_pdf/)
]


# ------------------------------------------------------------------
# Login (same logic as check_drive_access.py)
# ------------------------------------------------------------------

def authenticate():
    """Return valid Google credentials.

    1. Try the saved token.json.
    2. If expired, try to refresh it silently.
    3. If refresh fails (normal in Testing mode after ~7 days),
       delete token.json and open the browser to log in again.
    """
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
            sys.exit(f"{CREDENTIALS_FILE} not found. Run the setup first.")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)

    # Save (possibly refreshed) credentials for next time.
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    return creds


# ------------------------------------------------------------------
# Listing the Drive folder tree
# ------------------------------------------------------------------

def list_folder(service, folder_id):
    """Return every item DIRECTLY inside one folder (not recursive).

    Drive returns at most 1000 items per call, so we loop over pages
    using nextPageToken until there are no more pages.
    """
    items = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                # Only ask for the fields we actually use (faster, smaller).
                fields=(
                    "nextPageToken, files(id, name, mimeType, size, "
                    "md5Checksum, modifiedTime, shortcutDetails)"
                ),
                pageToken=page_token,
                pageSize=1000,
                # Needed in case any part lives in a Shared Drive.
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute(num_retries=API_RETRIES)  # auto-retry on transient errors
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def walk_tree(service, folder_id, path=""):
    """Yield every FILE under folder_id, going into all subfolders.

    Each yielded dict is the Drive metadata plus a "path" key, e.g.
    "/papers/01_Floods_GLOFs/2019_x.pdf".

    Uses a generator (yield) so we can show progress as we go.
    """
    for item in list_folder(service, folder_id):
        item_path = f"{path}/{item['name']}"
        mime = item["mimeType"]

        if mime == FOLDER_MIME:
            # A subfolder: print progress, then go inside it.
            print(f"  scanning {item_path}/")
            yield from walk_tree(service, item["id"], item_path)

        elif mime == SHORTCUT_MIME:
            # A shortcut points to another file/folder somewhere else.
            target = item.get("shortcutDetails", {})
            if target.get("targetMimeType") == FOLDER_MIME:
                print(f"  following shortcut {item_path}/")
                yield from walk_tree(service, target["targetId"], item_path)
            else:
                # Shortcut to a single file: record the TARGET's ID so the
                # download step fetches the real file. Size/md5 aren't
                # included in shortcut metadata, so they stay blank.
                item["id"] = target.get("targetId", item["id"])
                item["mimeType"] = target.get("targetMimeType", mime)
                item["path"] = item_path
                yield item

        else:
            # A normal file.
            item["path"] = item_path
            yield item


# ------------------------------------------------------------------
# Turning Drive metadata into a manifest row
# ------------------------------------------------------------------

def parse_paper_name(filename):
    """Extract year/author/title/openalex_id from a paper filename.

    Returns a dict with those keys; values are "" if the filename
    doesn't follow the Year_Author_Title_W<id>.pdf pattern.
    """
    match = PAPER_NAME_RE.match(filename)
    if not match:
        return {"year": "", "author": "", "title": "", "openalex_id": ""}
    return {
        "year": match["year"],
        "author": match["author"],
        "title": match["title"].replace("_", " "),
        "openalex_id": match["openalex_id"].upper(),
    }


def make_row(item):
    """Build one manifest.csv row (a dict) from a Drive file item."""
    # "/papers/01_Floods_GLOFs/x.pdf" -> ["papers", "01_Floods_GLOFs", "x.pdf"]
    parts = item["path"].strip("/").split("/")
    collection = parts[0] if len(parts) > 1 else ""
    category = parts[1] if len(parts) > 2 else ""

    # Only papers follow the Year_Author_Title_W<id> naming pattern.
    if collection == "papers":
        meta = parse_paper_name(item["name"])
    else:
        meta = {"year": "", "author": "", "title": "", "openalex_id": ""}

    # Skip the file if ANY folder in its path is in SKIP_FOLDERS.
    skip = any(p in SKIP_FOLDERS for p in parts[:-1])

    return {
        "file_id": item["id"],
        "path": item["path"],
        "name": item["name"],
        "size_bytes": item.get("size", ""),
        "md5": item.get("md5Checksum", ""),
        "mime_type": item["mimeType"],
        "modified_time": item.get("modifiedTime", ""),
        "collection": collection,
        "category": category,
        **meta,
        "skip": skip,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def human_size(num_bytes):
    """1234567 -> '1.2 MB'"""
    num_bytes = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def main():
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    # Quick check that the folder is reachable before a long scan.
    try:
        root = (
            service.files()
            .get(fileId=FOLDER_ID, fields="name", supportsAllDrives=True)
            .execute(num_retries=API_RETRIES)
        )
    except HttpError as e:
        sys.exit(f"Cannot open folder {FOLDER_ID}: {e}")

    print(f"Building manifest for: {root['name']}\n")

    # Collect all rows. We write to a temp file first and rename at the
    # end, so a crash halfway never leaves a half-written manifest.csv.
    rows = [make_row(item) for item in walk_tree(service, FOLDER_ID)]

    tmp_file = MANIFEST_FILE + ".tmp"
    # utf-8-sig = UTF-8 with a marker so Excel on Windows shows
    # non-English characters (e.g. Nepali names) correctly.
    with open(tmp_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_file, MANIFEST_FILE)

    # ---------------- Summary ----------------
    keep = [r for r in rows if not r["skip"]]
    skipped = [r for r in rows if r["skip"]]
    total_bytes = sum(int(r["size_bytes"] or 0) for r in keep)

    print(f"\nWrote {len(rows)} rows to {MANIFEST_FILE}")
    print(f"  to ingest: {len(keep)} files, {human_size(total_bytes)}")
    print(f"  skipped:   {len(skipped)} files (in {', '.join(SKIP_FOLDERS)})")

    # Files per collection (papers / reports).
    print("\nBy collection:")
    for name, count in Counter(r["collection"] for r in keep).most_common():
        print(f"  {name or '(root)':<12} {count:>6}")

    # How many paper filenames matched the naming pattern.
    papers = [r for r in keep if r["collection"] == "papers"]
    parsed = [r for r in papers if r["openalex_id"]]
    if papers:
        print(
            f"\nPaper filenames parsed: {len(parsed)} / {len(papers)} "
            f"({100 * len(parsed) / len(papers):.1f}%)"
        )

    # Non-PDF files (should be 0 - worth knowing if not).
    non_pdf = [r for r in keep if r["mime_type"] != "application/pdf"]
    if non_pdf:
        print(f"\nNon-PDF files: {len(non_pdf)} (check these in the CSV)")

    # Duplicates: same md5 = identical file content stored twice.
    md5_counts = Counter(r["md5"] for r in keep if r["md5"])
    dup_groups = {m: c for m, c in md5_counts.items() if c > 1}
    if dup_groups:
        extra = sum(c - 1 for c in dup_groups.values())
        print(
            f"\nDuplicates: {len(dup_groups)} files appear more than once "
            f"({extra} extra copies). Filter by 'md5' in the CSV to see them."
        )

    # Tiny files are often stubs/abstracts rather than full documents.
    tiny = [r for r in keep if r["size_bytes"] and int(r["size_bytes"]) < 20_000]
    if tiny:
        print(f"\nVery small files (<20 KB): {len(tiny)} - may be stubs, check later.")

    print("\nDone. Next step: download_drive.py (reads this manifest).")


if __name__ == "__main__":
    main()