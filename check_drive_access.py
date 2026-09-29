"""
check_drive_access.py

Purpose: verify you can LIST and READ files in a shared Google Drive folder
via the API, without downloading the whole ~95GB folder.

This does NOT download your files. It only:
  1. Walks the folder and ALL its subfolders (e.g. papers/, reports/)
  2. Reports file counts and total size, per subfolder
  3. Downloads ONE small PDF as a real end-to-end test

Works with folders in "Shared with me" -- you do NOT need to copy or add
the folder to your own Drive. The logged-in account just needs view access.

------------------------------------------------------------------
ONE-TIME SETUP
------------------------------------------------------------------
1. https://console.cloud.google.com/ -> select your project
2. APIs & Services -> Library -> "Google Drive API" -> Enable
3. Google Auth Platform -> Audience (a.k.a. OAuth consent screen):
     - User type: External
     - Publishing status: Testing
     - Test users: add the EXACT Google account you will log in with
       (the one the folder was shared with)
4. Credentials -> Create Credentials -> OAuth client ID -> Desktop app
   Download the JSON, rename to credentials.json, put it next to this script.
5. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
6. python check_drive_access.py

On first run a browser opens. You'll see "Google hasn't verified this app":
click Advanced -> Go to <app name> (unsafe) -> allow read-only access.
A token.json is then saved. In Testing mode tokens expire after ~7 days;
this script detects that and re-prompts login automatically.
------------------------------------------------------------------
"""

import os
import re
import sys
from collections import defaultdict

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Read-only scope: we only ever list and read, never modify or delete.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ID of the "LLM Reports" folder, from its URL: .../drive/folders/<FOLDER_ID>
FOLDER_ID = "1ew-Pj4ry2vIzYpFMA3h3LiGZT-87YvnP"

TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
PDF_MIME = "application/pdf"


# ---------------------------------------------------------------- auth

def authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            # Typical in Testing mode after ~7 days: token revoked/expired.
            print("Saved login expired. Logging in again...\n")
            os.remove(TOKEN_FILE)
            creds = None

    if not creds or not creds.valid:
        if not os.path.exists(CREDENTIALS_FILE):
            sys.exit(
                f"{CREDENTIALS_FILE} not found. Follow the setup steps "
                "at the top of this script first."
            )
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    return creds


# ---------------------------------------------------------------- listing

def get_folder_info(service, folder_id):
    """Fetch the root folder's metadata to confirm access and show its name."""
    return (
        service.files()
        .get(
            fileId=folder_id,
            fields="id, name, mimeType, owners(emailAddress)",
            supportsAllDrives=True,
        )
        .execute()
    )


def list_all_files(service, folder_id, path=""):
    """Return every non-folder file under folder_id, recursing into all
    subfolders. Each file dict gets a 'path' like '/papers/foo.pdf'."""
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size, "
                "shortcutDetails)",
                pageToken=page_token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        for f in response.get("files", []):
            f["path"] = f"{path}/{f['name']}"
            mime = f["mimeType"]

            if mime == FOLDER_MIME:
                print(f"  scanning {f['path']}/ ...")
                files.extend(list_all_files(service, f["id"], f["path"]))
            elif mime == SHORTCUT_MIME:
                # Follow shortcuts that point to folders; skip others for now.
                target = f.get("shortcutDetails", {})
                if target.get("targetMimeType") == FOLDER_MIME:
                    print(f"  following shortcut {f['path']}/ ...")
                    files.extend(
                        list_all_files(service, target["targetId"], f["path"])
                    )
            else:
                files.append(f)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


# ---------------------------------------------------------------- helpers

def human_size(num_bytes):
    num_bytes = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def top_level_folder(path):
    """'/papers/sub/foo.pdf' -> 'papers';  '/foo.pdf' -> '(root)'"""
    parts = path.strip("/").split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def download_file(service, file_id, out_path):
    """Stream one file to disk in chunks (doesn't hold it all in memory)."""
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return out_path


# ---------------------------------------------------------------- main

def main():
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    # 1. Confirm we can see the root folder at all.
    try:
        root = get_folder_info(service, FOLDER_ID)
    except HttpError as e:
        if e.resp.status == 404:
            sys.exit(
                "Folder not found (404). Either FOLDER_ID is wrong, or the "
                "account you logged in with doesn't have access to it.\n"
                f"Delete {TOKEN_FILE} and log in with the account the folder "
                "was shared with."
            )
        raise

    owners = ", ".join(o["emailAddress"] for o in root.get("owners", [])) or "?"
    print(f"Folder: {root['name']}  (owner: {owners})\n")

    # 2. Walk everything.
    print("Scanning folder tree...")
    files = list_all_files(service, FOLDER_ID)
    print()

    if not files:
        print(
            "No files found. The folder and its subfolders appear empty to "
            "this account."
        )
        return

    pdfs = [f for f in files if f["mimeType"] == PDF_MIME]
    google_native = [
        f for f in files if f["mimeType"].startswith("application/vnd.google-apps")
    ]
    other = [f for f in files if f not in pdfs and f not in google_native]
    total_bytes = sum(int(f.get("size", 0)) for f in files)

    print(f"Total files: {len(files)}   Total size: {human_size(total_bytes)}")
    print(f"  PDFs:               {len(pdfs)}")
    print(f"  Google Docs/Sheets: {len(google_native)}  (need export, not download)")
    print(f"  Other files:        {len(other)}\n")

    # 3. Breakdown by top-level subfolder (papers/, reports/, ...).
    by_folder = defaultdict(lambda: {"count": 0, "pdfs": 0, "bytes": 0})
    for f in files:
        b = by_folder[top_level_folder(f["path"])]
        b["count"] += 1
        b["bytes"] += int(f.get("size", 0))
        if f["mimeType"] == PDF_MIME:
            b["pdfs"] += 1

    print("By subfolder:")
    for name, b in sorted(by_folder.items()):
        print(
            f"  {name:<20} {b['count']:>6} files  {b['pdfs']:>6} PDFs  "
            f"{human_size(b['bytes']):>10}"
        )
    print()

    print("Sample PDFs:")
    for f in pdfs[:10]:
        size = human_size(f["size"]) if f.get("size") else "unknown size"
        print(f"  - {f['path']}  ({size})")

    if other:
        print("\nSample non-PDF files:")
        for f in other[:10]:
            print(f"  - {f['path']}  [{f['mimeType']}]")

    # 4. Real end-to-end proof: download the smallest PDF.
    sized_pdfs = [f for f in pdfs if f.get("size")]
    if not sized_pdfs:
        print("\nNo PDFs with size info to test-download.")
        return

    smallest = min(sized_pdfs, key=lambda f: int(f["size"]))
    out_path = f"_drive_test_{safe_filename(smallest['name'])}"
    print(
        f"\nTest-downloading smallest PDF: {smallest['path']} "
        f"({human_size(smallest['size'])})"
    )

    try:
        download_file(service, smallest["id"], out_path)
    except HttpError as e:
        reason = str(e)
        if "cannotDownloadFile" in reason or e.resp.status == 403:
            print(
                "Download blocked (403). Listing works, but the owner may have "
                "disabled downloads for viewers, or the file hit Google's "
                "download quota. Ask the owner to allow downloads, or retry later."
            )
        else:
            print(f"Download failed: {e}")
        return

    print(f"Success. Saved test copy to: {out_path}  (safe to delete)")
    print(
        "\nResult: this account can list the whole folder tree and download "
        "files one at a time. No need to download all 95GB up front."
    )


if __name__ == "__main__":
    main()