"""


Purpose: verify you can LIST and READ files in a Google Drive folder
via the API, without downloading the whole 95GB folder.

This does NOT download your files. It only:
  1. Lists what's in the folder (name, size, type)
  2. Adds up the total size so you know what you're dealing with
  3. Downloads ONE small file as a real end-to-end test

------------------------------------------------------------------
ONE-TIME SETUP (about 5 minutes)
------------------------------------------------------------------
1. Go to https://console.cloud.google.com/
2. Create a new project (any name, e.g. "nepal-env-rag")
3. Go to "APIs & Services" -> "Library" -> search "Google Drive API" -> Enable
4. Go to "APIs & Services" -> "Credentials" -> "Create Credentials"
   -> "OAuth client ID" -> Application type: "Desktop app" -> Create
5. Download the JSON. Rename it to credentials.json and put it in
   this project's root folder (same folder as this script).
6. Install the required packages:

   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

7. Run this script:

   python check_drive_access.py

   The first run opens a browser window asking you to log in and grant
   read-only access. After that, a token.json is saved so you won't
   need to log in again.

Note: the account you log in with must be an account that already has
access to the folder (owner, or it must be shared with that account).
------------------------------------------------------------------
"""

import io
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Read-only scope: we only ever need to list and read, never modify/delete.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Extracted from your shared link:
# https://drive.google.com/drive/folders/1ew-Pj4ry2vIzYpFMA3h3LiGZT-87YvnP
FOLDER_ID = "1ew-Pj4ry2vIzYpFMA3h3LiGZT-87YvnP"


def authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Follow the setup steps "
                    "in this script's docstring first."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def list_all_files(service, folder_id):
    """List every file in the folder, handling pagination and
    (one level of) subfolders."""
    files = []
    page_token = None

    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

    return files


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def download_small_test_file(service, file_id, file_name):
    """Download just ONE file fully, to prove read access actually works,
    not just metadata listing."""
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    out_path = f"_drive_test_{file_name}"
    with open(out_path, "wb") as f:
        f.write(buf.getvalue())
    return out_path


def main():
    creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    print(f"Listing files in folder: {FOLDER_ID}\n")
    files = list_all_files(service, FOLDER_ID)

    if not files:
        print(
            "No files found. Either the folder is empty, the folder ID is "
            "wrong, or the logged-in account doesn't have access to it."
        )
        return

    pdf_files = [f for f in files if f.get("mimeType") == "application/pdf"]
    other_files = [f for f in files if f.get("mimeType") != "application/pdf"]

    total_bytes = sum(int(f.get("size", 0)) for f in files if f.get("size"))

    print(f"Total items found: {len(files)}")
    print(f"  PDFs:   {len(pdf_files)}")
    print(f"  Other:  {len(other_files)} (folders / non-PDF files)")
    print(f"Total size (of items with size info): {human_size(total_bytes)}\n")

    print("First 10 PDFs found:")
    for f in pdf_files[:10]:
        size = human_size(int(f["size"])) if f.get("size") else "unknown size"
        print(f"  - {f['name']}  ({size})")

    if other_files:
        print("\nNon-PDF items (could be subfolders — this script only reads")
        print("one level deep; tell me if you have nested subfolders):")
        for f in other_files[:10]:
            print(f"  - {f['name']}  [{f.get('mimeType')}]")

    # Real end-to-end proof: actually download the smallest PDF.
    if pdf_files:
        smallest = min(
            (f for f in pdf_files if f.get("size")),
            key=lambda f: int(f["size"]),
            default=pdf_files[0],
        )
        print(f"\nDownloading smallest PDF as a live test: {smallest['name']}")
        path = download_small_test_file(service, smallest["id"], smallest["name"])
        print(f"Success. Saved test copy to: {path}")
        print("You can delete this test file — it just proves API access works.")

    print("\nResult: your setup CAN reach this Drive folder via the API,")
    print("list every file, and read/download files one at a time.")
    print("You do NOT need to download the whole 95GB up front.")


if __name__ == "__main__":
    main()