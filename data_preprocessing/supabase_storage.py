"""
Supabase Storage Utility for AutoML Preprocessing System

Uses the Supabase S3-compatible API (boto3) so that all preprocessing
modules can read/write files without touching local disk.

Storage path convention:
    input/    {user_id}/{session_id}/{filename}
    output/   {user_id}/{session_id}/{filename}
    meta_data/{user_id}/{session_id}/{filename}
"""

import os
import json

import boto3
from botocore.client import Config
from dotenv import load_dotenv

# Load .env from backend root
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_PATH)

_ACCESS_KEY_ID     = os.getenv("Access_key_ID", "")
_SECRET_ACCESS_KEY = os.getenv("Secret_access_key", "")
SUPABASE_BUCKET    = os.getenv("SUPABASE_BUCKET", "storage")

# Derive Supabase project ref from POSTGRES_USER (format: postgres.<project_ref>)
_POSTGRES_USER = os.getenv("POSTGRES_USER", "")
_PROJECT_REF   = _POSTGRES_USER.split(".")[-1] if "." in _POSTGRES_USER else _POSTGRES_USER
_S3_ENDPOINT   = f"https://{_PROJECT_REF}.supabase.co/storage/v1/s3"

_s3 = boto3.client(
    "s3",
    endpoint_url=_S3_ENDPOINT,
    aws_access_key_id=_ACCESS_KEY_ID,
    aws_secret_access_key=_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
    region_name="ap-southeast-1",
)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def download_file(path: str) -> bytes:
    """
    Download a file from Supabase storage and return its raw bytes.

    Args:
        path: Storage path, e.g. "input/user1/session1/data.csv"

    Returns:
        Raw file bytes.
    """
    resp = _s3.get_object(Bucket=SUPABASE_BUCKET, Key=path)
    return resp["Body"].read()


def upload_file(path: str, content: bytes, content_type: str = "application/octet-stream"):
    """
    Upload (upsert) bytes to Supabase storage.

    Args:
        path:         Storage path, e.g. "output/user1/session1/data.csv"
        content:      Raw bytes to store.
        content_type: MIME type of the content.
    """
    _s3.put_object(
        Bucket=SUPABASE_BUCKET,
        Key=path,
        Body=content,
        ContentType=content_type,
    )


def list_files(folder_path: str, recursive: bool = False) -> list:
    """
    List file names inside a Supabase storage folder.

    Args:
        folder_path: e.g. "input/user1/session1"
        recursive:  when True, include files in nested subfolders

    Returns:
        List of file name strings relative to folder_path.
        If recursive=False: only immediate children.
        If recursive=True: nested file paths are included.
    """
    prefix = folder_path.rstrip("/") + "/"
    paginator = _s3.get_paginator("list_objects_v2")
    filenames = []

    paginate_kwargs = {"Bucket": SUPABASE_BUCKET, "Prefix": prefix}
    if not recursive:
        paginate_kwargs["Delimiter"] = "/"

    for page in paginator.paginate(**paginate_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix):]  # strip folder prefix
            if filename and not filename.endswith("/"):
                filenames.append(filename)
    return filenames


def list_folders(folder_path: str) -> list:
    """
    List immediate subfolder names inside a Supabase storage folder.

    Args:
        folder_path: e.g. "input/user1"

    Returns:
        List of subfolder name strings (no trailing slash).
    """
    prefix = folder_path.rstrip("/") + "/"
    paginator = _s3.get_paginator("list_objects_v2")
    folders = []
    for page in paginator.paginate(Bucket=SUPABASE_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            pref = cp.get("Prefix", "")
            name = pref[len(prefix):].rstrip("/")
            if name:
                folders.append(name)
    return folders


def delete_file(path: str):
    """
    Delete a file from Supabase storage.

    Args:
        path: Storage path, e.g. "input/user1/session1/data.csv"
    """
    _s3.delete_object(Bucket=SUPABASE_BUCKET, Key=path)


def download_json(path: str) -> dict:
    """
    Download and parse a JSON file from Supabase storage.

    Args:
        path: Storage path to the JSON file.

    Returns:
        Parsed Python dict.
    """
    return json.loads(download_file(path).decode("utf-8"))


def upload_json(path: str, data: dict):
    """
    Serialize a dict to JSON and upload it to Supabase storage.

    Args:
        path: Destination storage path.
        data: Python dict to serialize.
    """
    content = json.dumps(data, indent=4, ensure_ascii=False).encode("utf-8")
    upload_file(path, content, content_type="application/json")

