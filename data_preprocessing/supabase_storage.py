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
import logging

import boto3
from botocore.client import Config
from dotenv import load_dotenv

# Load .env from backend root
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_PATH)

_ACCESS_KEY_ID     = os.getenv("Access_key_ID", "")
_SECRET_ACCESS_KEY = os.getenv("Secret_access_key", "")
SUPABASE_BUCKET    = os.getenv("SUPABASE_BUCKET", "storage")
_SUPABASE_LOCAL_FALLBACK = os.getenv("SUPABASE_LOCAL_FALLBACK", "1").strip().lower() not in {"0", "false", "no"}

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_STORAGE_ROOT = os.path.join(_BACKEND_ROOT, "storage")

logger = logging.getLogger("SupabaseStorage")

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


def _local_storage_path(path: str) -> str:
    """Resolve storage key to a safe local fallback path under backend/storage."""
    normalized = os.path.normpath(path).replace("\\", "/").lstrip("/")
    if normalized.startswith(".."):
        raise ValueError(f"Invalid storage path: {path}")
    return os.path.join(_LOCAL_STORAGE_ROOT, normalized)


def _write_local_fallback(path: str, content: bytes) -> str:
    local_path = _local_storage_path(path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(content)
    return local_path


def _read_local_fallback(path: str) -> bytes:
    local_path = _local_storage_path(path)
    with open(local_path, "rb") as f:
        return f.read()


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
    try:
        resp = _s3.get_object(Bucket=SUPABASE_BUCKET, Key=path)
        return resp["Body"].read()
    except Exception as exc:
        if not _SUPABASE_LOCAL_FALLBACK:
            raise

        local_path = _local_storage_path(path)
        if not os.path.exists(local_path):
            raise

        logger.warning(
            "Supabase download failed for '%s' (%s). Using local fallback: %s",
            path,
            exc,
            local_path,
        )
        return _read_local_fallback(path)


def upload_file(path: str, content: bytes, content_type: str = "application/octet-stream"):
    """
    Upload (upsert) bytes to Supabase storage.

    Args:
        path:         Storage path, e.g. "output/user1/session1/data.csv"
        content:      Raw bytes to store.
        content_type: MIME type of the content.
    """
    try:
        _s3.put_object(
            Bucket=SUPABASE_BUCKET,
            Key=path,
            Body=content,
            ContentType=content_type,
        )
    except Exception as exc:
        if not _SUPABASE_LOCAL_FALLBACK:
            raise

        local_path = _write_local_fallback(path, content)
        logger.warning(
            "Supabase upload failed for '%s' (%s). Wrote local fallback: %s",
            path,
            exc,
            local_path,
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
    try:
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
    except Exception as exc:
        if not _SUPABASE_LOCAL_FALLBACK:
            raise

        local_dir = _local_storage_path(folder_path)
        if not os.path.isdir(local_dir):
            return []

        filenames = []
        if recursive:
            for root, _, files in os.walk(local_dir):
                for name in files:
                    rel = os.path.relpath(os.path.join(root, name), local_dir)
                    filenames.append(rel.replace("\\", "/"))
        else:
            for name in os.listdir(local_dir):
                candidate = os.path.join(local_dir, name)
                if os.path.isfile(candidate):
                    filenames.append(name)

        logger.warning(
            "Supabase list_files failed for '%s' (%s). Using local fallback.",
            folder_path,
            exc,
        )
        return sorted(filenames)


def list_folders(folder_path: str) -> list:
    """
    List immediate subfolder names inside a Supabase storage folder.

    Args:
        folder_path: e.g. "input/user1"

    Returns:
        List of subfolder name strings (no trailing slash).
    """
    try:
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
    except Exception as exc:
        if not _SUPABASE_LOCAL_FALLBACK:
            raise

        local_dir = _local_storage_path(folder_path)
        if not os.path.isdir(local_dir):
            return []

        folders = [
            name
            for name in os.listdir(local_dir)
            if os.path.isdir(os.path.join(local_dir, name))
        ]
        logger.warning(
            "Supabase list_folders failed for '%s' (%s). Using local fallback.",
            folder_path,
            exc,
        )
        return sorted(folders)


def delete_file(path: str):
    """
    Delete a file from Supabase storage.

    Args:
        path: Storage path, e.g. "input/user1/session1/data.csv"
    """
    remote_exc = None
    try:
        _s3.delete_object(Bucket=SUPABASE_BUCKET, Key=path)
    except Exception as exc:
        remote_exc = exc
        if not _SUPABASE_LOCAL_FALLBACK:
            raise

    if _SUPABASE_LOCAL_FALLBACK:
        local_path = _local_storage_path(path)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

    if remote_exc is not None:
        logger.warning(
            "Supabase delete failed for '%s' (%s). Local fallback cleanup attempted.",
            path,
            remote_exc,
        )


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

