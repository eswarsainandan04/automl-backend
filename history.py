"""
History API

GET    /history/                       — list all sessions for the current user
GET    /history/{session_id}/detail    — input table, cleaned table, profiling per file
GET    /history/{session_id}/cleaned-csv/{filename} — download a cleaned CSV
DELETE /history/{session_id}           — delete every file in a session
"""

import io
import os
import sys
from datetime import datetime

import pandas as pd
import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from data_preprocessing.supabase_storage import (
    delete_file,
    download_file,
    download_json,
    list_files,
    list_folders,
)
from jwt_handler import ALGORITHM, SECRET_KEY

router = APIRouter(prefix="/history", tags=["History"])
security = HTTPBearer()


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _email_from_token(token: str) -> str:
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _user_id_by_email(email: str) -> str:
    conn = psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT,
        database=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return str(row[0])


def _get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    return _user_id_by_email(_email_from_token(credentials.credentials))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/")
def list_sessions(user_id: str = Depends(_get_current_user_id)):
    """Return all sessions for the current user, newest first."""
    try:
        session_ids = list_folders(f"input/{user_id}")
    except Exception:
        session_ids = []

    sessions = []
    for sid in sorted(session_ids, reverse=True):
        # parse the YYYYMMDDHHmmss timestamp used by the upload endpoint
        try:
            dt = datetime.strptime(sid, "%Y%m%d%H%M%S")
            label = dt.strftime("%B %d, %Y  %H:%M:%S")
        except ValueError:
            label = sid

        try:
            input_files = list_files(f"input/{user_id}/{sid}")
        except Exception:
            input_files = []

        sessions.append({
            "session_id":  sid,
            "datetime":    label,
            "input_files": input_files,
        })

    return sessions


@router.get("/{session_id}/detail")
def get_session_detail(
    session_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return input table, cleaned table, and profiling JSON for every file in a session."""
    try:
        input_files = list_files(f"input/{user_id}/{session_id}")
    except Exception:
        input_files = []

    datasets = []
    for fname in sorted(input_files):
        base = os.path.splitext(fname)[0]

        # Raw input table ──────────────────────────────────────────────────────
        input_columns, input_rows = [], []
        try:
            raw = download_file(f"input/{user_id}/{session_id}/{fname}")
            df_in = pd.read_csv(io.BytesIO(raw))
            input_columns = list(df_in.columns)
            input_rows = df_in.head(500).fillna("").astype(str).to_dict(orient="records")
        except Exception:
            pass

        # Cleaned output table ─────────────────────────────────────────────────
        cleaned_columns, cleaned_rows = [], []
        try:
            cb = download_file(f"output/{user_id}/{session_id}/{base}_cleaned.csv")
            df_clean = pd.read_csv(io.BytesIO(cb))
            cleaned_columns = list(df_clean.columns)
            cleaned_rows = df_clean.head(500).fillna("").astype(str).to_dict(orient="records")
        except Exception:
            pass

        # Profiling JSON ───────────────────────────────────────────────────────
        profiling = {}
        try:
            profiling = download_json(
                f"meta_data/{user_id}/{session_id}/{base}_profiling.json"
            )
        except Exception:
            pass

        datasets.append({
            "filename":        fname,
            "base":            base,
            "input_columns":   input_columns,
            "input_rows":      input_rows,
            "cleaned_columns": cleaned_columns,
            "cleaned_rows":    cleaned_rows,
            "profiling":       profiling,
        })

    try:
        meta_folders = list_folders(f"meta_data/{user_id}/{session_id}")
    except Exception:
        meta_folders = []

    has_analytics = "analytics" in meta_folders
    try:
        analytics_files = list_files(f"meta_data/{user_id}/{session_id}/analytics")
    except Exception:
        analytics_files = []

    return {
        "session_id": session_id,
        "datasets": datasets,
        "has_analytics": has_analytics,
        "analytics_files": sorted(analytics_files),
    }


@router.get("/{session_id}/cleaned-csv/{filename}")
def download_cleaned_csv(
    session_id: str,
    filename: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return the cleaned CSV bytes so the browser can save it."""
    # Sanitise filename to prevent path traversal
    safe_name = os.path.basename(filename)
    base = os.path.splitext(safe_name)[0]
    try:
        content = download_file(
            f"output/{user_id}/{session_id}/{base}_cleaned.csv"
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Cleaned CSV not found")
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{base}_cleaned.csv"'
        },
    )


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Delete every file in a session from input, meta_data, and output."""
    deleted = 0
    errors: list = []

    prefixes = [
        f"input/{user_id}/{session_id}",
        f"meta_data/{user_id}/{session_id}",
        f"output/{user_id}/{session_id}",
    ]

    for prefix in prefixes:
        try:
            files = list_files(prefix, recursive=True)
        except Exception:
            files = []
        for f in files:
            try:
                delete_file(f"{prefix}/{f}")
                deleted += 1
            except Exception as exc:
                errors.append(str(exc))

    return {"deleted": deleted, "errors": errors}
