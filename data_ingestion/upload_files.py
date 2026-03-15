"""
File Upload & Pipeline Trigger API

POST /upload/              - Upload CSV/XLSX files, trigger pipeline in background
GET  /upload/status/{sid}  - Poll pipeline step status

Overview endpoint lives in data_ingestion/overview.py.
Pipeline orchestration logic lives in pipeline.py.
"""

import os
import sys
import threading
from datetime import datetime
from typing import Dict, List

import psycopg2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt

# Ensure backend root on path so sibling packages are importable
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from data_preprocessing.supabase_storage import upload_file
from jwt_handler import ALGORITHM, SECRET_KEY
from pipeline import PIPELINE_STEPS, run_preprocessing

router = APIRouter(prefix="/upload", tags=["Upload"])
security = HTTPBearer()

# ─────────────────────────────────────────────────────────────────────────────
# In-memory pipeline status store  key = session_id
# ─────────────────────────────────────────────────────────────────────────────
_pipeline_status: Dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    email = _email_from_token(credentials.credentials)
    return _user_id_by_email(email)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/")
async def upload_files(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """Upload one or more CSV/XLSX files and kick off the preprocessing pipeline."""
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    uploaded = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: .csv, .xlsx, .xls",
            )
        content = await file.read()
        path = f"input/{user_id}/{session_id}/{file.filename}"
        content_type = (
            "text/csv"
            if ext == ".csv"
            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        upload_file(path, content, content_type)
        uploaded.append(file.filename)

    # Initialise pipeline status
    _pipeline_status[session_id] = {
        "user_id": user_id,
        "session_id": session_id,
        "status": "running",
        "files": uploaded,
        "steps": [{"name": s, "status": "pending", "error": None} for s in PIPELINE_STEPS],
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    # Fire-and-forget background thread — delegates all step logic to pipeline.py
    t = threading.Thread(
        target=run_preprocessing,
        args=(user_id, session_id, _pipeline_status[session_id]),
        daemon=True,
    )
    t.start()

    return {"session_id": session_id, "user_id": user_id, "files": uploaded}


@router.get("/status/{session_id}")
def get_pipeline_status(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return live pipeline status for a session."""
    entry = _pipeline_status.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Session not found")
    if entry["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return entry
