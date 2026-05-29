"""
History API

GET    /history/                                         — list all sessions for the current user
GET    /history/{session_id}/detail                      — input table, cleaned table, profiling per file
GET    /history/{session_id}/cleaned-csv/{filename}      — download a cleaned CSV
DELETE /history/{session_id}                             — delete every file in a session
GET    /history/{session_id}/model-info/{dataset_base}   — enriched model report (metrics, leaderboard, feature importance)
GET    /history/{session_id}/model-schema/{dataset_base} — input field schema for inline prediction
POST   /history/{session_id}/predict/{dataset_base}      — run prediction using the session's trained .pkl
"""

import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt
from pydantic import BaseModel

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
from model_info import build_model_info_response

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


# ── Subprocess helper (mirrors automl_router pattern) ─────────────────────────

def _resolve_model_python() -> Path:
    for env_name in ("MODEL_SELECTION_PYTHON", "FLAML_PYTHON"):
        configured = os.getenv(env_name)
        if not configured:
            continue
        p = Path(configured).expanduser()
        if p.exists() and p.is_file():
            return p
    current = Path(sys.executable).expanduser()
    if current.exists() and current.is_file():
        return current
    raise FileNotFoundError("Model Python not found. Set MODEL_SELECTION_PYTHON.")


def _run_model_testing_subprocess(function_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Run a model_testing function in the dedicated ML subprocess (same as automl_router)."""
    python_exe = _resolve_model_python()
    backend_dir = Path(_BACKEND_ROOT)
    sentinel = "__MODEL_TESTING_JSON__"

    code = (
        "import json,sys;"
        "import model_testing;"
        "fn=getattr(model_testing, sys.argv[1]);"
        "kwargs=json.loads(sys.argv[2]);"
        "res=fn(**kwargs);"
        f"print('{sentinel}'+json.dumps(res))"
    )

    proc = subprocess.run(
        [str(python_exe), "-c", code, function_name, json.dumps(kwargs)],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-20:])
        raise RuntimeError(
            f"Model testing subprocess failed (function={function_name}).\n"
            f"stderr:\n{stderr_tail}\nstdout:\n{stdout_tail}"
        )

    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(sentinel):
            return json.loads(line[len(sentinel):].strip())

    raise RuntimeError(
        f"Model testing subprocess completed but returned no structured output for {function_name}."
    )


# ── Pydantic body ──────────────────────────────────────────────────────────────

class HistoryPredictBody(BaseModel):
    row: Dict[str, Any]


# ── Original endpoints (unchanged) ────────────────────────────────────────────

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

    # ── Discover trained models for this session ───────────────────────────────
    trained_models: List[str] = []
    try:
        output_files = list_files(f"output/{user_id}/{session_id}")
        trained_models = [
            os.path.splitext(f)[0].replace("_model", "")
            for f in output_files
            if f.endswith("_model.pkl")
        ]
    except Exception:
        pass

    return {
        "session_id":     session_id,
        "datasets":       datasets,
        "has_analytics":  has_analytics,
        "analytics_files": sorted(analytics_files),
        "trained_models": trained_models,   # NEW — consumed by frontend Models tab
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


# ── NEW: Model endpoints for History Models tab ────────────────────────────────

@router.get("/{session_id}/model-info/{dataset_base}")
def get_history_model_info(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Return an enriched, frontend-ready model report for a history session.
    Downloads _model_report.json from Supabase and normalises it via
    build_model_info_response() — same logic as /automl/model-info.
    """
    report_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_model_report.json"
    try:
        report = download_json(report_path)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model report not found for '{dataset_base}' in session '{session_id}'. "
                   f"Model may not have been trained yet. ({exc})",
        )

    try:
        response = build_model_info_response(report)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process model report: {exc}",
        )

    return response


@router.get("/{session_id}/model-schema/{dataset_base}")
def get_history_model_schema(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Return the input field schema for testing a trained model from history.
    Runs get_model_testing_schema_supabase in the dedicated ML subprocess.
    """
    try:
        data = _run_model_testing_subprocess(
            function_name="get_model_testing_schema_supabase",
            kwargs={
                "user_id":      user_id,
                "session_id":   session_id,
                "dataset_base": dataset_base,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data


@router.post("/{session_id}/predict/{dataset_base}")
def history_predict(
    session_id: str,
    dataset_base: str,
    body: HistoryPredictBody,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Run a prediction against a trained model from a history session.
    Loads the session's .pkl via predict_from_session_model in the ML subprocess.
    """
    try:
        data = _run_model_testing_subprocess(
            function_name="predict_from_session_model",
            kwargs={
                "user_id":      user_id,
                "session_id":   session_id,
                "dataset_base": dataset_base,
                "row":          body.row,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data