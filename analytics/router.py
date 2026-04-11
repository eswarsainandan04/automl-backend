"""
FastAPI Analytics Router
========================
GET /analytics/run/{session_id}
    1. Loads every cleaned CSV + profiling JSON for the session.
    2. When 2+ datasets share column names (foreign keys), asks the LLM to
       propose a join plan, then executes it in Python.
    3. Runs the chart-planning pipeline on every individual dataset AND every
       joined dataset.
    4. Returns Chart.js-ready JSON — no matplotlib, no PNG files.
"""

import io
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt
from pydantic import BaseModel

_ANALYTICS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT  = os.path.dirname(_ANALYTICS_DIR)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
if _ANALYTICS_DIR not in sys.path:
    sys.path.insert(0, _ANALYTICS_DIR)

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from data_preprocessing.supabase_storage import download_file, download_json, list_files, upload_json
from data_preprocessing.structural_type_detector import StructuralTypeDetector
from jwt_handler import ALGORITHM, SECRET_KEY
from decision_maker import (
    extract_column_data,
    ask_llm_for_chart_plan,
    parse_llm_response,
    fill_chart_data,
    ensure_default_charts,
    ask_llm_for_join_plan,
    execute_join_plan,
)

router   = APIRouter(prefix="/analytics", tags=["Analytics"])
security = HTTPBearer()


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _email_from_token(token: str) -> str:
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _user_id_by_email_db(email: str) -> str:
    """Fallback DB lookup — only used for tokens that predate user_id embedding."""
    try:
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}")


def _get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Extract user_id directly from the JWT payload (no DB round-trip required)."""
    try:
        payload = jose_jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        user_id = payload.get("user_id")
        if user_id:
            return str(user_id)
        # Fallback for tokens issued before user_id was embedded in JWT
        return _user_id_by_email_db(email)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ── Chart post-processing helpers ─────────────────────────────────────────────

def _bucket_histogram(values: list, bins: int = 30) -> dict:
    """Convert a flat list of raw values into labelled histogram bins."""
    if not values:
        return {"x_values": [], "y_values": []}
    numeric = []
    for v in values:
        try:
            numeric.append(float(v))
        except (TypeError, ValueError):
            pass
    if not numeric:
        return {"x_values": [], "y_values": []}
    arr           = np.array(numeric, dtype=float)
    counts, edges = np.histogram(arr, bins=bins)
    labels        = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return {"x_values": labels, "y_values": counts.tolist()}


def _compute_box_stats(groups: dict) -> dict:
    """Return min / q1 / median / q3 / max per group for frontend box-plot."""
    stats = {}
    for label, values in groups.items():
        if not values:
            continue
        numeric = []
        for v in values:
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                pass
        if not numeric:
            continue
        arr = np.array(numeric, dtype=float)
        stats[label] = {
            "min":    round(float(np.min(arr)),            4),
            "q1":     round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.median(arr)),         4),
            "q3":     round(float(np.percentile(arr, 75)), 4),
            "max":    round(float(np.max(arr)),            4),
        }
    return stats


def _run_chart_pipeline(df: pd.DataFrame, profiling: dict, client, model_name: str) -> list:
    """
    Run the full chart-planning pipeline on one DataFrame.
    Returns a list of Chart.js-ready chart dicts.
    """
    column_data = extract_column_data(df, profiling)
    if not column_data:
        return []

    try:
        raw  = ask_llm_for_chart_plan(column_data, profiling, client, model_name)
        plan = parse_llm_response(raw)
        llm_charts = plan.get("charts", [])
    except Exception:
        llm_charts = []

    extra      = ensure_default_charts(llm_charts, df, column_data)
    all_charts = llm_charts + extra

    filled = []
    for spec in all_charts:
        data = fill_chart_data(spec, df)
        if data is None:
            continue

        chart_type = data.get("type", "")

        if chart_type == "histogram":
            bucketed         = _bucket_histogram(data.get("values", []))
            data["x_values"] = bucketed["x_values"]
            data["y_values"] = bucketed["y_values"]
            data.pop("values", None)

        box_stats = None
        if chart_type == "box":
            box_stats = _compute_box_stats(data.get("groups", {}))

        filled.append({
            "type":        chart_type,
            "title":       data.get("title",       ""),
            "description": data.get("description", ""),
            "x_column":    data.get("x_column",    ""),
            "y_column":    data.get("y_column",    ""),
            "x_values":    data.get("x_values",    []),
            "y_values":    data.get("y_values",    []),
            "box_stats":   box_stats,
        })

    return filled


_struct_detector = StructuralTypeDetector()


def _refresh_structural_types(profiling: dict) -> dict:
    """
    Re-run StructuralTypeDetector on every column in a profiling dict so that
    any stale `structural_type` values stored in Supabase are overwritten with
    the latest detection logic (e.g. sequential identifier detection).
    Mutates and returns the same dict.
    """
    total_rows = profiling.get("number_of_rows", 0)
    for col in profiling.get("column_wise_summary", []):
        col["structural_type"] = _struct_detector.detect(col, total_rows)
    return profiling


def _get_identifier_columns(profiling: dict) -> list:
    """Return column names whose structural_type is 'identifier'."""
    return [
        col["column_name"]
        for col in profiling.get("column_wise_summary", [])
        if col.get("structural_type") == "identifier"
    ]


def _make_joined_profiling(df: pd.DataFrame, result_name: str) -> dict:
    """
    Build a minimal profiling dict for a joined DataFrame so that
    extract_column_data() and ask_llm_for_chart_plan() can work with it.
    """
    col_summaries = []
    for col in df.columns:
        unique_count = int(df[col].nunique())
        n_rows       = len(df)
        dtype        = str(df[col].dtype)

        if pd.api.types.is_numeric_dtype(df[col]):
            stype = "numeric"
        elif unique_count <= max(20, n_rows * 0.05):
            stype = "categorical"
        else:
            stype = "text"

        col_summaries.append({
            "column_name":    col,
            "structural_type": stype,
            "semantic_type":  "",
            "unique_count":   unique_count,
        })

    return {
        "file_name":        result_name,
        "number_of_rows":   len(df),
        "number_of_columns": len(df.columns),
        "column_wise_summary": col_summaries,
        "missing_values_summary":  {},
        "row_processing_summary":  {},
        "transformation_summary":  {},
        "dataset_level_flags":     {},
    }


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.get("/run/{session_id}")
def run_analytics(
    session_id: str,
    user_id:    str = Depends(_get_current_user_id),
):
    """
    Full analytics pipeline for a session:
      - Runs chart pipeline on every individual cleaned CSV.
      - If 2+ datasets share columns (foreign keys), asks LLM for a join plan,
        executes the joins, and also runs the chart pipeline on each joined table.
    Returns Chart.js-ready JSON with individual + joined dataset sections.
    """
    from dotenv import load_dotenv
    from google import genai

    backend_env = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    analytics_env = os.path.join(_ANALYTICS_DIR, ".env")
    load_dotenv(backend_env)
    load_dotenv(analytics_env)

    api_key    = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server")

    client = genai.Client(api_key=api_key)

    # ── 1. Discover + load all datasets ──────────────────────────────────────
    try:
        meta_files = list_files(f"meta_data/{user_id}/{session_id}")
    except Exception:
        meta_files = []

    profiling_files = sorted(f for f in meta_files if f.endswith("_profiling.json"))
    if not profiling_files:
        raise HTTPException(status_code=404, detail="No profiling data found for this session")

    raw_datasets = []        # {"filename", "profiling", "df", "columns"}
    named_dfs    = {}        # {filename: df}   — seed for join execution

    for pf in profiling_files:
        base = pf.replace("_profiling.json", "")
        try:
            profiling = _refresh_structural_types(
                download_json(f"meta_data/{user_id}/{session_id}/{pf}")
            )
        except Exception:
            continue
        try:
            content = download_file(f"output/{user_id}/{session_id}/{base}_cleaned.csv")
            df      = pd.read_csv(io.BytesIO(content))
        except Exception:
            continue

        raw_datasets.append({
            "filename": base,
            "profiling": profiling,
            "df":        df,
            "columns":   list(df.columns),
        })
        named_dfs[base] = df

    if not raw_datasets:
        raise HTTPException(status_code=404, detail="No cleaned datasets found for this session")

    # ── 2. Detect foreign-key columns (shared across 2+ datasets) ────────────
    from collections import Counter
    col_count   = Counter(col for d in raw_datasets for col in d["columns"])
    foreign_keys = {col for col, cnt in col_count.items() if cnt >= 2}

    # ── 3. LLM join plan (only when ≥2 datasets share columns) ───────────────
    join_results = []   # [{"result_name", "description", "df"}]

    if len(raw_datasets) >= 2 and foreign_keys:
        # Feed the LLM only datasets that have at least one shared column
        datasets_with_fk = [
            d for d in raw_datasets
            if any(c in foreign_keys for c in d["columns"])
        ]
        if len(datasets_with_fk) >= 2:
            try:
                raw_join_resp = ask_llm_for_join_plan(datasets_with_fk, client, model_name)
                join_plan     = parse_llm_response(raw_join_resp)
                join_results  = execute_join_plan(join_plan, named_dfs)
            except Exception:
                join_results = []

    # ── 4. Run chart pipeline on individual datasets ──────────────────────────
    individual_results = []
    for d in raw_datasets:
        charts = _run_chart_pipeline(d["df"], d["profiling"], client, model_name)
        individual_results.append({
            "filename":     d["filename"],
            "is_joined":    False,
            "description":  "",
            "primary_keys": _get_identifier_columns(d["profiling"]),
            "charts":       charts,
        })

    # ── 5. Run chart pipeline on joined datasets ──────────────────────────────
    joined_results = []
    for jr in join_results:
        joined_df   = jr["df"]
        profiling_j = _make_joined_profiling(joined_df, jr["result_name"])
        charts      = _run_chart_pipeline(joined_df, profiling_j, client, model_name)
        joined_results.append({
            "filename":     jr["result_name"],
            "is_joined":    True,
            "description":  jr["description"],
            "primary_keys": [],
            "charts":       charts,
        })

    return {
        "session_id":   session_id,
        "foreign_keys": sorted(foreign_keys),
        "datasets":     individual_results + joined_results,
    }


# ── Generate & save dashboard JSON endpoint ───────────────────────────────────

@router.post("/generate-dashboards/{session_id}")
def generate_dashboards(
    session_id: str,
    user_id:    str = Depends(_get_current_user_id),
):
    """
    Run the full analytics pipeline, persist each dataset's charts as a JSON file
    in Supabase storage at:
        meta_data/{user_id}/{session_id}/analytics/{filename}_analytics.json
    and return Chart.js-ready JSON (same structure as GET /run).
    """
    from dotenv import load_dotenv
    from google import genai

    load_dotenv(os.path.join(_ANALYTICS_DIR, ".env"))

    api_key    = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server")

    client = genai.Client(api_key=api_key)

    # ── 1. Discover + load all datasets ──────────────────────────────────────
    try:
        meta_files = list_files(f"meta_data/{user_id}/{session_id}")
    except Exception:
        meta_files = []

    profiling_files = sorted(f for f in meta_files if f.endswith("_profiling.json"))
    if not profiling_files:
        raise HTTPException(status_code=404, detail="No profiling data found for this session")

    raw_datasets = []
    named_dfs    = {}

    for pf in profiling_files:
        base = pf.replace("_profiling.json", "")
        try:
            profiling = _refresh_structural_types(
                download_json(f"meta_data/{user_id}/{session_id}/{pf}")
            )
        except Exception:
            continue
        try:
            content = download_file(f"output/{user_id}/{session_id}/{base}_cleaned.csv")
            df      = pd.read_csv(io.BytesIO(content))
        except Exception:
            continue

        raw_datasets.append({
            "filename": base,
            "profiling": profiling,
            "df":        df,
            "columns":   list(df.columns),
        })
        named_dfs[base] = df

    if not raw_datasets:
        raise HTTPException(status_code=404, detail="No cleaned datasets found for this session")

    # ── 2. Detect foreign-key columns ─────────────────────────────────────────
    from collections import Counter
    col_count    = Counter(col for d in raw_datasets for col in d["columns"])
    foreign_keys = {col for col, cnt in col_count.items() if cnt >= 2}

    # ── 3. LLM join plan ──────────────────────────────────────────────────────
    join_results = []
    if len(raw_datasets) >= 2 and foreign_keys:
        datasets_with_fk = [
            d for d in raw_datasets
            if any(c in foreign_keys for c in d["columns"])
        ]
        if len(datasets_with_fk) >= 2:
            try:
                raw_join_resp = ask_llm_for_join_plan(datasets_with_fk, client, model_name)
                join_plan     = parse_llm_response(raw_join_resp)
                join_results  = execute_join_plan(join_plan, named_dfs)
            except Exception:
                join_results = []

    # ── 4. Run chart pipeline on each dataset, save JSON, collect results ──────
    individual_results = []
    joined_results_out = []

    def _save_dashboard(filename: str, charts: list, is_joined: bool, description: str = ""):
        """Persist a dashboard JSON to meta_data in Supabase (best-effort, silent)."""
        storage_path = (
            f"meta_data/{user_id}/{session_id}/analytics/{filename}_analytics.json"
        )
        payload = {
            "session_id":  session_id,
            "user_id":     user_id,
            "filename":    filename,
            "is_joined":   is_joined,
            "description": description,
            "charts":      charts,
        }
        try:
            upload_json(storage_path, payload)
        except Exception:
            pass

    for d in raw_datasets:
        charts = _run_chart_pipeline(d["df"], d["profiling"], client, model_name)
        _save_dashboard(d["filename"], charts, is_joined=False)
        individual_results.append({
            "filename":     d["filename"],
            "is_joined":    False,
            "description":  "",
            "primary_keys": _get_identifier_columns(d["profiling"]),
            "charts":       charts,
        })

    for jr in join_results:
        joined_df   = jr["df"]
        profiling_j = _make_joined_profiling(joined_df, jr["result_name"])
        charts      = _run_chart_pipeline(joined_df, profiling_j, client, model_name)
        _save_dashboard(jr["result_name"], charts, is_joined=True, description=jr["description"])
        joined_results_out.append({
            "filename":     jr["result_name"],
            "is_joined":    True,
            "description":  jr["description"],
            "primary_keys": [],
            "charts":       charts,
        })

    return {
        "session_id":   session_id,
        "foreign_keys": sorted(foreign_keys),
        "datasets":     individual_results + joined_results_out,
    }


# ── Load saved analytics endpoint ────────────────────────────────────────────

@router.get("/status/{session_id}")
def analytics_status(
    session_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return whether saved analytics JSON files already exist for this session."""
    analytics_prefix = f"meta_data/{user_id}/{session_id}/analytics"
    try:
        files = list_files(analytics_prefix)
    except Exception:
        files = []

    analytics_files = sorted(f for f in files if f.endswith("_analytics.json"))
    return {
        "session_id": session_id,
        "generated": bool(analytics_files),
        "file_count": len(analytics_files),
    }

@router.get("/load/{session_id}")
def load_analytics(
    session_id: str,
    user_id:    str = Depends(_get_current_user_id),
):
    """
    Load previously generated analytics from saved JSON files at
    meta_data/{user_id}/{session_id}/analytics/*.
    Returns 404 if no analytics have been generated yet.
    """
    analytics_prefix = f"meta_data/{user_id}/{session_id}/analytics"
    try:
        files = list_files(analytics_prefix)
    except Exception:
        files = []

    analytics_files = sorted(f for f in files if f.endswith("_analytics.json"))
    if not analytics_files:
        raise HTTPException(status_code=404, detail="No saved analytics found for this session")

    datasets = []
    for af in analytics_files:
        try:
            payload = download_json(f"{analytics_prefix}/{af}")
        except Exception:
            continue

        filename     = payload.get("filename", af.replace("_analytics.json", ""))
        primary_keys = payload.get("primary_keys", [])

        # If primary_keys weren't stored in the analytics JSON, re-derive from profiling
        if not primary_keys and not payload.get("is_joined", False):
            try:
                profiling = _refresh_structural_types(
                    download_json(f"meta_data/{user_id}/{session_id}/{filename}_profiling.json")
                )
                primary_keys = _get_identifier_columns(profiling)
            except Exception:
                pass

        datasets.append({
            "filename":     filename,
            "is_joined":    payload.get("is_joined", False),
            "description":  payload.get("description", ""),
            "primary_keys": primary_keys,
            "charts":       payload.get("charts", []),
        })

    if not datasets:
        raise HTTPException(status_code=404, detail="No valid analytics files found")

    # Detect foreign keys from profiling column overlap
    from collections import Counter
    col_count = Counter()
    try:
        meta_files = list_files(f"meta_data/{user_id}/{session_id}")
        for pf in [f for f in meta_files if f.endswith("_profiling.json")]:
            try:
                prof = download_json(f"meta_data/{user_id}/{session_id}/{pf}")
                for c in prof.get("column_wise_summary", []):
                    col_count[c["column_name"]] += 1
            except Exception:
                pass
    except Exception:
        pass
    foreign_keys = sorted({col for col, cnt in col_count.items() if cnt >= 2})

    return {
        "session_id":   session_id,
        "foreign_keys": foreign_keys,
        "datasets":     datasets,
    }


# ── Save (patch) a single chart in a saved analytics JSON ────────────────────

class ChartSaveBody(BaseModel):
    filename:  str
    chart_idx: int
    chart:     dict


@router.patch("/save-chart/{session_id}")
def save_chart(
    session_id: str,
    body:       ChartSaveBody,
    user_id:    str = Depends(_get_current_user_id),
):
    """
    Replace one chart in a saved analytics JSON file.
    The frontend sends the full updated chart dict; we overwrite that index.
    """
    storage_path = (
        f"meta_data/{user_id}/{session_id}/analytics/{body.filename}_analytics.json"
    )
    try:
        payload = download_json(storage_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Analytics file not found")

    charts = payload.get("charts", [])
    if body.chart_idx < 0 or body.chart_idx >= len(charts):
        raise HTTPException(status_code=400, detail="Invalid chart index")

    charts[body.chart_idx] = body.chart
    payload["charts"] = charts

    try:
        upload_json(storage_path, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save: {exc}")

    return {"status": "saved", "chart_idx": body.chart_idx}

