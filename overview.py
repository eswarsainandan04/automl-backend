"""
Dataset Overview API

GET /upload/overview/{session_id}
    Reads every *_profiling.json from meta_data/{user_id}/{session_id}/,
    computes aggregate totals across all files, detects primary / foreign keys,
    and returns a 50-row cleaned-CSV preview per dataset.
"""

import io
import os
import re
import sys

import numpy as np
import pandas as pd
import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt

_BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from data_preprocessing.structural_type_detector import ConstraintGenerator
from data_preprocessing.supabase_storage import download_file, download_json, list_files
from jwt_handler import ALGORITHM, SECRET_KEY

router = APIRouter(prefix="/upload", tags=["Overview"])
security = HTTPBearer()

# ── Column-name pattern that qualifies a fully-unique column as a primary key ──
_PK_NAME_RE = re.compile(
    r"(^id$|_id$|^no$|_no$|^num$|_num$|^number$|_number$"
    r"|^key$|_key$|^code$|_code$|^pk$|_pk$|^idx$|_idx$)",
    re.IGNORECASE,
)


# ─── Auth helpers ──────────────────────────────────────────────────────────────

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


# ─── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/overview/{session_id}")
def get_overview(
    session_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return comprehensive overview built from _profiling.json files.

    Reads every *_profiling.json from meta_data/{user_id}/{session_id}/,
    computes aggregate totals, detects primary/foreign keys, and returns
    a 50-row CSV preview for each dataset.
    """
    # ── list profiling JSONs ───────────────────────────────────────────────────
    try:
        meta_files = list_files(f"meta_data/{user_id}/{session_id}")
    except Exception:
        meta_files = []

    profiling_files = sorted(f for f in meta_files if f.endswith("_profiling.json"))

    # ── aggregation totals (across all files) ─────────────────────────────────
    agg = {
        "file_count": 0,
        "total_rows_before": 0,
        "total_rows_after": 0,
        "total_columns": 0,
        "total_missing_before": 0,
        "total_missing_after": 0,
        "total_duplicates_removed": 0,
        "total_columns_transformed": 0,
        "total_columns_dropped": 0,
        "total_rows_dropped": 0,
        "all_primary_keys": [],   # [{"file": ..., "column": ...}]
        "all_foreign_keys": [],   # [{"file": ..., "column": ...}]
    }

    # ── First pass: collect raw data per file ─────────────────────────────────
    raw = []
    for pf in profiling_files:
        base = pf.replace("_profiling.json", "")
        try:
            profiling = download_json(f"meta_data/{user_id}/{session_id}/{pf}")
        except Exception:
            continue

        n_rows = profiling.get("number_of_rows", 0)
        col_summary = profiling.get("column_wise_summary", [])

        # Primary keys — BOTH conditions must hold:
        #   1. 100 % unique  (unique_count == number_of_rows)
        #   2. Column name matches an identifier pattern (_id, _no, _number, …)
        primary_keys = [
            c["column_name"]
            for c in col_summary
            if c.get("unique_count") == n_rows
            and n_rows > 0
            and _PK_NAME_RE.search(c["column_name"])
        ]

        # CSV preview from cleaned output
        preview, columns = [], []
        try:
            content = download_file(f"output/{user_id}/{session_id}/{base}_cleaned.csv")
            df = pd.read_csv(io.BytesIO(content))
            columns = list(df.columns)
            preview = df.head(50).fillna("").astype(str).to_dict(orient="records")
        except Exception:
            pass

        raw.append({
            "base":         base,
            "profiling":    profiling,
            "primary_keys": primary_keys,
            "columns":      columns,
            "preview":      preview,
        })

    # ── Determine true foreign keys: columns present in 2+ datasets ───────────
    from collections import Counter
    col_file_count = Counter()
    for r in raw:
        for col in set(r["columns"]):
            col_file_count[col] += 1
    common_cols = {col for col, cnt in col_file_count.items() if cnt >= 2}

    # ── Second pass: build datasets list and accumulate aggregates ────────────
    datasets = []
    for r in raw:
        base         = r["base"]
        profiling    = r["profiling"]
        primary_keys = r["primary_keys"]
        columns      = r["columns"]

        # Foreign keys = columns in this file that are also in at least one other file
        foreign_keys = [col for col in columns if col in common_cols]

        rp = profiling.get("row_processing_summary", {})
        mv = profiling.get("missing_values_summary", {})
        tr = profiling.get("transformation_summary", {})
        n_rows = profiling.get("number_of_rows", 0)

        agg["file_count"]                += 1
        agg["total_rows_before"]         += rp.get("original_row_count", n_rows)
        agg["total_rows_after"]          += rp.get("final_row_count",    n_rows)
        agg["total_columns"]             += profiling.get("number_of_columns", 0)
        agg["total_missing_before"]      += mv.get("total_missing_before",   0)
        agg["total_missing_after"]       += mv.get("total_missing_after",    0)
        agg["total_duplicates_removed"]  += rp.get("duplicate_rows_removed", 0)
        agg["total_columns_transformed"] += tr.get("columns_transformed",    0)
        agg["total_columns_dropped"]     += tr.get("columns_dropped",        0)
        agg["total_rows_dropped"]        += rp.get("dropped_missing_rows",   0)
        agg["all_primary_keys"] += [{"file": base, "column": k} for k in primary_keys]
        agg["all_foreign_keys"] += [{"file": base, "column": k} for k in foreign_keys]

        datasets.append({
            "filename":     base,
            "profiling":    profiling,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "columns":      columns,
            "preview":      r["preview"],
        })

    # ── Load or generate constraints ──────────────────────────────────────────
    try:
        constraints = download_json(f"meta_data/{user_id}/{session_id}/constraints.json")
    except Exception:
        try:
            constraints = ConstraintGenerator(user_id, session_id).generate()
        except Exception:
            constraints = {"primary_keys": [], "unique_keys": [], "foreign_keys": []}

    return {
        "session_id":  session_id,
        "user_id":     user_id,
        "total":       agg,
        "datasets":    datasets,
        "constraints": constraints,
    }


@router.get("/eda/{session_id}/{filename}/{column}")
def get_column_eda(
    session_id: str,
    filename:   str,
    column:     str,
    user_id: str = Depends(_get_current_user_id),
):
    """Return EDA stats + chart data for a single column in a cleaned CSV."""
    try:
        content = download_file(f"output/{user_id}/{session_id}/{filename}_cleaned.csv")
    except Exception:
        raise HTTPException(status_code=404, detail="Cleaned CSV not found")

    df = pd.read_csv(io.BytesIO(content))

    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    series = df[column]
    total      = len(series)
    null_count = int(series.isna().sum())

    result = {
        "column":           column,
        "total_rows":       total,
        "null_count":       null_count,
        "null_percentage":  round(null_count / total * 100, 2) if total else 0.0,
        "unique_count":     int(series.nunique()),
    }

    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        result["dtype"] = "numeric"
        result["min"]      = round(float(s.min()),      4) if len(s) else None
        result["max"]      = round(float(s.max()),      4) if len(s) else None
        result["sum"]      = round(float(s.sum()),      4) if len(s) else None
        result["range"]    = round(float(s.max() - s.min()), 4) if len(s) else None
        result["mean"]     = round(float(s.mean()),     4) if len(s) else None
        result["median"]   = round(float(s.median()),   4) if len(s) else None
        result["std"]      = round(float(s.std()),      4) if len(s) else None
        result["variance"] = round(float(s.var()),      4) if len(s) else None
        result["skewness"] = round(float(s.skew()),     4) if len(s) else None
        result["kurtosis"] = round(float(s.kurtosis()), 4) if len(s) else None
        result["q1"]       = round(float(s.quantile(0.25)), 4) if len(s) else None
        result["q2"]       = round(float(s.quantile(0.50)), 4) if len(s) else None
        result["q3"]       = round(float(s.quantile(0.75)), 4) if len(s) else None
        result["iqr"]      = round(float(s.quantile(0.75) - s.quantile(0.25)), 4) if len(s) else None

        # Histogram bins (up to 20)
        n_bins = min(20, max(2, int(s.nunique())))
        counts, bins = np.histogram(s, bins=n_bins)
        result["chart"] = {
            "type":   "histogram",
            "labels": [f"{bins[i]:.4g}" for i in range(len(bins) - 1)],
            "values": [int(c) for c in counts],
        }
    else:
        result["dtype"] = "categorical"
        vc = series.dropna().astype(str).value_counts().head(20)
        result["mode"] = str(vc.index[0]) if len(vc) > 0 else None
        result["chart"] = {
            "type":   "bar",
            "labels": [str(k) for k in vc.index],
            "values": [int(v) for v in vc.values],
        }

    return result
