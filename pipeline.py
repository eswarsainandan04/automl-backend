"""
pipeline.py — Preprocessing Pipeline Orchestrator
==================================================

This is the single source of truth for the AutoML preprocessing pipeline.
The pipeline runs five steps in strict order for every uploaded session.

Step order
----------
1. profiling            → Scans raw CSV/XLSX and builds per-column statistics
                          (dtype, null %, unique count, sample values).
                          Output: meta_data/{user}/{session}/*_profiling.json

2. column_handler       → Detects semantic types (currency, date, percentage …)
                          and renames / transforms columns accordingly.
                          Output: output/{user}/{session}/*_cleaned.csv

3. column_type_resolver → Resolves remaining varchar/object columns:
                          splits ranges (e.g. "10-20 km") into _min/_max,
                          extracts embedded numerics, skips ID-like text.
                          Output: updates *_cleaned.csv in-place

4. row_handler          → Drops duplicate rows; updates profiling metadata
                          with new row counts.
                          Output: updates *_cleaned.csv in-place

5. missing_values       → AutoGluon-based imputation (numeric, temporal,
                          categorical). Runs in a subprocess using the
                          backend Python (or AUTOGLUON_PYTHON when set).
                          Executable: AUTOGLUON_PYTHON (configured below)
                          Output: updates *_cleaned.csv in-place

How to add a new step
---------------------
1. Write your preprocessing module in backend/data_preprocessing/.
   It must expose:  process_user_datasets(user_id: str, session_id: str)

2. Add the step name to PIPELINE_STEPS (list below).

3. Add a corresponding entry to STEP_META (for the UI label / icon).

4. Add a `_run_step_<name>` function following the same pattern as the
   existing ones, then call it inside run_preprocessing() in order.

That's it — the FastAPI endpoint and the frontend UI pick up changes
automatically through PIPELINE_STEPS / STEP_META.
"""

import os
import subprocess
import sys
from datetime import datetime
from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Python executable that has AutoGluon installed.
# Step 5 runs in a subprocess and defaults to the current backend interpreter.
def _resolve_autogluon_python() -> str:
    """Return the best available Python executable for AutoGluon step."""
    env_python = os.getenv("AUTOGLUON_PYTHON")
    if env_python:
        return env_python

    current_python = sys.executable
    if current_python and os.path.exists(current_python):
        return current_python

    # Default cloud/container location (Dockerfile sets this conda env).
    linux_conda = "/opt/conda/envs/py37/bin/python"
    if os.path.exists(linux_conda):
        return linux_conda

    raise FileNotFoundError(
        "Python executable for missing-values subprocess not found. "
        "Set AUTOGLUON_PYTHON explicitly."
    )


AUTOGLUON_PYTHON = _resolve_autogluon_python()

# Absolute path to the standalone runner script for step 5.
# It imports AutoGluonMissingValueHandler and calls process_all_datasets().
_BACKEND_ROOT       = os.path.dirname(os.path.abspath(__file__))
MISSING_VALUES_RUNNER = os.path.join(
    _BACKEND_ROOT, "data_preprocessing", "missing_values_runner.py"
)

# Ordered list of step identifiers.
# This list drives the status dict shown in the UI.
PIPELINE_STEPS = [
    "profiling",            # Step 1
    "column_handler",       # Step 2
    "column_type_resolver", # Step 3
    "row_handler",          # Step 4
    "missing_values",       # Step 5
    "structural_type",      # Step 6
]

# Display metadata consumed by the frontend preprocessing page.
STEP_META = {
    "profiling":            {"label": "Data Profiling",       "icon": "🔍"},
    "column_handler":       {"label": "Column Detection",     "icon": "📋"},
    "column_type_resolver": {"label": "Type Resolution",      "icon": "🔎"},
    "row_handler":          {"label": "Row Handler",          "icon": "📏"},
    "missing_values":       {"label": "Missing Values",       "icon": "🩹"},
    "structural_type":      {"label": "Attribute Typing",     "icon": "🏷️"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_step(status: dict, name: str, step_status: str, error: str = None):
    """Update a single step's status inside the shared status dict."""
    for step in status["steps"]:
        if step["name"] == name:
            step["status"] = step_status
            if error:
                step["error"] = error
            break


def _fail(status: dict, step_name: str, exc: Exception, label: str) -> None:
    """Mark a step as errored and set the overall pipeline status to 'error'."""
    _set_step(status, step_name, "error", str(exc))
    status["status"] = "error"
    status["error"] = f"{label} failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Individual step runners
# (Each function returns True on success, False on failure.)
# ─────────────────────────────────────────────────────────────────────────────

def _run_step_profiling(user_id: str, session_id: str, status: dict) -> bool:
    """
    Step 1 — Data Profiling
    Reads every raw input file and writes a *_profiling.json metadata file.
    """
    _set_step(status, "profiling", "running")
    try:
        from data_preprocessing.profiling import process_user_datasets
        process_user_datasets(user_id, session_id)
        _set_step(status, "profiling", "done")
        return True
    except Exception as exc:
        _fail(status, "profiling", exc, "Profiling")
        return False


def _run_step_column_handler(user_id: str, session_id: str, status: dict) -> bool:
    """
    Step 2 — Column Handler
    Detects semantic column types and writes *_cleaned.csv output files.
    """
    _set_step(status, "column_handler", "running")
    try:
        from data_preprocessing.column_handler import process_user_datasets
        process_user_datasets(user_id, session_id)
        _set_step(status, "column_handler", "done")
        return True
    except Exception as exc:
        _fail(status, "column_handler", exc, "Column handler")
        return False


def _run_step_column_type_resolver(user_id: str, session_id: str, status: dict) -> bool:
    """
    Step 3 — Column Type Resolver
    Handles remaining varchar columns: splits ranges, extracts embedded
    numerics, drops ID-like / unstructured text columns.
    Operates on the *_cleaned.csv files produced by step 2.
    """
    _set_step(status, "column_type_resolver", "running")
    try:
        from data_preprocessing.column_type_resolver import process_user_datasets
        process_user_datasets(user_id, session_id)
        _set_step(status, "column_type_resolver", "done")
        return True
    except Exception as exc:
        _fail(status, "column_type_resolver", exc, "Column type resolver")
        return False


def _run_step_row_handler(user_id: str, session_id: str, status: dict) -> bool:
    """
    Step 4 — Row Handler
    Removes duplicate rows and updates profiling metadata with new row counts.
    """
    _set_step(status, "row_handler", "running")
    try:
        from data_preprocessing.row_handler import process_user_datasets
        process_user_datasets(user_id, session_id)
        _set_step(status, "row_handler", "done")
        return True
    except Exception as exc:
        _fail(status, "row_handler", exc, "Row handler")
        return False


def _run_step_missing_values(user_id: str, session_id: str, status: dict) -> bool:
    """
    Step 5 — Missing Values Handler (AutoGluon)
    Imputes missing values using AutoGluon-based strategies.

    WHY a subprocess?
    AutoGluon is a large ML library that is installed in a separate virtual
    environment (AUTOGLUON_PYTHON). Importing it inside the main FastAPI
    process would fail with ImportError.  Running it as a subprocess lets
    the two environments stay independent.
    """
    _set_step(status, "missing_values", "running")
    try:
        result = subprocess.run(
            [AUTOGLUON_PYTHON, MISSING_VALUES_RUNNER, user_id, session_id],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute hard limit per session
        )
        if result.returncode != 0:
            error_msg = (
                result.stderr or result.stdout or "subprocess exited with non-zero code"
            ).strip()
            raise RuntimeError(error_msg)
        _set_step(status, "missing_values", "done")
        return True
    except subprocess.TimeoutExpired:
        _fail(status, "missing_values",
              RuntimeError("Timed out after 600 s"), "Missing values handler")
        return False
    except Exception as exc:
        _fail(status, "missing_values", exc, "Missing values handler")
        return False


def _run_step_structural_type(user_id: str, session_id: str, status: dict) -> bool:
    """
    Step 6 — Structural Type Detection
    Reads every *_profiling.json for the session, runs StructuralTypeDetector
    on each column, and writes the 'structural_type' field back into the JSON.
    The result is stored in Supabase so the overview endpoint can read it.
    """
    _set_step(status, "structural_type", "running")
    try:
        from data_preprocessing.supabase_storage import download_json, list_files, upload_json
        from data_preprocessing.structural_type_detector import StructuralTypeDetector

        detector = StructuralTypeDetector()
        meta_prefix = f"meta_data/{user_id}/{session_id}"

        try:
            meta_files = list_files(meta_prefix)
        except Exception:
            meta_files = []

        profiling_files = [f for f in meta_files if f.endswith("_profiling.json")]

        for pf in profiling_files:
            profiling = download_json(f"{meta_prefix}/{pf}")
            total_rows = profiling.get("number_of_rows", 0)
            for col in profiling.get("column_wise_summary", []):
                col["structural_type"] = detector.detect(col, total_rows)
            upload_json(f"{meta_prefix}/{pf}", profiling)

        _set_step(status, "structural_type", "done")
        return True
    except Exception as exc:
        _fail(status, "structural_type", exc, "Structural type detection")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by upload_files.py in a background thread
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(user_id: str, session_id: str, status: dict) -> None:
    """
    Run all preprocessing steps in order for a given user session.

    Parameters
    ----------
    user_id    : ID of the user who uploaded the files.
    session_id : Unique identifier for this upload/preprocessing session.
    status     : Shared mutable dict (from _pipeline_status) updated in-place
                 so the /upload/status endpoint can reflect live progress.

    Pipeline execution is intentionally sequential — each step depends on the
    output of the previous one.  If any step fails the pipeline stops and the
    overall status is set to "error".
    """
    try:
        # ── Step 1 ────────────────────────────────────────────────────────────
        if not _run_step_profiling(user_id, session_id, status):
            return

        # ── Step 2 ────────────────────────────────────────────────────────────
        if not _run_step_column_handler(user_id, session_id, status):
            return

        # ── Step 3 ────────────────────────────────────────────────────────────
        if not _run_step_column_type_resolver(user_id, session_id, status):
            return

        # ── Step 4 ────────────────────────────────────────────────────────────
        if not _run_step_row_handler(user_id, session_id, status):
            return

        # ── Step 5 ────────────────────────────────────────────────────────────
        if not _run_step_missing_values(user_id, session_id, status):
            return

        # ── Step 6 ────────────────────────────────────────────────────────────
        if not _run_step_structural_type(user_id, session_id, status):
            return

        # All steps completed successfully
        status["status"] = "done"
        status["completed_at"] = datetime.utcnow().isoformat()

    except Exception as exc:
        # Unexpected error outside a step runner
        status["status"] = "error"
        status["error"] = str(exc)
