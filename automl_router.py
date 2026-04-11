from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt
from pydantic import BaseModel, Field

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from feature_engineering.feature_selection import (
    create_target_column_for_session,
    get_feature_selection_for_session,
    list_session_cleaned_datasets,
    recommend_features_for_session,
    save_feature_selection_for_session,
)
from jwt_handler import ALGORITHM, SECRET_KEY
from model_testing import get_model_report_supabase
# ── NEW: import model_info router ─────────────────────────────────────────────
from model_info import build_model_info_response
from data_preprocessing.supabase_storage import download_json, list_files

router = APIRouter(prefix="/automl", tags=["AutoML"])
security = HTTPBearer()
logger = logging.getLogger("AutoMLRouter")


class FeatureRecommendBody(BaseModel):
    dataset_base: str
    target: str


class FeatureSelectBody(BaseModel):
    dataset_base: str
    target: str
    selected_features: List[str] = Field(default_factory=list)


class ModelRunBody(BaseModel):
    dataset_base: str


class PredictBody(BaseModel):
    dataset_base: str
    row: Dict[str, Any]
    confidence_threshold_percent: float = Field(default=50.0, ge=0.0, le=100.0)


class TargetConditionBody(BaseModel):
    metric: str
    operator: str
    value: Any
    value2: Any | None = None
    output_value: Any | None = None


class CreateTargetBody(BaseModel):
    dataset_base: str
    target_column_name: str
    metric_columns: List[str] = Field(default_factory=list)
    target_type: str = Field(description="binary_classification | multiclass_classification | regression")
    conditions: List[TargetConditionBody] = Field(default_factory=list)
    custom_expression: str | None = None
    default_value: Any | None = None


# Auth helpers

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
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
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


def _resolve_model_selection_python() -> Path:
    """Resolve Python executable used for model selection/model testing subprocesses."""
    for env_name in ("MODEL_SELECTION_PYTHON", "FLAML_PYTHON"):
        configured = os.getenv(env_name)
        if not configured:
            continue
        p = Path(configured).expanduser()
        if p.exists() and p.is_file():
            return p

    current_python = Path(sys.executable).expanduser()
    if current_python.exists() and current_python.is_file():
        return current_python

    legacy_autogluon = os.getenv("AUTOGLUON_PYTHON")
    if legacy_autogluon:
        p = Path(legacy_autogluon).expanduser()
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        "Dedicated model-selection Python not found. Set MODEL_SELECTION_PYTHON "
        "to your model-selection environment executable."
    )


def _run_model_selection_supabase_in_dedicated_env(
    user_id: str,
    session_id: str,
    dataset_base: str,
) -> Dict[str, Any]:
    """Run model selection in dedicated env so FastAPI can stay on a different Python version."""
    python_exe = _resolve_model_selection_python()
    backend_dir = Path(__file__).resolve().parent
    sentinel = "__MODEL_SELECTION_JSON__"

    code = (
        "import json,sys;"
        "from model_building.model_selection import run_model_selection_supabase;"
        "r=run_model_selection_supabase(user_id=sys.argv[1],session_id=sys.argv[2],dataset_base=sys.argv[3]);"
        f"print('{sentinel}'+json.dumps(r))"
    )

    logger.info(
        "Launching model-selection subprocess | session_id=%s | dataset_base=%s | python=%s",
        session_id, dataset_base, python_exe,
    )

    proc = subprocess.run(
        [str(python_exe), "-c", code, user_id, session_id, dataset_base],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
    )

    stdout_lines = (proc.stdout or "").splitlines()
    stderr_lines = (proc.stderr or "").splitlines()
    for line in stdout_lines:
        if line.startswith(sentinel):
            continue
        if line.strip():
            logger.info("[model-selection] %s", line)
    for line in stderr_lines:
        if line.strip():
            logger.warning("[model-selection][stderr] %s", line)

    logger.info(
        "Model-selection subprocess finished | session_id=%s | dataset_base=%s | exit_code=%s",
        session_id, dataset_base, proc.returncode,
    )

    if proc.returncode != 0:
        # FIX-R2: include the FULL stderr so the real Python traceback is visible
        # in the API response and in server logs. The old 20-line tail frequently
        # truncated the actual error, hiding the root cause.
        full_stderr = proc.stderr or ""
        full_stdout = proc.stdout or ""
        detail = (
            f"Model selection subprocess failed "
            f"(python={python_exe}, exit_code={proc.returncode}).\n"
            f"--- stderr ---\n{full_stderr}\n"
            f"--- stdout ---\n{full_stdout}"
        )
        raise RuntimeError(detail)

    for line in reversed(stdout_lines):
        if line.startswith(sentinel):
            payload = line[len(sentinel):].strip()
            return json.loads(payload)

    raise RuntimeError(
        "Model selection subprocess completed but did not return structured report output."
    )


def _run_model_testing_function_in_dedicated_env(
    function_name: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run selected model_testing function in dedicated env (AutoGluon-capable)."""
    python_exe = _resolve_model_selection_python()
    backend_dir = Path(__file__).resolve().parent
    sentinel = "__MODEL_TESTING_JSON__"

    code = (
        "import json,sys;"
        "import model_testing;"
        "fn=getattr(model_testing, sys.argv[1]);"
        "kwargs=json.loads(sys.argv[2]);"
        "res=fn(**kwargs);"
        f"print('{sentinel}'+json.dumps(res))"
    )

    logger.info(
        "Launching model-testing subprocess | function=%s | python=%s",
        function_name, python_exe,
    )

    proc = subprocess.run(
        [str(python_exe), "-c", code, function_name, json.dumps(kwargs)],
        cwd=str(backend_dir),
        capture_output=True,
        text=True,
    )

    stdout_lines = (proc.stdout or "").splitlines()
    stderr_lines = (proc.stderr or "").splitlines()
    for line in stdout_lines:
        if line.startswith(sentinel):
            continue
        if line.strip():
            logger.info("[model-testing:%s] %s", function_name, line)
    for line in stderr_lines:
        if line.strip():
            logger.warning("[model-testing:%s][stderr] %s", function_name, line)

    logger.info(
        "Model-testing subprocess finished | function=%s | exit_code=%s",
        function_name, proc.returncode,
    )

    if proc.returncode != 0:
        # FIX-R3: full stderr so real tracebacks are visible (same fix as model-selection)
        full_stderr = proc.stderr or ""
        full_stdout = proc.stdout or ""
        detail = (
            f"Model testing subprocess failed "
            f"(python={python_exe}, function={function_name}, exit_code={proc.returncode}).\n"
            f"--- stderr ---\n{full_stderr}\n"
            f"--- stdout ---\n{full_stdout}"
        )
        raise RuntimeError(detail)

    for line in reversed(stdout_lines):
        if line.startswith(sentinel):
            payload = line[len(sentinel):].strip()
            return json.loads(payload)

    raise RuntimeError(
        f"Model testing subprocess completed but did not return structured output for {function_name}."
    )


# ─── Feature Engineering Endpoints ───────────────────────────────────────────

@router.get("/feature-engineering/status/{session_id}/{dataset_base}")
def feature_saved_status(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Check whether feature selection JSON exists for a dataset/session."""
    meta_prefix = f"meta_data/{user_id}/{session_id}"
    expected = f"{dataset_base}_features.json"
    try:
        files = list_files(meta_prefix)
    except Exception:
        files = []

    return {
        "session_id": session_id,
        "dataset_base": dataset_base,
        "generated": expected in files,
    }

@router.get("/feature-engineering/datasets/{session_id}")
def feature_datasets(
    session_id: str,
    user_id: str = Depends(_get_current_user_id),
):
    try:
        datasets = list_session_cleaned_datasets(user_id=user_id, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    meta_prefix = f"meta_data/{user_id}/{session_id}"
    try:
        meta_files = set(list_files(meta_prefix))
    except Exception:
        meta_files = set()

    enriched_datasets: List[Dict[str, Any]] = []
    for ds in datasets:
        base = str(ds.get("dataset_base", ""))
        feature_file = f"{base}_features.json"
        model_file = f"{base}_model_report.json"
        enriched = dict(ds)
        enriched["feature_engineering_completed"] = feature_file in meta_files
        enriched["model_building_completed"] = model_file in meta_files
        enriched_datasets.append(enriched)

    dataset_count = len(enriched_datasets)
    single_dataset_base = enriched_datasets[0].get("dataset_base") if dataset_count == 1 else None

    return {
        "session_id": session_id,
        "datasets": enriched_datasets,
        "dataset_count": dataset_count,
        "requires_dataset_selection": dataset_count > 1,
        "single_dataset_base": single_dataset_base,
        "any_feature_engineering_completed": any(d.get("feature_engineering_completed") for d in enriched_datasets),
        "any_model_building_completed": any(d.get("model_building_completed") for d in enriched_datasets),
    }


@router.get("/feature-engineering/saved/{session_id}/{dataset_base}")
def feature_saved(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    try:
        data = get_feature_selection_for_session(
            user_id=user_id,
            session_id=session_id,
            dataset_base=dataset_base,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data


@router.post("/feature-engineering/recommend/{session_id}")
def feature_recommend(
    session_id: str,
    body: FeatureRecommendBody,
    user_id: str = Depends(_get_current_user_id),
):
    try:
        data = recommend_features_for_session(
            user_id=user_id,
            session_id=session_id,
            dataset_base=body.dataset_base,
            target=body.target,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data


@router.post("/feature-engineering/select/{session_id}")
def feature_select(
    session_id: str,
    body: FeatureSelectBody,
    user_id: str = Depends(_get_current_user_id),
):
    try:
        payload = save_feature_selection_for_session(
            user_id=user_id,
            session_id=session_id,
            dataset_base=body.dataset_base,
            target=body.target,
            selected_features=body.selected_features,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "saved",
        "session_id": session_id,
        "dataset_base": body.dataset_base,
        "target": payload.get("target"),
        "selected_features": payload.get("selected_features", []),
        "feature_importance": payload.get("feature_importance", {}),
        "task": payload.get("task"),
    }


@router.post("/feature-engineering/create-target/{session_id}")
def create_target_column(
    session_id: str,
    body: CreateTargetBody,
    user_id: str = Depends(_get_current_user_id),
):
    try:
        payload = create_target_column_for_session(
            user_id=user_id,
            session_id=session_id,
            dataset_base=body.dataset_base,
            target_column_name=body.target_column_name,
            metric_columns=body.metric_columns,
            target_type=body.target_type,
            conditions=[c.model_dump() for c in body.conditions],
            custom_expression=body.custom_expression,
            default_value=body.default_value,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return payload


# ─── Model Building Endpoints ─────────────────────────────────────────────────

@router.get("/model-building/status/{session_id}/{dataset_base}")
def model_building_status(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """Check whether model report JSON exists for a dataset/session."""
    meta_prefix = f"meta_data/{user_id}/{session_id}"
    expected = f"{dataset_base}_model_report.json"
    try:
        files = list_files(meta_prefix)
    except Exception:
        files = []

    return {
        "session_id": session_id,
        "dataset_base": dataset_base,
        "generated": expected in files,
    }

@router.post("/model-building/run/{session_id}")
def run_model_building(
    session_id: str,
    body: ModelRunBody,
    user_id: str = Depends(_get_current_user_id),
):
    logger.info("API request received | route=model-building/run | session_id=%s", session_id)
    try:
        report = _run_model_selection_supabase_in_dedicated_env(
            user_id=user_id,
            session_id=session_id,
            dataset_base=body.dataset_base,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return report


@router.get("/model-building/report/{session_id}/{dataset_base}")
def model_building_report(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Legacy endpoint — returns raw model report.
    Prefer /automl/model-info/{session_id}/{dataset_base} for the enriched response.
    """
    try:
        data = get_model_report_supabase(
            user_id=user_id,
            session_id=session_id,
            dataset_base=dataset_base,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data


# ─── NEW: Enriched Model Info Endpoint ───────────────────────────────────────

@router.get("/model-info/{session_id}/{dataset_base}")
def get_model_info(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Return a rich, frontend-ready model report.

    Downloads _model_report.json from Supabase and normalises it into a
    consistent shape containing all classification or regression metrics,
    leaderboard with fold scores, feature importance chart data, and
    chart-ready confusion matrix / residuals histogram.
    """
    report_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_model_report.json"
    try:
        report = download_json(report_path)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model report not found for '{dataset_base}'. Run model selection first. ({exc})",
        )

    try:
        response = build_model_info_response(report)
    except Exception as exc:
        logger.exception("Failed to build model info response: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to process model report: {exc}")

    return response


# ─── Model Testing Endpoints ──────────────────────────────────────────────────

@router.get("/model-testing/schema/{session_id}/{dataset_base}")
def model_testing_schema(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    logger.info("API request received | route=model-testing/schema | session_id=%s | dataset_base=%s", session_id, dataset_base)
    try:
        data = _run_model_testing_function_in_dedicated_env(
            function_name="get_model_testing_schema_supabase",
            kwargs={
                "user_id": user_id,
                "session_id": session_id,
                "dataset_base": dataset_base,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data


@router.post("/model-testing/predict/{session_id}")
def model_testing_predict(
    session_id: str,
    body: PredictBody,
    user_id: str = Depends(_get_current_user_id),
):
    logger.info(
        "API request received | route=model-testing/predict | session_id=%s | dataset_base=%s | threshold_percent=%s",
        session_id, body.dataset_base, body.confidence_threshold_percent,
    )

    # FIX-R1: Pre-stringify all row values at the API boundary.
    #
    # JSON deserialization converts integer-encoded categorical fields (sex=0,
    # fbs=1, cp=2, …) to Python int. When these reach model_testing via the
    # subprocess, _coerce_input_df handles the conversion — but only if the
    # type_map is correctly populated (which the embedded pkl type_map now
    # guarantees). This pre-stringification is a belt-and-suspenders safety net:
    # even on old pkls without an embedded type_map, the values will already be
    # str "0"/"1" before _coerce_input_df is called, so OHE always gets what it
    # expects.
    #
    # Numeric columns are NOT affected: _coerce_input_df calls pd.to_numeric()
    # on them, so "120" → 120.0 works identically to 120 → 120.0.
    # Sending everything as str is therefore safe for all column types.
    sanitized_row: Dict[str, Any] = {
        k: str(v) if v is not None else None
        for k, v in body.row.items()
    }
    logger.debug("predict | sanitized_row: %s", sanitized_row)

    try:
        data = _run_model_testing_function_in_dedicated_env(
            function_name="predict_from_session_model",
            kwargs={
                "user_id": user_id,
                "session_id": session_id,
                "dataset_base": body.dataset_base,
                "row": sanitized_row,
                "confidence_threshold_percent": body.confidence_threshold_percent,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return data