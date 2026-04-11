"""
model_info.py — Model Report Info API
======================================
Provides a rich, frontend-ready model report endpoint that:
  - Downloads the _model_report.json from Supabase
  - Normalises and enriches it for the frontend dashboard
  - Returns classification OR regression metrics in a structured shape
  - Exposes chart-ready data (confusion matrix, residuals, leaderboard, etc.)

Endpoint:
    GET /automl/model-info/{session_id}/{dataset_base}
    → Returns ModelInfoResponse (see schema below)

This is imported and registered in automl_router.py.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt
import psycopg2

from config import (
    POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD,
    POSTGRES_PORT, POSTGRES_USER,
)
from jwt_handler import ALGORITHM, SECRET_KEY
from data_preprocessing.supabase_storage import download_json

logger = logging.getLogger("ModelInfo")
router = APIRouter(prefix="/automl", tags=["AutoML"])
security = HTTPBearer()


# ─── Auth helpers ─────────────────────────────────────────────────────────────

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


# ─── Normalisation helpers ────────────────────────────────────────────────────

def _safe_float(v, digits: int = 6) -> Optional[float]:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def _normalise_classification_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull all classification metrics into a normalised flat dict.
    Supports both the new 'evaluation_metrics' key and legacy flat keys.
    """
    em = report.get("evaluation_metrics") or report

    def _g(key):
        return _safe_float(em.get(key) or report.get(key))

    return {
        # Core accuracy
        "accuracy":          _g("accuracy"),
        "balanced_accuracy": _g("balanced_accuracy"),
        # Precision / Recall / F1 — macro
        "precision_macro":   _g("precision_macro"),
        "recall_macro":      _g("recall_macro"),
        "f1_macro":          _g("f1_macro"),
        # Precision / Recall / F1 — weighted
        "precision_weighted": _g("precision_weighted"),
        "recall_weighted":    _g("recall_weighted"),
        "f1_weighted":        _g("f1_weighted"),
        # Precision / Recall / F1 — micro
        "precision_micro":   _g("precision_micro"),
        "recall_micro":      _g("recall_micro"),
        "f1_micro":          _g("f1_micro"),
        # F-beta
        "f2_macro":          _g("f2_macro"),
        # Specificity (binary only)
        "specificity":       _g("specificity"),
        # Probability-based
        "roc_auc":           _g("roc_auc"),
        "pr_auc":            _g("pr_auc"),
        "log_loss":          _g("log_loss"),
        # Statistical
        "mcc":               _g("mcc"),
        "cohen_kappa":       _g("cohen_kappa"),
        # Error-based
        "hamming_loss":      _g("hamming_loss"),
        "jaccard_macro":     _g("jaccard_macro"),
        # Top-K
        "top_k_accuracy_k3": _g("top_k_accuracy_k3"),
        # Per-class detail
        "classes":           em.get("classes") or [],
        "per_class_precision": em.get("per_class_precision") or [],
        "per_class_recall":    em.get("per_class_recall") or [],
        "per_class_f1":        em.get("per_class_f1") or [],
        # Confusion matrix
        "confusion_matrix":  em.get("confusion_matrix") or [],
    }


def _normalise_regression_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    em = report.get("evaluation_metrics") or report

    def _g(key):
        return _safe_float(em.get(key) or report.get(key))

    return {
        "r2":               _g("r2"),
        "adj_r2":           _g("adj_r2"),
        "mae":              _g("mae"),
        "mse":              _g("mse"),
        "rmse":             _g("rmse"),
        "rmsle":            _g("rmsle"),
        "mape":             _g("mape"),
        "smape":            _g("smape"),
        "median_abs_error": _g("median_abs_error"),
        "explained_var":    _g("explained_var"),
        # Residuals
        "residuals_mean":  _g("residuals_mean"),
        "residuals_std":   _g("residuals_std"),
        "residuals_hist_counts": em.get("residuals_hist_counts") or [],
        "residuals_hist_edges":  em.get("residuals_hist_edges") or [],
        # Actual vs predicted
        "actual_vs_predicted": em.get("actual_vs_predicted") or {"actual": [], "predicted": []},
    }


def _normalise_leaderboard(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    task = str(report.get("task", "classification")).lower()
    raw = report.get("leaderboard") or []
    out = []
    for entry in raw:
        # cv score key may be 'cv_accuracy' or 'cv_r2_score'
        cv_key = "cv_accuracy" if task == "classification" else "cv_r2_score"
        cv_val = _safe_float(entry.get("cv_score") or entry.get(cv_key) or entry.get("cv_r2"))
        out.append({
            "model_name":        str(entry.get("model_name", "")),
            "scaling_required":  bool(entry.get("scaling_required", False)),
            "cv_score":          cv_val,
            "cv_std":            _safe_float(entry.get("cv_std")),
            "cv_scores":         entry.get("cv_scores") or [],
            "train_time_seconds": _safe_float(entry.get("train_time_seconds")),
            "hpo_status":        entry.get("hpo_status"),
            "hpo_best_score":    _safe_float(entry.get("hpo_best_score")),
            "hpo_best_params":   entry.get("hpo_best_params") or {},
            "hpo_trials":        entry.get("hpo_trials"),
            "hpo_time_seconds":  _safe_float(entry.get("hpo_time_seconds")),
        })
    return out


def _build_feature_importance_chart(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return feature_importance as {labels, values} ready for Chart.js."""
    fi = {}

    # Try features metadata embedded in report (older reports may not have this)
    raw_fi = report.get("feature_importance") or {}
    if not raw_fi:
        # Fallback: treat feature processing plan as presence only
        fp = report.get("feature_processing") or {}
        for feat in report.get("selected_features") or []:
            fi[feat] = None
    else:
        fi = raw_fi

    if not fi:
        return {"labels": [], "values": []}

    # Sort by value descending (skip None)
    sortable = {k: v for k, v in fi.items() if v is not None}
    sorted_items = sorted(sortable.items(), key=lambda x: x[1], reverse=True)

    return {
        "labels": [item[0] for item in sorted_items],
        "values": [_safe_float(item[1]) for item in sorted_items],
    }


def _resolve_metrics_representation(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve which model/system produced the metrics shown in the UI.

    In most pipelines, evaluation metrics are generated from the final model
    (single model or ensemble). This helper makes that explicit for the client.
    """
    final_model_raw = str(report.get("final_model") or "").strip()
    final_model = final_model_raw.lower()
    best_model = str(report.get("best_model") or "").strip()

    ensemble = report.get("ensemble") or {}
    ensemble_enabled = bool(ensemble.get("enabled"))
    ensemble_models = ensemble.get("models") or report.get("top_k_models") or []
    voting = str(ensemble.get("voting") or "hard")

    if final_model == "ensemble" or ensemble_enabled:
        top_k = ensemble.get("top_k") or (len(ensemble_models) if ensemble_models else None)
        return {
            "type": "ensemble",
            "name": "Ensemble",
            "display_name": f"Ensemble ({voting.capitalize()} voting)",
            "members": ensemble_models,
            "voting": voting,
            "top_k": top_k,
        }

    if final_model and final_model != "single":
        return {
            "type": "model",
            "name": final_model_raw,
            "display_name": final_model_raw,
            "members": [],
            "voting": None,
            "top_k": None,
        }

    resolved_name = best_model or final_model_raw or "unknown"
    return {
        "type": "model",
        "name": resolved_name,
        "display_name": resolved_name,
        "members": [],
        "voting": None,
        "top_k": None,
    }


def build_model_info_response(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the full, frontend-ready model info response from raw report JSON."""
    task = str(report.get("task", "classification")).lower()

    if task == "classification":
        metrics = _normalise_classification_metrics(report)
    else:
        metrics = _normalise_regression_metrics(report)

    leaderboard = _normalise_leaderboard(report)
    feature_chart = _build_feature_importance_chart(report)
    metrics_representation = _resolve_metrics_representation(report)

    hpo_block = report.get("hpo") or {}
    hpo_payload = {
        "status": hpo_block.get("status") or report.get("hpo_status"),
        "model_name": hpo_block.get("best_model") or hpo_block.get("model_name") or report.get("hpo_model"),
        "best_score": _safe_float(hpo_block.get("best_score") or report.get("hpo_best_score")),
        "best_params": hpo_block.get("best_params") or report.get("hpo_best_params") or {},
        "n_trials": hpo_block.get("per_model_trials") or hpo_block.get("n_trials") or report.get("hpo_trials"),
        "cv": hpo_block.get("cv") or report.get("hpo_cv"),
        "time_seconds": _safe_float(hpo_block.get("time_seconds") or report.get("hpo_time_seconds")),
        "total_models": hpo_block.get("total_models"),
        "tuned_models": hpo_block.get("tuned_models"),
        "failed_models": hpo_block.get("failed_models"),
        "skipped_models": hpo_block.get("skipped_models"),
        "top_k": hpo_block.get("top_k") or report.get("top_k"),
    }

    return {
        # Identity
        "status":          report.get("status", "completed"),
        "dataset_base":    report.get("dataset_base", ""),
        "task":            task,
        "framework":       report.get("framework", "sklearn"),
        "target":          report.get("target", ""),
        "row_count":       report.get("row_count", 0),
        "feature_count":   report.get("feature_count", 0),
        "selected_features": report.get("selected_features") or [],
        "feature_types":   report.get("feature_types", ""),
        "feature_processing": report.get("feature_processing") or {},
        "dropped_columns": report.get("dropped_columns") or [],
        # Best model summary
        "best_model":      report.get("best_model", ""),
        "best_score":      _safe_float(report.get("best_score") or report.get("cv_score")),
        "cv_score":        _safe_float(report.get("cv_score")),
        "cv_std":          _safe_float(report.get("cv_std")),
        "final_validation_score": _safe_float(report.get("final_validation_score")),
        "metric":          report.get("metric", ""),
        "models_trained":  report.get("models_trained", 0),
        "models_failed":   report.get("models_failed", 0),
        "total_pipeline_time_seconds": _safe_float(report.get("total_pipeline_time_seconds")),
        "baseline_cv_score": _safe_float(report.get("baseline_cv_score") or report.get("cv_score")),
        "baseline_cv_std": _safe_float(report.get("baseline_cv_std") or report.get("cv_std")),
        "hpo": hpo_payload,
        "hpo_models": report.get("hpo_models") or [],
        "ensemble": report.get("ensemble") or {},
        "final_model": report.get("final_model") or "single",
        "top_k_models": report.get("top_k_models") or [],
        # Explicitly tell UI which model/system generated the displayed metrics.
        "metrics_model_type": metrics_representation.get("type"),
        "metrics_model_name": metrics_representation.get("name"),
        "metrics_model_display_name": metrics_representation.get("display_name"),
        "metrics_representation": metrics_representation,
        # Full metrics (classification or regression)
        "metrics":         metrics,
        # Leaderboard
        "leaderboard":     leaderboard,
        # Feature importance chart data
        "feature_importance_chart": feature_chart,
        # Raw evaluation_metrics passthrough (for client-side flexibility)
        "evaluation_metrics": report.get("evaluation_metrics") or {},
    }


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.get("/model-info/{session_id}/{dataset_base}")
def get_model_info(
    session_id: str,
    dataset_base: str,
    user_id: str = Depends(_get_current_user_id),
):
    """
    Return a rich, frontend-ready model report for the given session + dataset.

    Downloads _model_report.json from Supabase and normalises it into a
    consistent shape containing all classification or regression metrics,
    leaderboard, feature importance chart data, and chart-ready confusion
    matrix / residuals.
    """
    report_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_model_report.json"
    try:
        report = download_json(report_path)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model report not found for dataset '{dataset_base}'. "
                   f"Run model selection first. ({exc})",
        )

    try:
        response = build_model_info_response(report)
    except Exception as exc:
        logger.exception("Failed to build model info response: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to process model report: {exc}")

    return response