"""
model_building/model_selection.py
================================
AutoML Model Selection + Training Pipeline

Flow:
1. Load profiling + features metadata from Supabase
2. Download cleaned CSV from Supabase
3. Enforce correct dtypes from profiling metadata
4. Detect global feature type from structural types
5. Build feature processing plan (encoding strategies)
6. Select candidate models based on task × dataset size × feature type
7. Split data → build preprocessor → transform training data once
8. Run Optuna HPO for EACH candidate model (transformed data)
9. Rank tuned models by HPO CV score → select top K
10. Ensemble top K (Voting) or use the single best model
11. Train final pipeline on full training data
12. Validate once on hold-out set (final_validation_score)
13. Compute FULL metrics suite (classification or regression)
14. Serialize pipeline + report, upload to Supabase

The pkl format is:
    {"framework": "sklearn", "pipeline": <fitted Pipeline>, "label_encoder": <LabelEncoder or None>}

This matches what model_testing.py expects for prediction.
"""

from __future__ import annotations

import io
import logging
import math
import pickle
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse

# ── Sklearn core ──────────────────────────────────────────────────────────────
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.ensemble import VotingClassifier, VotingRegressor

# ── Sklearn: Metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    hamming_loss,
    jaccard_score,
    confusion_matrix,
    classification_report,
    # Regression
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
)

# ── models.py: ML model imports, constants & logic ───────────────────────────
from .models import (
    TEXT_VEC_TYPES,
    _TEMPORAL_TYPES,
    _NEEDS_LABEL_ENCODING,
    _NEEDS_DENSE_INPUT,
    _MAX_TEXT_FEATURES,
    _build_model_instance,
    select_classification_models,
    select_regression_models,
)

# ── Internal project modules ─────────────────────────────────────────────────
from data_preprocessing.supabase_storage import (
    download_file,
    download_json,
    upload_file,
    upload_json,
)
from feature_engineering.feature_processing import (
    detect_feature_type,
    build_feature_processing_plan,
    build_fitted_preprocessor,
    build_training_stats,
)
from feature_engineering.data_split import split_dataset, enforce_dtypes
from hyper_parameter_optimization.auto_parameters import run_hpo


# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ModelSelection")


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL PICKLABLE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _squeeze_column(X) -> np.ndarray:
    if hasattr(X, "iloc"):
        return X.iloc[:, 0].fillna("").astype(str).values
    if issparse(X):
        return np.asarray(X.toarray()).ravel().astype(str)
    arr = np.asarray(X)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.ravel().astype(str)


def _sparse_to_dense(X) -> np.ndarray:
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


# ══════════════════════════════════════════════════════════════════════════════
# FULL METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _safe(fn, *args, default=None, **kwargs):
    """Call fn safely, returning default on any exception."""
    try:
        result = fn(*args, **kwargs)
        if result is None or (isinstance(result, float) and math.isnan(result)):
            return default
        return result
    except Exception:
        return default


def _round(v, digits=6):
    if v is None:
        return None
    try:
        return round(float(v), digits)
    except Exception:
        return None


def compute_classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    label_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, Any]:
    """
    Compute the full classification metrics suite.

    Parameters
    ----------
    y_true       : Ground-truth labels (original encoded form).
    y_pred       : Predicted labels (same encoding as y_true).
    y_prob       : Predicted probabilities (n_samples, n_classes) or None.
    label_encoder: LabelEncoder used for XGB/LGBM — decodes integer preds.

    Returns a flat dict of all metrics, safe against any computation failure.
    """
    # Decode integer labels if label encoder was used
    if label_encoder is not None:
        try:
            y_true = label_encoder.inverse_transform(y_true)
        except Exception:
            pass
        try:
            y_pred = label_encoder.inverse_transform(y_pred)
        except Exception:
            pass

    classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    is_binary = len(classes) == 2
    avg = "binary" if is_binary else "macro"

    metrics: Dict[str, Any] = {}

    # ── Core scores ──────────────────────────────────────────────────────────
    metrics["accuracy"]          = _round(_safe(accuracy_score, y_true, y_pred))
    metrics["balanced_accuracy"] = _round(_safe(balanced_accuracy_score, y_true, y_pred))
    metrics["precision_macro"]   = _round(_safe(precision_score,  y_true, y_pred, average="macro",  zero_division=0))
    metrics["recall_macro"]      = _round(_safe(recall_score,     y_true, y_pred, average="macro",  zero_division=0))
    metrics["f1_macro"]          = _round(_safe(f1_score,         y_true, y_pred, average="macro",  zero_division=0))
    metrics["precision_weighted"]= _round(_safe(precision_score,  y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall_weighted"]   = _round(_safe(recall_score,     y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_weighted"]       = _round(_safe(f1_score,         y_true, y_pred, average="weighted", zero_division=0))
    metrics["mcc"]               = _round(_safe(matthews_corrcoef, y_true, y_pred))
    metrics["cohen_kappa"]       = _round(_safe(cohen_kappa_score, y_true, y_pred))
    metrics["hamming_loss"]      = _round(_safe(hamming_loss,      y_true, y_pred))
    metrics["jaccard_macro"]     = _round(_safe(jaccard_score,     y_true, y_pred, average="macro",   zero_division=0))

    # F-beta (beta=2 weights recall higher)
    metrics["f2_macro"]          = _round(_safe(fbeta_score, y_true, y_pred, beta=2.0, average="macro", zero_division=0))

    # Micro averages
    metrics["precision_micro"]   = _round(_safe(precision_score, y_true, y_pred, average="micro", zero_division=0))
    metrics["recall_micro"]      = _round(_safe(recall_score,    y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_micro"]          = _round(_safe(f1_score,        y_true, y_pred, average="micro", zero_division=0))

    # ── Probability-based metrics ─────────────────────────────────────────────
    if y_prob is not None:
        try:
            y_prob_arr = np.asarray(y_prob)
            if is_binary:
                prob_pos = y_prob_arr[:, 1] if y_prob_arr.ndim == 2 else y_prob_arr
                metrics["roc_auc"]   = _round(_safe(roc_auc_score,          y_true, prob_pos))
                metrics["pr_auc"]    = _round(_safe(average_precision_score, y_true, prob_pos))
                metrics["log_loss"]  = _round(_safe(log_loss,                y_true, prob_pos))
            else:
                metrics["roc_auc"]   = _round(_safe(roc_auc_score,  y_true, y_prob_arr, multi_class="ovr", average="macro"))
                metrics["log_loss"]  = _round(_safe(log_loss,        y_true, y_prob_arr))
                metrics["pr_auc"]    = None  # PR-AUC not directly available for multi-class
        except Exception:
            metrics["roc_auc"]  = None
            metrics["pr_auc"]   = None
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"]  = None
        metrics["pr_auc"]   = None
        metrics["log_loss"] = None

    # ── Top-K accuracy ────────────────────────────────────────────────────────
    if y_prob is not None and not is_binary:
        try:
            from sklearn.metrics import top_k_accuracy_score
            metrics["top_k_accuracy_k3"] = _round(
                _safe(top_k_accuracy_score, y_true, np.asarray(y_prob), k=min(3, len(classes)))
            )
        except Exception:
            metrics["top_k_accuracy_k3"] = None
    else:
        metrics["top_k_accuracy_k3"] = None

    # ── Confusion matrix ──────────────────────────────────────────────────────
    try:
        cm = confusion_matrix(y_true, y_pred, labels=list(classes))
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classes"]          = [str(c) for c in classes]

        # Per-class precision / recall / f1
        per_class_p = _safe(precision_score, y_true, y_pred, average=None, zero_division=0)
        per_class_r = _safe(recall_score,    y_true, y_pred, average=None, zero_division=0)
        per_class_f = _safe(f1_score,        y_true, y_pred, average=None, zero_division=0)
        metrics["per_class_precision"] = [_round(v) for v in per_class_p] if per_class_p is not None else []
        metrics["per_class_recall"]    = [_round(v) for v in per_class_r] if per_class_r is not None else []
        metrics["per_class_f1"]        = [_round(v) for v in per_class_f] if per_class_f is not None else []

        # Specificity (binary only: TN / (TN + FP))
        if is_binary and cm.shape == (2, 2):
            tn, fp, fn_val, tp = cm.ravel()
            metrics["specificity"] = _round(tn / (tn + fp + 1e-12))
        else:
            metrics["specificity"] = None
    except Exception:
        metrics["confusion_matrix"]    = []
        metrics["classes"]             = []
        metrics["per_class_precision"] = []
        metrics["per_class_recall"]    = []
        metrics["per_class_f1"]        = []
        metrics["specificity"]         = None

    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Compute the full regression metrics suite.

    Returns a flat dict of all metrics, safe against any computation failure.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    metrics: Dict[str, Any] = {}

    metrics["r2"]              = _round(_safe(r2_score,              y_true_arr, y_pred_arr))
    metrics["mae"]             = _round(_safe(mean_absolute_error,   y_true_arr, y_pred_arr))
    metrics["mse"]             = _round(_safe(mean_squared_error,    y_true_arr, y_pred_arr))
    metrics["rmse"]            = _round(_safe(lambda a, b: math.sqrt(mean_squared_error(a, b)), y_true_arr, y_pred_arr))
    metrics["median_abs_error"]= _round(_safe(median_absolute_error, y_true_arr, y_pred_arr))
    metrics["explained_var"]   = _round(_safe(explained_variance_score, y_true_arr, y_pred_arr))
    metrics["mape"]            = _round(_safe(mean_absolute_percentage_error, y_true_arr, y_pred_arr))

    # Adjusted R²
    try:
        n = len(y_true_arr)
        p = 1  # placeholder; caller can pass n_features to override
        r2_val = metrics.get("r2")
        if r2_val is not None and n > p + 1:
            metrics["adj_r2"] = _round(1 - (1 - r2_val) * (n - 1) / (n - p - 1))
        else:
            metrics["adj_r2"] = None
    except Exception:
        metrics["adj_r2"] = None

    # SMAPE
    try:
        denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
        smape = np.mean(np.where(denom == 0, 0.0, np.abs(y_true_arr - y_pred_arr) / denom)) * 100
        metrics["smape"] = _round(smape)
    except Exception:
        metrics["smape"] = None

    # RMSLE (requires non-negative values)
    try:
        if np.all(y_true_arr >= 0) and np.all(y_pred_arr >= 0):
            rmsle = math.sqrt(mean_squared_error(
                np.log1p(y_true_arr), np.log1p(np.clip(y_pred_arr, 0, None))
            ))
            metrics["rmsle"] = _round(rmsle)
        else:
            metrics["rmsle"] = None
    except Exception:
        metrics["rmsle"] = None

    # Residuals summary (for charts)
    try:
        residuals = (y_true_arr - y_pred_arr).tolist()
        metrics["residuals_mean"]   = _round(float(np.mean(residuals)))
        metrics["residuals_std"]    = _round(float(np.std(residuals)))
        # Histogram of residuals (20 bins max)
        counts, bin_edges = np.histogram(residuals, bins=min(20, max(5, len(residuals) // 10)))
        metrics["residuals_hist_counts"] = counts.tolist()
        metrics["residuals_hist_edges"]  = [_round(e, 4) for e in bin_edges.tolist()]
    except Exception:
        metrics["residuals_mean"]         = None
        metrics["residuals_std"]          = None
        metrics["residuals_hist_counts"]  = []
        metrics["residuals_hist_edges"]   = []

    # Actual vs Predicted (sample up to 200 points for charting)
    try:
        n = len(y_true_arr)
        idx = np.linspace(0, n - 1, min(200, n), dtype=int)
        metrics["actual_vs_predicted"] = {
            "actual":    [_round(v, 4) for v in y_true_arr[idx].tolist()],
            "predicted": [_round(v, 4) for v in y_pred_arr[idx].tolist()],
        }
    except Exception:
        metrics["actual_vs_predicted"] = {"actual": [], "predicted": []}

    return metrics


def compute_adjusted_r2(r2: float, n_samples: int, n_features: int) -> Optional[float]:
    """Correctly compute Adjusted R² given n_samples and n_features."""
    try:
        if n_samples > n_features + 1:
            return _round(1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSOR PATCH
# ══════════════════════════════════════════════════════════════════════════════

def _patch_preprocessor(preprocessor: Any) -> Any:
    if not hasattr(preprocessor, "transformers"):
        return preprocessor

    squeeze_step = ("_squeeze", FunctionTransformer(_squeeze_column, validate=False))

    new_transformers = []
    patched_cols = []

    for name, transformer, cols in preprocessor.transformers:
        is_text_vec = False
        vec_obj = None
        if isinstance(transformer, TEXT_VEC_TYPES):
            is_text_vec = True
            vec_obj = transformer
        elif isinstance(transformer, Pipeline):
            last = transformer.steps[-1][1]
            if isinstance(last, TEXT_VEC_TYPES):
                is_text_vec = True
                vec_obj = last

        if not is_text_vec:
            new_transformers.append((name, transformer, cols))
            continue

        needs_cap = False
        if isinstance(vec_obj, TEXT_VEC_TYPES[2]):  # HashingVectorizer
            needs_cap = getattr(vec_obj, "n_features", 2 ** 18) > _MAX_TEXT_FEATURES
        else:
            mf = getattr(vec_obj, "max_features", None)
            needs_cap = mf is None or mf > _MAX_TEXT_FEATURES

        if needs_cap:
            from sklearn.feature_extraction.text import TfidfVectorizer
            replacement_vec = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=_MAX_TEXT_FEATURES,
                sublinear_tf=True,
            )
        else:
            replacement_vec = vec_obj

        if isinstance(transformer, Pipeline):
            prior_steps = transformer.steps[:-1]
            new_pipeline = Pipeline(
                [squeeze_step] + list(prior_steps) + [("_vectorizer", replacement_vec)]
            )
        else:
            new_pipeline = Pipeline([
                squeeze_step,
                ("_vectorizer", replacement_vec),
            ])

        patched_cols.append(cols)
        new_transformers.append((name, new_pipeline, cols))

    preprocessor.transformers = new_transformers
    if patched_cols:
        logger.info("Patched text vectoriser for column(s): %s", patched_cols)

    return preprocessor


def _resolve_hpo_budget(row_count: int, model_count: int) -> Tuple[int, int]:
    """Return per-model trial count and total HPO budget based on dataset size."""
    if row_count <= 1_000:
        total_budget = 60
    elif row_count <= 10_000:
        total_budget = 120
    else:
        total_budget = 180

    per_model = max(8, int(total_budget / max(1, model_count)))
    per_model = min(40, per_model)
    return per_model, total_budget


def _resolve_top_k(model_count: int, row_count: int) -> int:
    """Select an ensemble size that scales with dataset size and model count."""
    if model_count <= 1:
        return model_count
    if row_count < 1_000:
        return min(3, model_count)
    if row_count < 10_000:
        return min(4, model_count)
    return min(5, model_count)


def _score_key(value: Optional[float]) -> float:
    if value is None or not np.isfinite(value):
        return float("-inf")
    return float(value)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _train_single_model(
    model_name: str,
    scaling_required: bool,
    task: str,
    preprocessor: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: List[str],
    X_train_transformed: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    model = _build_model_instance(model_name, task)
    if model is None:
        return None

    try:
        start = time.time()

        label_encoder: Optional[LabelEncoder] = None
        y_train_fit = y_train
        if task == "classification" and model_name in _NEEDS_LABEL_ENCODING:
            label_encoder = LabelEncoder()
            y_train_fit = pd.Series(
                label_encoder.fit_transform(y_train),
                index=y_train.index,
            )

        if task == "classification":
            scoring = "accuracy"
            metric_name = "accuracy"
        else:
            scoring = "r2"
            metric_name = "r2_score"

        if X_train_transformed is not None:
            cv_steps: List[Tuple[str, Any]] = []
            if model_name in _NEEDS_DENSE_INPUT:
                cv_steps.append(("to_dense", FunctionTransformer(_sparse_to_dense)))
            if scaling_required:
                cv_steps.append(("scaler", StandardScaler()))
            cv_steps.append(("model", model))
            cv_pipeline = Pipeline(cv_steps)

            cv_scores = cross_val_score(
                cv_pipeline, X_train_transformed, y_train_fit, cv=5, scoring=scoring,
            )
        else:
            cv_steps = [("preprocessor", preprocessor)]
            if model_name in _NEEDS_DENSE_INPUT:
                cv_steps.append(("to_dense", FunctionTransformer(_sparse_to_dense)))
            if scaling_required:
                cv_steps.append(("scaler", StandardScaler()))
            cv_steps.append(("model", model))
            cv_pipeline = Pipeline(cv_steps)

            cv_scores = cross_val_score(
                cv_pipeline, X_train[selected_features], y_train_fit, cv=5, scoring=scoring,
            )

        mean_cv_score = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        elapsed = time.time() - start

        logger.info(
            "  %-35s  cv_%s=%.4f (+-%.4f)  time=%.2fs",
            model_name, metric_name, mean_cv_score, cv_std, elapsed,
        )

        inner_steps: List[Tuple[str, Any]] = []
        if model_name in _NEEDS_DENSE_INPUT:
            inner_steps.append(("to_dense", FunctionTransformer(_sparse_to_dense)))
        if scaling_required:
            inner_steps.append(("scaler", StandardScaler()))
        inner_steps.append(("model", _build_model_instance(model_name, task)))
        inner_pipeline = Pipeline(inner_steps)

        return {
            "model_name": model_name,
            "scaling_required": scaling_required,
            "inner_pipeline": inner_pipeline,
            "metric_name": metric_name,
            "score": mean_cv_score,
            "cv_score": mean_cv_score,
            "cv_std": cv_std,
            "cv_scores": [round(float(s), 6) for s in cv_scores.tolist()],
            "train_time_seconds": round(elapsed, 3),
            "label_encoder": label_encoder,
        }

    except Exception as exc:
        logger.warning("  %-35s  FAILED: %s", model_name, exc)
        logger.debug(traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_model_selection_supabase(
    user_id: str,
    session_id: str,
    dataset_base: str,
) -> Dict[str, Any]:
    """End-to-end model selection + training pipeline with full metrics."""
    logger.info("=" * 64)
    logger.info("MODEL SELECTION PIPELINE START")
    logger.info("  user_id=%s  session_id=%s  dataset_base=%s", user_id, session_id, dataset_base)
    logger.info("=" * 64)

    pipeline_start = time.time()

    # STEP 1: Load metadata from Supabase
    base_path = f"meta_data/{user_id}/{session_id}"
    profiling = download_json(f"{base_path}/{dataset_base}_profiling.json")
    features_meta = download_json(f"{base_path}/{dataset_base}_features.json")

    # STEP 2: Extract core info
    row_count = profiling.get("number_of_rows", 0)
    task = str(features_meta.get("task", "classification")).strip().lower()
    target = str(features_meta.get("target", "")).strip()
    selected_features = list(features_meta.get("selected_features", []))

    if not target:
        raise ValueError("Target column not specified in features metadata.")
    if not selected_features:
        raise ValueError("No selected features found in features metadata.")

    # STEP 3: Detect feature type & build processing plan
    feature_types = detect_feature_type(selected_features, profiling)
    feature_processing_plan = build_feature_processing_plan(selected_features, profiling, row_count)

    # STEP 4: Select candidate models
    if task == "classification":
        candidates = select_classification_models(row_count, feature_types)
    else:
        candidates = select_regression_models(row_count, feature_types)

    available_candidates = [
        (mn, sf) for mn, sf in candidates
        if _build_model_instance(mn, task) is not None
    ]
    if not available_candidates:
        raise RuntimeError("No candidate models available.")

    # STEP 5: Load cleaned data
    csv_bytes = download_file(f"output/{user_id}/{session_id}/{dataset_base}_cleaned.csv")
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df = enforce_dtypes(df, profiling)

    # STEP 6: Train / validation split
    X_train, X_val, y_train, y_val = split_dataset(
        df=df, target=target, selected_features=selected_features,
        task=task, test_size=0.2, random_state=42,
    )

    training_stats = build_training_stats(
        X_train=X_train, selected_features=selected_features, profiling_meta=profiling,
    )

    # STEP 7: Build & patch preprocessor (fit=False → patch → fit)
    preprocessor, drop_cols = build_fitted_preprocessor(
        X_train=X_train, selected_features=selected_features,
        profiling_meta=profiling, row_count=row_count,
        y_train=y_train, fit=False,
    )
    preprocessor = _patch_preprocessor(preprocessor)
    X_fit = X_train[selected_features]
    if y_train is not None:
        preprocessor.fit(X_fit, y_train)
    else:
        preprocessor.fit(X_fit)

    # Transform training data once for HPO
    X_train_transformed_hpo = preprocessor.transform(X_train[selected_features])

    # STEP 8: HPO for each candidate model
    hpo_cv = 3
    per_model_trials, hpo_total_budget = _resolve_hpo_budget(row_count, len(available_candidates))
    hpo_models: List[Dict[str, Any]] = []
    tuned_results: List[Dict[str, Any]] = []
    total_hpo_time = 0.0

    label_enc: Optional[LabelEncoder] = None
    y_train_encoded: Optional[pd.Series] = None
    if task == "classification":
        needs_any_label = any(mn in _NEEDS_LABEL_ENCODING for mn, _ in available_candidates)
        if needs_any_label:
            label_enc = LabelEncoder()
            y_train_encoded = pd.Series(
                label_enc.fit_transform(y_train),
                index=y_train.index,
            )

    for model_name, scaling_flag in available_candidates:
        base_model = _build_model_instance(model_name, task)
        if base_model is None:
            continue

        y_train_hpo = y_train
        needs_label = task == "classification" and model_name in _NEEDS_LABEL_ENCODING
        if needs_label and y_train_encoded is not None:
            y_train_hpo = y_train_encoded

        X_hpo = X_train_transformed_hpo
        if model_name in _NEEDS_DENSE_INPUT or scaling_flag:
            X_hpo = _sparse_to_dense(X_hpo)
        if scaling_flag:
            X_hpo = StandardScaler().fit_transform(X_hpo)

        hpo_start = time.time()
        hpo_result = run_hpo(
            model_name=model_name,
            model_instance=base_model,
            X_train=X_hpo,
            y_train=y_train_hpo,
            task=task,
            n_trials=per_model_trials,
            cv=hpo_cv,
        )
        hpo_elapsed = round(time.time() - hpo_start, 3)
        total_hpo_time += hpo_elapsed

        hpo_status = str(hpo_result.get("status") or "completed").lower()
        best_score = hpo_result.get("best_score")
        best_score_value = None
        if best_score is not None and np.isfinite(best_score):
            best_score_value = float(best_score)
        best_params = hpo_result.get("best_params") or {}
        trials_run = hpo_result.get("n_trials", per_model_trials)

        tuned_model = _build_model_instance(model_name, task)
        if tuned_model is not None and best_params:
            tuned_model.set_params(**best_params)

        inner_steps: List[Tuple[str, Any]] = []
        if model_name in _NEEDS_DENSE_INPUT:
            inner_steps.append(("to_dense", FunctionTransformer(_sparse_to_dense)))
        if scaling_flag:
            inner_steps.append(("scaler", StandardScaler()))
        inner_steps.append(("model", tuned_model if tuned_model is not None else base_model))
        inner_pipeline = Pipeline(inner_steps)

        tuned_results.append({
            "model_name": model_name,
            "scaling_required": scaling_flag,
            "inner_pipeline": inner_pipeline,
            "cv_score": best_score_value,
            "cv_std": None,
            "cv_scores": [],
            "train_time_seconds": hpo_elapsed,
            "hpo_status": hpo_status,
            "hpo_best_params": best_params,
            "hpo_best_score": best_score_value,
            "hpo_trials": trials_run,
            "hpo_time_seconds": hpo_elapsed,
            "needs_label_encoding": needs_label,
        })

        hpo_models.append({
            "model_name": model_name,
            "status": hpo_status,
            "best_score": best_score_value,
            "best_params": best_params,
            "n_trials": trials_run,
            "time_seconds": hpo_elapsed,
        })

    if not tuned_results:
        raise RuntimeError("All candidate models failed during HPO.")

    # STEP 9: Rank tuned models, select top K
    ranked = sorted(tuned_results, key=lambda r: _score_key(r.get("cv_score")), reverse=True)
    top_k = _resolve_top_k(len(ranked), row_count)
    top_models = ranked[:top_k]
    best = top_models[0]

    # STEP 10: Build ensemble (or single best model)
    ensemble_info: Dict[str, Any] = {
        "enabled": len(top_models) > 1,
        "strategy": "voting" if len(top_models) > 1 else "single",
        "top_k": top_k,
        "models": [m["model_name"] for m in top_models],
    }

    if len(top_models) > 1:
        estimators: List[Tuple[str, Any]] = []
        supports_soft = True
        for idx, entry in enumerate(top_models, start=1):
            est_name = f"{entry['model_name']}_{idx}"
            estimator = entry["inner_pipeline"]
            if task == "classification" and not hasattr(estimator, "predict_proba"):
                supports_soft = False
            estimators.append((est_name, estimator))

        if task == "classification":
            voting = "soft" if supports_soft else "hard"
            ensemble_model = VotingClassifier(estimators=estimators, voting=voting)
            ensemble_info["voting"] = voting
        else:
            ensemble_model = VotingRegressor(estimators=estimators)
            ensemble_info["voting"] = "mean"

        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("ensemble", ensemble_model),
        ])
    else:
        best_inner = best["inner_pipeline"]
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            *best_inner.steps,
        ])

    # STEP 11: Refit final pipeline on full training data
    final_label_encoder = None
    if task == "classification" and any(m.get("needs_label_encoding") for m in top_models):
        final_label_encoder = label_enc or LabelEncoder()
        y_train_final = pd.Series(
            final_label_encoder.fit_transform(y_train),
            index=y_train.index,
        )
    else:
        y_train_final = y_train

    final_pipeline.fit(X_train[selected_features], y_train_final)

    fitted_model = final_pipeline.named_steps.get("model") or final_pipeline.named_steps.get("ensemble")
    if fitted_model is not None:
        for attr in ("estimators_", "feature_importances_", "coef_", "support_vectors_"):
            if hasattr(fitted_model, attr):
                logger.info("Fitted model exposes attribute: %s", attr)

    # STEP 12: Evaluate on validation set + compute FULL metrics
    y_val_pred = final_pipeline.predict(X_val[selected_features])

    # Get probabilities for classification (if supported)
    y_val_prob = None
    if task == "classification":
        try:
            y_val_prob = final_pipeline.predict_proba(X_val[selected_features])
        except Exception:
            y_val_prob = None

    if task == "classification":
        y_val_for_score = y_val
        if final_label_encoder is not None:
            y_val_for_score = pd.Series(final_label_encoder.transform(y_val), index=y_val.index)
        final_validation_score = float(accuracy_score(y_val_for_score, y_val_pred))

        # Compute full classification metrics
        full_metrics = compute_classification_metrics(
            y_true=y_val_for_score,
            y_pred=y_val_pred,
            y_prob=y_val_prob,
            label_encoder=final_label_encoder,
        )
    else:
        final_validation_score = float(r2_score(y_val, y_val_pred))

        # Compute full regression metrics
        full_metrics = compute_regression_metrics(y_true=y_val, y_pred=y_val_pred)
        # Patch adjusted R² with correct n_features
        r2_val = full_metrics.get("r2")
        if r2_val is not None:
            full_metrics["adj_r2"] = compute_adjusted_r2(r2_val, len(y_val), len(selected_features))

    logger.info("Full metrics computed: %s", list(full_metrics.keys()))

    # Smoke-test
    smoke_preds = final_pipeline.predict(X_val.head(2)[selected_features])
    logger.info("Smoke-test PASSED: %s", smoke_preds)

    # ── FIX: Build type_map from profiling and embed it in the pkl. ─────────────
    # ROOT CAUSE: model_testing._coerce_input_df reads from type_map to decide
    # how to cast each input column before prediction. Previously, type_map was
    # rebuilt at inference time by downloading profiling JSON from Supabase. If
    # that download failed (network error, path mismatch, session rotation) the
    # type_map came back empty → NO coercion → OHE received int 0 instead of
    # str "0" → all-zeros one-hot → wrong / flipped prediction.
    #
    # Embedding the type_map directly in the pkl eliminates the entire failure
    # class: coercion now works identically offline, in tests, and in production
    # without any Supabase dependency at inference time.
    #
    # The type_map is built from the same profiling JSON used to build and fit
    # the preprocessor, so it is guaranteed to be consistent with the encoder.
    _col_summary_map: Dict[str, Dict] = {
        item["column_name"]: item
        for item in profiling.get("column_wise_summary", [])
        if item.get("column_name")
    }
    embedded_type_map: Dict[str, str] = {
        feat: str(_col_summary_map.get(feat, {}).get("structural_type", "")).lower()
        for feat in selected_features
    }
    logger.info(
        "Embedded type_map into pkl | %d/%d features typed | map=%s",
        sum(1 for v in embedded_type_map.values() if v),
        len(selected_features),
        embedded_type_map,
    )

    # STEP 13: Serialize & upload pkl
    status_counts = {"completed": 0, "skipped": 0, "failed": 0}
    for r in tuned_results:
        status = (r.get("hpo_status") or "failed").lower()
        status_counts[status] = status_counts.get(status, 0) + 1

    best_score = best.get("cv_score")
    if status_counts.get("completed"):
        hpo_status = "completed"
    elif status_counts.get("skipped"):
        hpo_status = "skipped"
    else:
        hpo_status = "failed"

    hpo_summary = {
        "status": hpo_status,
        "total_models": len(available_candidates),
        "tuned_models": len(tuned_results),
        "failed_models": status_counts.get("failed", 0),
        "skipped_models": status_counts.get("skipped", 0),
        "per_model_trials": per_model_trials,
        "total_budget": hpo_total_budget,
        "cv": hpo_cv,
        "time_seconds": round(total_hpo_time, 3),
        "best_model": best["model_name"],
        "best_score": round(float(best_score), 4) if best_score is not None else None,
        "top_k": top_k,
    }

    pkl_payload = {
        "framework": "sklearn",
        "pipeline": final_pipeline,
        "label_encoder": final_label_encoder,
        "selected_features": selected_features,   # authoritative column order
        "training_stats": training_stats,
        "profiling_meta": profiling,
        # ── FIX: embedded type_map — primary source for _coerce_input_df ──────
        "type_map": embedded_type_map,
        "hpo": hpo_summary,
        "hpo_models": hpo_models,
        "ensemble": ensemble_info,
        "top_k_models": [m["model_name"] for m in top_models],
    }
    pkl_bytes = pickle.dumps(pkl_payload)
    pkl_path = f"output/{user_id}/{session_id}/{dataset_base}_model.pkl"
    upload_file(pkl_path, pkl_bytes, content_type="application/octet-stream")

    # STEP 14: Build and upload model report with full metrics
    pipeline_elapsed = time.time() - pipeline_start

    leaderboard = []
    for r in ranked:
        cv_score = r.get("cv_score")
        entry = {
            "model_name": r["model_name"],
            "scaling_required": r["scaling_required"],
            "cv_score": round(float(cv_score), 4) if cv_score is not None else None,
            "cv_std": r.get("cv_std"),
            "cv_scores": r.get("cv_scores", []),
            "train_time_seconds": r.get("train_time_seconds"),
            "hpo_status": r.get("hpo_status"),
            "hpo_best_score": r.get("hpo_best_score"),
            "hpo_best_params": r.get("hpo_best_params", {}),
            "hpo_trials": r.get("hpo_trials"),
            "hpo_time_seconds": r.get("hpo_time_seconds"),
        }
        leaderboard.append(entry)

    metric_name = "accuracy" if task == "classification" else "r2_score"
    report: Dict[str, Any] = {
        "status": "completed",
        "framework": "sklearn",
        "dataset_base": dataset_base,
        "task": task,
        "target": target,
        "row_count": row_count,
        "feature_count": len(selected_features),
        "selected_features": selected_features,
        "feature_types": feature_types,
        "feature_processing": feature_processing_plan,
        "best_model": best["model_name"],
        "best_score": round(float(best_score), 4) if best_score is not None else None,
        "cv_score": round(float(best_score), 4) if best_score is not None else None,
        "cv_std": None,
        "final_validation_score": round(final_validation_score, 4),
        "metric": metric_name,
        "leaderboard": leaderboard,
        "models_trained": len(tuned_results),
        "models_failed": len(available_candidates) - len(tuned_results),
        "total_pipeline_time_seconds": round(pipeline_elapsed, 2),
        "dropped_columns": drop_cols,
        "baseline_cv_score": round(float(best_score), 4) if best_score is not None else None,
        "baseline_cv_std": None,
        "hpo_status": hpo_summary["status"],
        "hpo_model": best["model_name"],
        "hpo_best_score": round(float(best_score), 4) if best_score is not None else None,
        "hpo_best_params": best.get("hpo_best_params", {}),
        "hpo_trials": per_model_trials,
        "hpo_cv": hpo_cv,
        "hpo_time_seconds": round(total_hpo_time, 3),
        "hpo": hpo_summary,
        "hpo_models": hpo_models,
        "ensemble": ensemble_info,
        "final_model": "ensemble" if ensemble_info.get("enabled") else "single",
        "top_k_models": [m["model_name"] for m in top_models],
        # ── FULL METRICS ──────────────────────────────────────────────────────
        "evaluation_metrics": full_metrics,
    }

    # Also hoist top-level convenience keys for backward compatibility
    if task == "classification":
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "roc_auc", "pr_auc", "log_loss", "mcc", "cohen_kappa",
                    "balanced_accuracy", "f1_weighted", "specificity",
                    "hamming_loss", "jaccard_macro"):
            report[key] = full_metrics.get(key)
    else:
        for key in ("r2", "mae", "mse", "rmse", "rmsle", "mape", "smape",
                    "median_abs_error", "explained_var", "adj_r2"):
            report[key] = full_metrics.get(key)

    report_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_model_report.json"
    upload_json(report_path, report)
    logger.info("Uploaded model report with full metrics: %s", report_path)
    logger.info("MODEL SELECTION PIPELINE COMPLETE - %.2fs", pipeline_elapsed)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uid = input("user_id: ").strip()
    sid = input("session_id: ").strip()
    dsn = input("dataset_name (without suffix): ").strip()

    if not uid or not sid or not dsn:
        print("Error: all three inputs are required.")
    else:
        output = run_model_selection_supabase(uid, sid, dsn)
        print("\n" + "=" * 64)
        print("MODEL REPORT")
        print("=" * 64)
        for k, v in output.items():
            if k in ("leaderboard", "evaluation_metrics", "confusion_matrix"):
                print(f"\n{k}: {v}")
            elif k == "feature_processing":
                print(f"\n{k}:")
                for feat, enc in v.items():
                    print(f"  {feat}: {enc}")
            else:
                print(f"{k}: {v}")