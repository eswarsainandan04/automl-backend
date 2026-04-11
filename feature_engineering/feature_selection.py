"""
Feature Engineering + Feature Selection CLI

Behavior:
1) Ask for userid and load all *_cleaned.csv files under storage/output/{userid}
2) Route processing by dataset shape:
   - Single cleaned file: ask target, run feature selection
   - Multiple files with same columns: concatenate and use filename as target
   - Multiple files with common identifier columns: merge by common IDs (inner join), ask target
   - Multiple files with different columns/no common IDs: process each separately, ask target per file
3) Drop identifier-like columns before feature selection
4) Select features using adaptive model + MI scoring (type-aware)
5) Save *_features.json files under storage/meta_data/{userid}
"""

from __future__ import annotations

import json
import logging
import re
import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from data_preprocessing.supabase_storage import download_file, download_json, list_files, upload_json
from data_preprocessing.column_handler import (
    detect_semantic,
    get_sample_values,
    infer_normalized_dtype,
    load_patterns,
)
from data_preprocessing.structural_type_detector import StructuralTypeDetector


# Paths
def _resolve_backend_root() -> Path:
    """Resolve backend root so storage paths work from any script location."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "storage").exists():
            return candidate
    return here.parent.parent


BASE_DIR = _resolve_backend_root()
STORAGE_DIR = BASE_DIR / "storage"
OUTPUT_DIR = STORAGE_DIR / "output"
META_DIR = STORAGE_DIR / "meta_data"


# Logging
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("FeatureEngineeringCLI")


ID_PATTERN_SAMPLE_MAX = 5000
TYPE_INFER_SAMPLE_MAX = 5000
HYBRID_TOP_K = 10
MI_FIXED_THRESHOLD = 0.01
SMALL_DATASET_MAX_ROWS = 500


def _prompt_userid() -> str:
    while True:
        userid = input("Enter userid: ").strip()
        if userid:
            return userid
        print("Userid cannot be empty.")


def _load_cleaned_csvs(userid: str) -> Dict[str, pd.DataFrame]:
    user_dir = OUTPUT_DIR / userid
    if not user_dir.exists():
        raise FileNotFoundError(f"User folder not found: {user_dir}")

    csv_paths = sorted(user_dir.glob("*_cleaned.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_cleaned.csv files found in: {user_dir}")

    loaded: Dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        df = pd.read_csv(path)
        loaded[path.name] = df
        logger.info("Loaded %s (%s rows x %s cols)", path.name, len(df), len(df.columns))

    return loaded


def _prompt_target_column(columns: List[str], label: str) -> str:
    print(f"\nAvailable columns for {label}:")
    print(", ".join(columns))

    while True:
        target = input("Enter target column: ").strip()
        if target in columns:
            return target
        print(f"Column '{target}' is not present. Please try again.")


def _profiling_identifier_columns(userid: str, dataset_base_name: str) -> List[str]:
    """Return identifier columns from profiling metadata for a dataset."""
    profiling_path = META_DIR / userid / f"{dataset_base_name}_profiling.json"
    if not profiling_path.exists():
        return []

    try:
        with open(profiling_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read profiling metadata %s: %s", profiling_path, exc)
        return []

    ids: List[str] = []
    for item in profile.get("column_wise_summary", []):
        stype = str(item.get("structural_type", "")).strip().lower()
        if stype in {"identifier", "identifer"}:
            col = item.get("column_name")
            if isinstance(col, str) and col:
                ids.append(col)

    return list(dict.fromkeys(ids))


def _drop_profiling_identifiers(df: pd.DataFrame, userid: str, dataset_base_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """Drop identifier columns inferred by profiling metadata before target prompt."""
    prof_ids = _profiling_identifier_columns(userid, dataset_base_name)
    to_drop = [c for c in prof_ids if c in df.columns]
    if not to_drop:
        return df, []

    logger.info("Dropped profiling identifier columns before target prompt: %s", to_drop)
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def _drop_profiling_identifiers_from_payload(df: pd.DataFrame, profiling_payload: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Drop identifier columns inferred by a profiling payload dict."""
    ids: List[str] = []
    for item in profiling_payload.get("column_wise_summary", []):
        stype = str(item.get("structural_type", "")).strip().lower()
        if stype in {"identifier", "identifer"}:
            col = item.get("column_name")
            if isinstance(col, str) and col:
                ids.append(col)

    to_drop = [c for c in list(dict.fromkeys(ids)) if c in df.columns]
    if not to_drop:
        return df, []

    return df.drop(columns=to_drop, errors="ignore"), to_drop


def _to_datetime_silent(series: pd.Series) -> pd.Series:
    """Parse datetimes while silencing pandas format-inference fallback warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Could not infer format, so each element will be parsed individually.*",
            category=UserWarning,
        )
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            # Backward compatibility for pandas versions without format='mixed'.
            return pd.to_datetime(series, errors="coerce")


def _datetime_parse_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    parsed = _to_datetime_silent(series.astype(str))
    return float(parsed.notna().mean())


def _encode_feature_for_mi(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Encode a feature into numeric form for mutual information scoring.

    Returns encoded series and whether the feature should be treated as discrete.
    """
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        if num.isna().all():
            num = num.fillna(0.0)
        else:
            num = num.fillna(float(num.median()))
        return num.astype(float), False

    non_null = series.dropna()
    if len(non_null) > TYPE_INFER_SAMPLE_MAX:
        non_null = non_null.sample(n=TYPE_INFER_SAMPLE_MAX, random_state=42)

    parse_rate = 0.0
    if len(non_null):
        parse_rate = _datetime_parse_rate(non_null)

    if parse_rate >= 0.8:
        dt = _to_datetime_silent(series)
        dt_numeric = pd.Series(dt.view("int64"), index=series.index).replace(-9223372036854775808, np.nan)
        if dt_numeric.isna().all():
            dt_numeric = dt_numeric.fillna(0.0)
        else:
            dt_numeric = dt_numeric.fillna(float(dt_numeric.median()))
        return dt_numeric.astype(float), False

    cat = series.fillna("__MISSING__").astype(str)
    codes, _ = pd.factorize(cat, sort=True)
    return pd.Series(codes, index=series.index).astype(float), True


def _recommend_features_by_mi(df: pd.DataFrame, target: str, task: str) -> List[Tuple[str, float]]:
    """Rank candidate features using mutual information and return top recommendations."""
    features = [c for c in df.columns if c != target]
    if not features:
        return []

    mi_df = df[features + [target]].copy()
    mi_df = mi_df.loc[~mi_df[target].isna()].copy()
    if mi_df.empty:
        return [(f, 0.0) for f in features]

    encoded = pd.DataFrame(index=mi_df.index)
    discrete_flags: List[bool] = []

    for feat in features:
        enc, is_discrete = _encode_feature_for_mi(mi_df[feat])
        encoded[feat] = enc
        discrete_flags.append(is_discrete)

    X = encoded.to_numpy(dtype=float)

    try:
        if task == "classification":
            y = mi_df[target]
            if pd.api.types.is_numeric_dtype(y):
                y_enc = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
            else:
                y_enc, _ = pd.factorize(y.astype(str), sort=True)
            scores = mutual_info_classif(X, y_enc, discrete_features=discrete_flags, random_state=42)
        else:
            y_num = pd.to_numeric(mi_df[target], errors="coerce")
            valid_mask = ~y_num.isna()
            if not valid_mask.any():
                scores = np.zeros(len(features), dtype=float)
            else:
                X_valid = X[valid_mask.values]
                y_valid = y_num.loc[valid_mask].to_numpy(dtype=float)
                scores = mutual_info_regression(X_valid, y_valid, discrete_features=discrete_flags, random_state=42)
    except Exception as exc:
        logger.warning("Mutual information scoring failed, falling back to zero scores: %s", exc)
        scores = np.zeros(len(features), dtype=float)

    ranking = sorted(
        [(feat, float(score)) for feat, score in zip(features, scores)],
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranking:
        return []

    # Fixed-threshold selection strategy:
    # keep features if score >= 0.01 after top-k ranking.
    top_k = ranking[: min(HYBRID_TOP_K, len(ranking))]
    selected = [(f, s) for f, s in top_k if s >= MI_FIXED_THRESHOLD]

    if not selected:
        selected = top_k[:1]

    return selected


def _score_all_features_by_mi(df: pd.DataFrame, target: str, task: str) -> List[Tuple[str, float]]:
    """Return MI scores for all candidate features sorted descending.

    This is used by session APIs where UI must display full manual selection
    candidates with importance tags, excluding only identifier columns.
    """
    features = [c for c in df.columns if c != target]
    if not features:
        return []

    mi_df = df[features + [target]].copy()
    mi_df = mi_df.loc[~mi_df[target].isna()].copy()
    if mi_df.empty:
        return [(f, 0.0) for f in features]

    encoded = pd.DataFrame(index=mi_df.index)
    discrete_flags: List[bool] = []

    for feat in features:
        enc, is_discrete = _encode_feature_for_mi(mi_df[feat])
        encoded[feat] = enc
        discrete_flags.append(is_discrete)

    X = encoded.to_numpy(dtype=float)

    try:
        if task == "classification":
            y = mi_df[target]
            if pd.api.types.is_numeric_dtype(y):
                y_enc = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
            else:
                y_enc, _ = pd.factorize(y.astype(str), sort=True)
            scores = mutual_info_classif(X, y_enc, discrete_features=discrete_flags, random_state=42)
        else:
            y_num = pd.to_numeric(mi_df[target], errors="coerce")
            valid_mask = ~y_num.isna()
            if not valid_mask.any():
                scores = np.zeros(len(features), dtype=float)
            else:
                X_valid = X[valid_mask.values]
                y_valid = y_num.loc[valid_mask].to_numpy(dtype=float)
                scores = mutual_info_regression(X_valid, y_valid, discrete_features=discrete_flags, random_state=42)
    except Exception as exc:
        logger.warning("Mutual information scoring failed, falling back to zero scores: %s", exc)
        scores = np.zeros(len(features), dtype=float)

    ranking = sorted(
        [(feat, float(score)) for feat, score in zip(features, scores)],
        key=lambda x: x[1],
        reverse=True,
    )

    return ranking


def _detect_task(df: pd.DataFrame, target: str) -> str:
    """Infer ML task type from the target column distribution."""
    series = df[target].dropna()
    if series.empty:
        return "classification"
    unique = series.nunique()
    if unique <= 20:
        return "classification"
    if pd.api.types.is_numeric_dtype(series):
        return "regression"
    return "classification"


def _infer_feature_type_from_profiling(feat: str, series: pd.Series, profiling_payload: Dict) -> str:
    """Infer a feature's type from profiling payload or fall back to series dtype."""
    col_summaries = profiling_payload.get("column_wise_summary", [])
    for entry in col_summaries:
        if entry.get("column_name") == feat:
            stype = str(entry.get("structural_type", "")).strip().lower()
            if stype in {"numeric", "categorical", "boolean", "datetime", "text"}:
                return stype
            semantic = str(entry.get("semantic_type", "")).strip().lower()
            if semantic in {"integer", "float"}:
                return "numeric"
            if semantic in {"boolean"}:
                return "categorical"
            if semantic in {"datetime", "date", "time"}:
                return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _build_profiling_card_for_feature(
    feat: str,
    profiling_payload: Dict,
    feature_importance: Dict[str, float],
    feature_importance_normalized: Dict[str, float],
) -> Dict[str, Any]:
    """Build a rich display card for a feature from profiling metadata."""
    col_summaries = profiling_payload.get("column_wise_summary", [])
    col_meta: Dict[str, Any] = {}
    for entry in col_summaries:
        if entry.get("column_name") == feat:
            col_meta = entry
            break

    return {
        "feature": feat,
        "structural_type": col_meta.get("structural_type", "unknown"),
        "semantic_type": col_meta.get("semantic_type", "unknown"),
        "inferred_dtype": col_meta.get("inferred_dtype", "unknown"),
        "null_count": col_meta.get("null_count", 0),
        "null_percentage": col_meta.get("null_percentage", 0.0),
        "unique_count": col_meta.get("unique_count", None),
        "sample_values": col_meta.get("sample_values", []),
        "mi_score": feature_importance.get(feat, 0.0),
        "importance_normalized": feature_importance_normalized.get(feat, 0.0),
        "importance_tag": f"{feature_importance_normalized.get(feat, 0.0) * 100:.1f}%",
    }


def _build_profiling_card_for_target(
    target: str,
    profiling_payload: Dict,
) -> Dict[str, Any]:
    """Build a display card for the target column from profiling metadata."""
    col_summaries = profiling_payload.get("column_wise_summary", [])
    col_meta: Dict[str, Any] = {}
    for entry in col_summaries:
        if entry.get("column_name") == target:
            col_meta = entry
            break

    return {
        "column": target,
        "structural_type": col_meta.get("structural_type", "unknown"),
        "semantic_type": col_meta.get("semantic_type", "unknown"),
        "inferred_dtype": col_meta.get("inferred_dtype", "unknown"),
        "null_count": col_meta.get("null_count", 0),
        "null_percentage": col_meta.get("null_percentage", 0.0),
        "unique_count": col_meta.get("unique_count", None),
        "sample_values": col_meta.get("sample_values", []),
    }


def _all_same_columns(datasets: Dict[str, pd.DataFrame]) -> bool:
    """Return True if all DataFrames share the exact same column set."""
    col_sets = [frozenset(df.columns) for df in datasets.values()]
    return len(set(col_sets)) == 1


def _concatenate_with_filename_target(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames row-wise and add a 'filename' column as target."""
    parts = []
    for fname, df in datasets.items():
        base = Path(fname).stem.replace("_cleaned", "")
        tmp = df.copy()
        tmp["filename"] = base
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def _common_identifier_columns(datasets: Dict[str, pd.DataFrame]) -> List[str]:
    """Return column names present in ALL datasets that look like identifiers."""
    if not datasets:
        return []
    dfs = list(datasets.values())
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)

    id_re = re.compile(
        r"(^id$|_id$|^no$|_no$|^num$|_num$|^number$|_number$"
        r"|^key$|_key$|^code$|_code$|^pk$|_pk$|^idx$|_idx$)",
        re.IGNORECASE,
    )
    return [c for c in common if id_re.search(c)]


def _merge_on_common_ids(datasets: Dict[str, pd.DataFrame], common_ids: List[str]) -> pd.DataFrame:
    """Inner-join all DataFrames on the shared identifier columns."""
    dfs = list(datasets.values())
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=common_ids, how="inner", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols, errors="ignore")
    return merged


def _safe_default_target_value(target_type: str) -> Any:
    if target_type == "regression":
        return 0.0
    if target_type == "multiclass_classification":
        return "class_0"
    return 0


def _build_target_series(
    df: pd.DataFrame,
    target_type: str,
    conditions: List[Dict],
    custom_expression: Optional[str],
    default_value: Any,
) -> pd.Series:
    """Build a target column Series from rule conditions or a custom pandas expression."""
    result = pd.Series([default_value] * len(df), index=df.index)

    if custom_expression and custom_expression.strip():
        try:
            mask = df.eval(custom_expression.strip())
            if hasattr(mask, "values"):
                mask = mask.values
            result = pd.Series(
                [1 if m else default_value for m in mask],
                index=df.index,
            )
            return result
        except Exception as exc:
            raise ValueError(f"Custom expression failed to evaluate: {exc}") from exc

    for rule in (conditions or []):
        metric = str(rule.get("metric", "")).strip()
        operator = str(rule.get("operator", "=")).strip()
        raw_value = rule.get("value", "")
        raw_output = rule.get("output_value", None)

        if not metric or metric not in df.columns:
            continue

        try:
            cmp_value = float(raw_value)
        except (TypeError, ValueError):
            cmp_value = str(raw_value)

        try:
            out_value = float(raw_output) if raw_output not in (None, "") else 1
        except (TypeError, ValueError):
            out_value = str(raw_output) if raw_output not in (None, "") else 1

        col = df[metric]
        if operator == ">":
            mask = col > cmp_value
        elif operator == ">=":
            mask = col >= cmp_value
        elif operator == "<":
            mask = col < cmp_value
        elif operator == "<=":
            mask = col <= cmp_value
        elif operator == "!=":
            mask = col != cmp_value
        else:
            mask = col == cmp_value

        result = result.where(~mask, other=out_value)

    return result


def _update_profiling_for_new_column(
    profiling_payload: Dict,
    df: pd.DataFrame,
    new_column: str,
) -> Dict:
    """Append a column_wise_summary entry for a newly created target column."""
    series = df[new_column]
    null_count = int(series.isna().sum())
    total = max(len(series), 1)

    detector = StructuralTypeDetector()
    sample_vals = series.dropna().head(5).tolist()

    col_meta: Dict[str, Any] = {
        "column_name": new_column,
        "inferred_dtype": "integer" if pd.api.types.is_integer_dtype(series) else (
            "float" if pd.api.types.is_float_dtype(series) else "object"
        ),
        "null_count": null_count,
        "null_percentage": round(null_count / total * 100, 4),
        "unique_count": int(series.nunique()),
        "sample_values": sample_vals,
        "semantic_type": "boolean" if series.nunique() <= 2 else "integer",
        "semantic_confidence": 1.0,
    }
    col_meta["structural_type"] = detector.detect(col_meta, total)

    summaries = profiling_payload.get("column_wise_summary", [])
    existing_names = {e.get("column_name") for e in summaries}
    if new_column not in existing_names:
        summaries.append(col_meta)
    else:
        for i, e in enumerate(summaries):
            if e.get("column_name") == new_column:
                summaries[i] = col_meta
                break

    profiling_payload["column_wise_summary"] = summaries
    profiling_payload["number_of_columns"] = len(summaries)
    return profiling_payload


def _run_feature_selection_for_dataset(
    userid: str,
    output_name: str,
    df: pd.DataFrame,
    target: str,
    source_files: List[str],
    processing_mode: str,
    pre_dropped_identifiers: Optional[List[str]] = None,
    merged_on_ids: Optional[List[str]] = None,
) -> None:
    """Run feature selection for a prepared DataFrame and save the JSON."""
    if pre_dropped_identifiers is None:
        pre_dropped_identifiers = []

    available_features = [c for c in df.columns if c != target]
    if not available_features:
        print(f"No features available after filtering. Skipping {output_name}.")
        return

    task = _detect_task(df, target)
    print(f"\nDetected task: {task}")

    ranked = _recommend_features_by_mi(df, target=target, task=task)
    if not ranked:
        print("No informative features found by MI scoring.")
        return

    all_scores = _score_all_features_by_mi(
        df[available_features + [target]], target=target, task=task
    )
    all_score_map = {f: s for f, s in all_scores}

    selected_features = [f for f, _ in ranked]
    selected_scores = {f: float(all_score_map.get(f, 0.0)) for f in selected_features}
    max_score = max(selected_scores.values()) if selected_scores else 0.0
    normalized = (
        {f: float(s / max_score) for f, s in selected_scores.items()}
        if max_score > 0
        else {f: 0.0 for f in selected_features}
    )

    patterns = load_patterns()
    detector = StructuralTypeDetector()

    feature_types: Dict[str, str] = {}
    structural_types: Dict[str, str] = {}
    for feat in selected_features:
        series = df[feat]
        sample = get_sample_values(series)
        normalized_dtype = infer_normalized_dtype(series)
        sem_type, _ = detect_semantic(sample, normalized_dtype, patterns)
        col_meta_tmp = {
            "column_name": feat,
            "inferred_dtype": normalized_dtype,
            "unique_count": int(series.nunique()),
            "null_count": int(series.isna().sum()),
            "null_percentage": round(series.isna().mean() * 100, 4),
            "sample_values": sample,
            "semantic_type": sem_type,
            "semantic_confidence": 1.0,
        }
        stype = detector.detect(col_meta_tmp, len(df))
        feature_types[feat] = stype
        structural_types[feat] = stype

    output_path = META_DIR / userid / f"{output_name}_features.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "type": "supervised",
        "task": task,
        "target": target,
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "feature_types": feature_types,
        "structural_types": structural_types,
        "selection_method": "auto_mi_threshold_selection",
        "selector_components": [
            "target_selection",
            "identifier_drop",
            "mutual_information_ranking",
            "threshold_cutoff",
        ],
        "dropped_identifier_columns": pre_dropped_identifiers,
        "merged_on_ids": merged_on_ids or [],
        "source_files": source_files,
        "processing_mode": processing_mode,
        "feature_importance": selected_scores,
        "feature_importance_normalized": normalized,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved feature selection → {output_path}")
    print(f"  Task        : {task}")
    print(f"  Target      : {target}")
    print(f"  Features ({len(selected_features)}): {', '.join(selected_features)}")
    print(f"  Mode        : {processing_mode}")


# ─── Session-scoped API functions (called by automl_router) ──────────────────

def list_session_cleaned_datasets(user_id: str, session_id: str) -> List[Dict[str, Any]]:
    """List available cleaned datasets for a user session from Supabase storage."""
    output_prefix = f"output/{user_id}/{session_id}"
    files = list_files(output_prefix)
    cleaned = sorted([f for f in files if f.endswith("_cleaned.csv")])

    datasets: List[Dict[str, Any]] = []
    for fname in cleaned:
        base = fname.replace("_cleaned.csv", "")
        content = download_file(f"{output_prefix}/{fname}")
        df = pd.read_csv(io.BytesIO(content))
        datasets.append(
            {
                "dataset_base": base,
                "filename": fname,
                "columns": list(df.columns),
                "row_count": int(len(df)),
            }
        )
    return datasets


def recommend_features_for_session(user_id: str, session_id: str, dataset_base: str, target: str) -> Dict[str, Any]:
    """Compute MI-ranked feature candidates for a selected session dataset."""
    cleaned_path = f"output/{user_id}/{session_id}/{dataset_base}_cleaned.csv"
    profile_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_profiling.json"

    content = download_file(cleaned_path)
    df = pd.read_csv(io.BytesIO(content))

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in dataset '{dataset_base}'.")

    try:
        profiling_payload = download_json(profile_path)
    except Exception:
        profiling_payload = {}

    working_df, dropped_ids = _drop_profiling_identifiers_from_payload(df, profiling_payload)
    if target not in working_df.columns:
        raise ValueError(f"Target column '{target}' was removed by identifier filtering.")

    available_features = [c for c in working_df.columns if c != target]
    if not available_features:
        raise ValueError("No feature columns available after identifier filtering.")

    task = _detect_task(working_df, target)
    ranked = _score_all_features_by_mi(working_df[available_features + [target]], target=target, task=task)

    max_score = max((score for _, score in ranked), default=0.0)
    recommendations = []
    for feat, score in ranked:
        normalized = 0.0 if max_score <= 0 else float(score / max_score)
        recommendations.append(
            {
                "feature": feat,
                "mi_score": float(score),
                "importance": float(normalized),
                "importance_tag": f"{normalized * 100:.1f}%",
            }
        )

    return {
        "dataset_base": dataset_base,
        "target": target,
        "task": task,
        "dropped_identifier_columns": dropped_ids,
        "available_features": available_features,
        "recommendations": recommendations,
    }


def save_feature_selection_for_session(
    user_id: str,
    session_id: str,
    dataset_base: str,
    target: str,
    selected_features: List[str],
) -> Dict[str, Any]:
    """Persist selected features metadata to Supabase for a session dataset."""
    cleaned_path = f"output/{user_id}/{session_id}/{dataset_base}_cleaned.csv"
    profile_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_profiling.json"
    feature_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_features.json"

    content = download_file(cleaned_path)
    df = pd.read_csv(io.BytesIO(content))

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is not present in dataset '{dataset_base}'.")

    try:
        profiling_payload = download_json(profile_path)
    except Exception:
        profiling_payload = {}

    working_df, dropped_ids = _drop_profiling_identifiers_from_payload(df, profiling_payload)
    if target not in working_df.columns:
        raise ValueError(f"Target column '{target}' was removed by identifier filtering.")

    available_features = [c for c in working_df.columns if c != target]
    available_set = set(available_features)
    invalid = [c for c in selected_features if c not in available_set]
    if invalid:
        raise ValueError(f"Invalid feature(s) not present in candidate list: {invalid}")
    if not selected_features:
        raise ValueError("At least one feature must be selected.")

    task = _detect_task(working_df, target)
    ranked = _score_all_features_by_mi(working_df[available_features + [target]], target=target, task=task)
    ranked_score_map = {feat: float(score) for feat, score in ranked}

    selected_types = {
        feat: _infer_feature_type_from_profiling(feat, working_df[feat], profiling_payload)
        for feat in selected_features
        if feat in working_df.columns
    }
    # structural_types is the authoritative profiling-backed type map.
    # feature_types is kept for backward compat with existing consumers.
    selected_structural_types = selected_types

    selected_scores = {feat: float(ranked_score_map.get(feat, 0.0)) for feat in selected_features}
    max_score = max(selected_scores.values()) if selected_scores else 0.0
    if max_score <= 0:
        normalized = {f: 0.0 for f in selected_features}
    else:
        normalized = {f: float(selected_scores.get(f, 0.0) / max_score) for f in selected_features}

    payload: Dict[str, Any] = {
        "type": "supervised",
        "task": task,
        "target": target,
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "feature_types": selected_types,
        # structural_types mirrors profiling structural_type per feature.
        # model_testing._build_type_map reads this to pick the correct encoder
        # at inference time (categorical → OHE, boolean → int8, etc.).
        "structural_types": selected_structural_types,
        "selection_method": "manual_user_selected_features_with_mi_importance_tags",
        "selector_components": [
            "target_selection",
            "identifier_drop",
            "mutual_information_ranking",
            "manual_selection_from_ranked_features",
        ],
        "dropped_identifier_columns": dropped_ids,
        "source_files": [f"{dataset_base}_cleaned.csv"],
        "processing_mode": "single_file",
        "feature_importance": selected_scores,
        "feature_importance_normalized": normalized,
    }

    upload_json(feature_path, payload)
    return payload


def get_feature_selection_for_session(
    user_id: str,
    session_id: str,
    dataset_base: str,
) -> Dict[str, Any]:
    """
    Read saved _features.json from Supabase and enrich each selected feature
    with profiling metadata (type, null %, unique count, sample values).

    Returns a dict with:
      - target: str
      - task: str
      - selected_features: list of feature names
      - feature_count: int
      - target_card: profiling card dict for the target column
      - feature_cards: list of enriched profiling card dicts per selected feature
      - dataset_base: str
      - source_files: list
      - selection_method: str
    """
    feature_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_features.json"
    profile_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_profiling.json"

    try:
        features_payload = download_json(feature_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"No saved feature selection found for dataset '{dataset_base}'. "
            f"Please select and save features first. (detail: {exc})"
        )

    try:
        profiling_payload = download_json(profile_path)
    except Exception:
        profiling_payload = {}

    target = features_payload.get("target", "")
    selected_features: List[str] = features_payload.get("selected_features", [])
    feature_importance: Dict[str, float] = features_payload.get("feature_importance", {})
    feature_importance_normalized: Dict[str, float] = features_payload.get("feature_importance_normalized", {})

    # Build enriched cards for each selected feature
    feature_cards = [
        _build_profiling_card_for_feature(
            feat=feat,
            profiling_payload=profiling_payload,
            feature_importance=feature_importance,
            feature_importance_normalized=feature_importance_normalized,
        )
        for feat in selected_features
    ]

    # Build card for target column
    target_card = _build_profiling_card_for_target(
        target=target,
        profiling_payload=profiling_payload,
    )

    return {
        "dataset_base": dataset_base,
        "target": target,
        "task": features_payload.get("task", ""),
        "selected_features": selected_features,
        "feature_count": features_payload.get("feature_count", len(selected_features)),
        "target_card": target_card,
        "feature_cards": feature_cards,
        "source_files": features_payload.get("source_files", []),
        "selection_method": features_payload.get("selection_method", ""),
        "dropped_identifier_columns": features_payload.get("dropped_identifier_columns", []),
    }


def create_target_column_for_session(
    user_id: str,
    session_id: str,
    dataset_base: str,
    target_column_name: str,
    metric_columns: List[str],
    target_type: str,
    conditions: List[Dict],
    custom_expression: Optional[str] = None,
    default_value: Any = None,
) -> Dict[str, Any]:
    """Create and persist a new target column for a session dataset."""
    cleaned_path = f"output/{user_id}/{session_id}/{dataset_base}_cleaned.csv"
    profile_path = f"meta_data/{user_id}/{session_id}/{dataset_base}_profiling.json"

    content = download_file(cleaned_path)
    df = pd.read_csv(io.BytesIO(content))

    col_name = target_column_name.strip()
    if not col_name:
        raise ValueError("Target column name cannot be empty.")

    if not metric_columns:
        raise ValueError("At least one metric column must be selected.")

    missing_metrics = [m for m in metric_columns if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Metric column(s) not found: {missing_metrics}")

    default = _safe_default_target_value(target_type) if default_value is None else default_value

    # Restrict rules to selected metrics for safer UX and reproducibility.
    for rule in conditions or []:
        m = str(rule.get("metric", "")).strip()
        if m and m not in metric_columns:
            raise ValueError(f"Rule metric '{m}' is not in selected metric columns")

    new_target = _build_target_series(
        df=df,
        target_type=target_type,
        conditions=conditions,
        custom_expression=custom_expression,
        default_value=default,
    )

    df[col_name] = new_target

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    from data_preprocessing.supabase_storage import upload_file  # local import to avoid global changes
    upload_file(cleaned_path, csv_bytes)

    try:
        profiling_payload = download_json(profile_path)
    except Exception:
        profiling_payload = {}

    profiling_payload = _update_profiling_for_new_column(
        profiling_payload=profiling_payload,
        df=df,
        new_column=col_name,
    )
    upload_json(profile_path, profiling_payload)

    preview = df[[*metric_columns, col_name]].head(25).to_dict(orient="records")

    return {
        "status": "created",
        "dataset_base": dataset_base,
        "target_column": col_name,
        "target_type": str(target_type).strip().lower(),
        "metric_columns": metric_columns,
        "preview": preview,
    }


def main() -> None:
    print("=" * 64)
    print("Feature Engineering + Feature Selection")
    print("=" * 64)

    try:
        userid = _prompt_userid()
        datasets = _load_cleaned_csvs(userid)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    names = list(datasets.keys())
    print(f"\nFound {len(names)} cleaned CSV file(s):")
    for n in names:
        print(f"- {n}")

    # Case 1: single cleaned CSV
    if len(datasets) == 1:
        fname, df = next(iter(datasets.items()))
        dataset_base = Path(fname).stem.replace("_cleaned", "")
        df, dropped_profile_ids = _drop_profiling_identifiers(df, userid, dataset_base)
        if dropped_profile_ids:
            print(f"Dropped profiling identifier columns: {dropped_profile_ids}")

        target = _prompt_target_column(list(df.columns), fname)
        output_name = dataset_base

        try:
            _run_feature_selection_for_dataset(
                userid=userid,
                output_name=output_name,
                df=df,
                target=target,
                source_files=[fname],
                processing_mode="single_file",
                pre_dropped_identifiers=dropped_profile_ids,
            )
        except Exception as exc:
            print(f"Failed: {exc}")
        return

    # Case 2: multiple files, same columns -> filename target
    if _all_same_columns(datasets):
        print("\nDetected mode: multiple files with same columns")
        print("Using target column: filename (derived from source file names)")

        precleaned: Dict[str, pd.DataFrame] = {}
        for fname, df in datasets.items():
            dataset_base = Path(fname).stem.replace("_cleaned", "")
            cleaned_df, _ = _drop_profiling_identifiers(df, userid, dataset_base)
            precleaned[fname] = cleaned_df

        combined_df = _concatenate_with_filename_target(precleaned)

        try:
            _run_feature_selection_for_dataset(
                userid=userid,
                output_name=f"{userid}_combined",
                df=combined_df,
                target="filename",
                source_files=sorted(list(datasets.keys())),
                processing_mode="concatenate_with_filename_target",
            )
        except Exception as exc:
            print(f"Failed: {exc}")
        return

    # Case 3: multiple files, not same columns, but with common identifiers -> merge
    common_ids = _common_identifier_columns(datasets)
    if common_ids:
        print("\nDetected mode: multiple files with common identifier columns")
        print(f"Common identifier columns: {common_ids}")

        merged_df = _merge_on_common_ids(datasets, common_ids)
        # Common merge keys are identifier-like by definition; drop before target input.
        merged_df = merged_df.drop(columns=common_ids, errors="ignore")
        if merged_df.empty:
            print("Merged dataset is empty after inner join. Nothing to process.")
            return

        target = _prompt_target_column(list(merged_df.columns), "merged dataset")

        try:
            _run_feature_selection_for_dataset(
                userid=userid,
                output_name=f"{userid}_merged",
                df=merged_df,
                target=target,
                source_files=sorted(list(datasets.keys())),
                processing_mode="merge_on_common_identifier",
                pre_dropped_identifiers=common_ids,
                merged_on_ids=common_ids,
            )
        except Exception as exc:
            print(f"Failed: {exc}")
        return

    # Case 4: multiple files, different columns and no shared IDs -> process separately
    print("\nDetected mode: multiple files with different columns (separate processing)")

    for fname, df in datasets.items():
        print(f"\nProcessing file: {fname}")
        dataset_base = Path(fname).stem.replace("_cleaned", "")
        df, dropped_profile_ids = _drop_profiling_identifiers(df, userid, dataset_base)
        if dropped_profile_ids:
            print(f"Dropped profiling identifier columns: {dropped_profile_ids}")

        target = _prompt_target_column(list(df.columns), fname)
        output_name = dataset_base

        try:
            _run_feature_selection_for_dataset(
                userid=userid,
                output_name=output_name,
                df=df,
                target=target,
                source_files=[fname],
                processing_mode="separate_processing",
                pre_dropped_identifiers=dropped_profile_ids,
            )
        except Exception as exc:
            print(f"Failed for {fname}: {exc}")


if __name__ == "__main__":
    main()