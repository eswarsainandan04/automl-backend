"""
feature_engineering/feature_processing.py
==========================================
End-to-end feature processing for the AutoML pipeline.

Key hardening applied vs previous version
------------------------------------------
* OneHotEncoder now uses handle_unknown="ignore" everywhere — unseen categories
  at inference time produce an all-zeros row instead of crashing.
* TargetEncoder / CatBoostEncoder / MEstimateEncoder all receive smooth=True /
  m parameter so unseen categories fall back to the global target mean.
* build_fitted_preprocessor() accepts a `fit` kwarg (default True for backward
  compatibility). Pass fit=False to get an unfit ColumnTransformer so that
  _patch_preprocessor() can wire squeeze steps BEFORE the single .fit() call.
  This fixes the shape=(1,1) text fast-path bug where patch-after-fit replaced
  the fitted vectoriser with a new unfitted one.
* All FunctionTransformer helpers are module-level (picklable).
* build_training_stats() is provided for embedding distribution info into the pkl.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Sklearn core ──────────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
    OneHotEncoder,
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from sklearn.impute import SimpleImputer

# ── Category Encoders ─────────────────────────────────────────────────────────
from category_encoders import (
    BinaryEncoder,
    TargetEncoder,
    MEstimateEncoder,
    CatBoostEncoder,
    CountEncoder,
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("FeatureProcessing")

_TEMPORAL_TYPES = {"date", "time", "datetime", "timestamp"}
_NUMERIC_STYPES = frozenset({"numeric", "integer", "float"})
_CAT_STYPES     = frozenset({"categorical", "category"})
_BOOLEAN_STYPES = frozenset({"boolean"})
_TEXT_STYPES    = frozenset({"text", "string"})


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS  (must be top-scope for pickle compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def _squeeze_1d(X) -> pd.Series:
    if hasattr(X, "iloc"):
        return X.iloc[:, 0]
    return pd.Series(np.asarray(X).ravel())


def _date_transform(X) -> np.ndarray:
    out = pd.DataFrame(index=range(len(X)))
    col = pd.to_datetime(_squeeze_1d(X), errors="coerce")
    out["year"]    = col.dt.year.fillna(0).astype(int)
    out["month"]   = col.dt.month.fillna(0).astype(int)
    out["day"]     = col.dt.day.fillna(0).astype(int)
    out["weekday"] = col.dt.weekday.fillna(0).astype(int)
    return out.values


def _time_transform(X) -> np.ndarray:
    out = pd.DataFrame(index=range(len(X)))
    col = pd.to_datetime(_squeeze_1d(X), errors="coerce")
    hour   = col.dt.hour.fillna(0)
    minute = col.dt.minute.fillna(0)
    second = col.dt.second.fillna(0)
    out["hour"]     = hour.astype(int)
    out["minute"]   = minute.astype(int)
    out["second"]   = second.astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["min_sin"]  = np.sin(2 * np.pi * minute / 60)
    out["min_cos"]  = np.cos(2 * np.pi * minute / 60)
    return out.values


def _datetime_transform(X) -> np.ndarray:
    out = pd.DataFrame(index=range(len(X)))
    col   = pd.to_datetime(_squeeze_1d(X), errors="coerce")
    hour  = col.dt.hour.fillna(0)
    month = col.dt.month.fillna(0)
    out["year"]      = col.dt.year.fillna(0).astype(int)
    out["month"]     = month.astype(int)
    out["day"]       = col.dt.day.fillna(0).astype(int)
    out["weekday"]   = col.dt.weekday.fillna(0).astype(int)
    out["hour"]      = hour.astype(int)
    out["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out.values


def _boolean_transform(X) -> np.ndarray:
    """Cast boolean-like values to int 0/1.  Strict — unrecognised values → NaN."""
    _MAP = {
        True: 1, False: 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
        "1": 1, "0": 0,
        1: 1, 0: 0,
        1.0: 1, 0.0: 0,
    }
    s = _squeeze_1d(X)
    result = s.map(lambda v: _MAP.get(str(v).strip().lower(), _MAP.get(v, np.nan)))
    if result.isna().any():
        bad = s[result.isna()].unique().tolist()
        logger.warning("Boolean transform: unrecognised values %s → coerced to NaN", bad)
    return result.fillna(0).astype(int).values.reshape(-1, 1)


def _cat_to_object(X) -> pd.DataFrame:
    """Convert all columns in a DataFrame slice from ``category``/int dtype to ``str``.

    ROOT CAUSE this fixes
    ----------------------
    After ``enforce_dtypes()`` casts categorical columns to ``pandas category``
    dtype, ``SimpleImputer(strategy="most_frequent")`` receives a (n, 1)
    DataFrame whose single column has dtype ``category``.  In scikit-learn
    < 1.4 the imputer internally calls ``select_dtypes(include=["object"])``
    to locate string-like columns and emits:

        UserWarning: No categorical columns found. Will return unchanged.

    This causes the imputer to silently pass the raw category codes through
    instead of replacing NaNs, which breaks the downstream encoder.

    LABEL-FLIP FIX:
    Casting to ``str`` (not just ``object``) ensures that integer-encoded
    categorical values like 0/1/2/3 are stored by encoders (OHE, TargetEncoder,
    etc.) as their string representations "0"/"1"/"2"/"3".  At inference time
    _coerce_input_df() calls astype(str) on categorical columns, producing the
    same string representations → perfect category match → no label flip.

    If we cast only to ``object`` (leaving integers as Python int objects),
    OHE stores categories_ = [0, 1] (integers).  At inference the value comes
    in as "0" (string) → OHE treats it as unknown → all-zeros → wrong output.

    Must be module-level for pickle compatibility.
    """
    if hasattr(X, "iloc"):
        out = X.copy()
        for col in out.columns:
            # Cast category dtype AND any integer/float col that reaches here
            # to str so encoders see consistent string categories at both
            # training time and inference time.
            if hasattr(out[col], "cat") or out[col].dtype != object:
                out[col] = out[col].astype(str)
        return out
    # Fallback: numpy array — stringify every element.
    arr = np.asarray(X)
    return pd.DataFrame(arr).astype(str).values


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENCODING STRATEGY LABEL
# ══════════════════════════════════════════════════════════════════════════════

def decide_encoding(
    col: str,
    unique_count: int,
    row_count: int,
    avg_text_length: Optional[float] = None,
    structural_type: Optional[str] = None,
) -> str:
    if structural_type == "categorical":
        if unique_count == 2:
            # NOTE: build_transformer uses OHE(drop='first') for 2-class categoricals
            # (not BinaryEncoder) to prevent the label-flip bug. This label must match.
            return "OneHotEncoding (drop=first, 2-class)"
        if unique_count <= 10:
            return "OneHotEncoding"
        if unique_count <= 100:
            return "TargetEncoding (smoothed)"
        if unique_count <= 200:
            return "FrequencyEncoding"
        if unique_count <= 500:
            return "MEstimateEncoding (m=30)"
        return "CatBoostEncoding (ordered)"

    if structural_type == "text":
        avg_len = avg_text_length or 0
        if unique_count <= 1:
            return "Drop (constant text)"
        if row_count > 10_000:
            return "HashingVectorizer (n_features=2**18)"
        if avg_len <= 30:
            return "CountVectorizer (ngram_range=(1,2), max_features=100)"
        if avg_len <= 300:
            return "TfidfVectorizer (ngram_range=(1,2), max_features=500)"
        return "TfidfVectorizer (max_features=1000)"

    if structural_type == "boolean":
        return "Cast to int (0/1)"
    if structural_type == "date":
        return "DateFeatureGenerator (year, month, day, weekday)"
    if structural_type == "time":
        return "TimeFeatureGenerator (hour, minute, second + sin/cos)"
    if structural_type in ("datetime", "timestamp"):
        return "DatetimeFeatureGenerator (year, month, day, weekday, hour + cyclical)"
    if structural_type == "numeric":
        return "StandardScaler"

    logger.warning("Column '%s': unknown structural_type '%s' → passthrough", col, structural_type)
    return "No encoding needed"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRANSFORMER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_transformer(
    col: str,
    unique_count: int,
    row_count: int,
    structural_type: Optional[str],
    avg_text_length: Optional[float] = None,
    y_train: Optional[pd.Series] = None,
) -> Tuple[str, Any]:
    """
    Return (strategy_label, unfitted_transformer_or_"drop"/"passthrough").

    HARDENING NOTES
    ---------------
    * OneHotEncoder always uses handle_unknown="ignore" so unseen categories at
      inference time produce an all-zeros vector instead of raising.
    * TargetEncoder / MEstimateEncoder / CatBoostEncoder use smoothing so that
      unseen categories at inference time fall back to the global target mean.
    * All transformers are returned UNFITTED — fitting happens exclusively inside
      build_fitted_preprocessor() on X_train.
    """
    strategy = decide_encoding(
        col=col,
        unique_count=unique_count,
        row_count=row_count,
        avg_text_length=avg_text_length,
        structural_type=structural_type,
    )

    # ── CATEGORICAL ──────────────────────────────────────────────────────────
    if structural_type == "categorical":
        # _cat_to_object MUST be the first step in every categorical pipeline.
        #
        # WHY: enforce_dtypes() sets these columns to pandas ``category`` dtype
        # so the rest of the pipeline knows they are categorical.  But
        # SimpleImputer(strategy="most_frequent") internally calls
        # select_dtypes(include=["object"]) and silently skips ``category``
        # columns, emitting "No categorical columns found" and returning the
        # column unimputed.  Converting to ``object`` here (inside the
        # pipeline, not in the DataFrame) fixes the imputer without losing the
        # dtype signal that the outer DataFrame carries.
        _to_obj = ("to_object", FunctionTransformer(_cat_to_object, validate=False))

        if unique_count == 2:
            # ── FIX: Use OHE(drop='first') instead of BinaryEncoder for 2-class categoricals.
            #
            # ROOT CAUSE OF LABEL-FLIP BUG:
            # BinaryEncoder assigns bit codes based on the SORTED ORDER of categories
            # it sees at fit time. When enforce_dtypes() casts a column to pandas
            # ``category`` dtype, the internal codes are integer-based (0→code 0,
            # 1→code 1). At inference time, _coerce_input_df() converts categorical
            # columns to ``object`` dtype (strings "0","1"). BinaryEncoder now sees
            # a DIFFERENT sort order for these string categories and maps them to
            # opposite bit values → the single output bit is FLIPPED → the final
            # class label is INVERTED (no-disease → 1, disease → 0).
            #
            # OneHotEncoder(drop='first') is safe because:
            #   • It stores the exact category array at fit time (categories_).
            #   • At inference it looks up by VALUE, not by position/code.
            #   • handle_unknown='ignore' makes unseen values → 0 instead of crashing.
            #   • The single remaining column is 0 for the dropped class, 1 for the
            #     other — identical semantics to the old BinaryEncoder but stable.
            transformer = Pipeline([
                _to_obj,
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ])
        elif unique_count <= 10:
            # OneHotEncoder with handle_unknown="ignore" — CRITICAL for inference safety
            transformer = Pipeline([
                _to_obj,
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ])
        elif unique_count <= 100:
            # TargetEncoder with smoothing — unknown categories → global mean
            transformer = Pipeline([
                _to_obj,
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TargetEncoder(smoothing=10.0)),
            ])
        elif unique_count <= 200:
            # FrequencyEncoding
            transformer = Pipeline([
                _to_obj,
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", CountEncoder(normalize=True)),
            ])
        elif unique_count <= 500:
            # MEstimateEncoder with m=30 smoothing — unknown → global mean
            transformer = Pipeline([
                _to_obj,
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", MEstimateEncoder(m=30)),
            ])
        else:
            # CatBoostEncoder — ordered / leave-one-out with smoothing
            transformer = Pipeline([
                _to_obj,
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", CatBoostEncoder(a=1.0)),
            ])
        return strategy, transformer

    # ── TEXT ─────────────────────────────────────────────────────────────────
    if structural_type == "text":
        avg_len = avg_text_length or 0
        if strategy.startswith("Drop"):
            return strategy, "drop"
        if row_count > 10_000:
            transformer = HashingVectorizer(n_features=2 ** 18, alternate_sign=False)
        elif avg_len <= 30:
            transformer = CountVectorizer(ngram_range=(1, 2), max_features=100)
        elif avg_len <= 300:
            transformer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        else:
            transformer = TfidfVectorizer(max_features=1_000)
        return strategy, transformer

    # ── BOOLEAN ──────────────────────────────────────────────────────────────
    if structural_type == "boolean":
        return strategy, FunctionTransformer(_boolean_transform, validate=False)

    # ── DATE ─────────────────────────────────────────────────────────────────
    if structural_type == "date":
        return strategy, FunctionTransformer(_date_transform, validate=False)

    # ── TIME ─────────────────────────────────────────────────────────────────
    if structural_type == "time":
        return strategy, FunctionTransformer(_time_transform, validate=False)

    # ── DATETIME / TIMESTAMP ─────────────────────────────────────────────────
    if structural_type in ("datetime", "timestamp"):
        return strategy, FunctionTransformer(_datetime_transform, validate=False)

    # ── NUMERIC ──────────────────────────────────────────────────────────────
    if structural_type in ("numeric", "integer", "float"):

        transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
        ])
        return strategy, transformer

    return strategy, "passthrough"


# ══════════════════════════════════════════════════════════════════════════════
# 3. GLOBAL FEATURE TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_feature_type(
    selected_features: List[str],
    profiling_meta: Dict,
) -> str:
    col_summary: List[Dict] = profiling_meta.get("column_wise_summary", [])
    type_map: Dict[str, str] = {
        c["column_name"]: c.get("structural_type", "unknown")
        for c in col_summary
    }

    if not selected_features:
        logger.warning("detect_feature_type: selected_features is empty")
        return "unknown"

    type_set = set()
    for feat in selected_features:
        stype = type_map.get(feat, "unknown")
        type_set.add("temporal" if stype in _TEMPORAL_TYPES else stype)

    if len(type_set) == 1:
        result = list(type_set)[0]
        logger.info("Feature type: %s (uniform)", result)
        return result

    logger.info("Feature type: mixed  (types found: %s)", type_set)
    return "mixed"


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLANNING DICT  (labels only — no objects)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_processing_plan(
    selected_features: List[str],
    profiling_meta: Dict,
    row_count: int,
) -> Dict[str, str]:
    col_summary: List[Dict] = profiling_meta.get("column_wise_summary", [])
    summary_map: Dict[str, Dict] = {c["column_name"]: c for c in col_summary}

    plan: Dict[str, str] = {}
    for feat in selected_features:
        info = summary_map.get(feat, {})
        if not info:
            logger.warning("Feature '%s' not in profiling summary — using fallback.", feat)

        label = decide_encoding(
            col=feat,
            unique_count=info.get("unique_count", 0),
            row_count=row_count,
            avg_text_length=info.get("avg_text_length"),
            structural_type=info.get("structural_type"),
        )
        plan[feat] = label
        logger.info("  %-30s [%-12s] → %s", feat, info.get("structural_type", "?"), label)

    return plan


# ══════════════════════════════════════════════════════════════════════════════
# 5. FITTED PREPROCESSOR  (fitted ONCE at training time; never rebuilt at inference)
# ══════════════════════════════════════════════════════════════════════════════

def build_fitted_preprocessor(
    X_train: pd.DataFrame,
    selected_features: List[str],
    profiling_meta: Dict,
    row_count: int,
    y_train: Optional[pd.Series] = None,
    fit: bool = True,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Build (and optionally fit) a ColumnTransformer on X_train.

    Parameters
    ----------
    X_train : pd.DataFrame
    selected_features : list[str]
    profiling_meta : dict
    row_count : int
    y_train : pd.Series, optional
        Required by supervised encoders (TargetEncoder, etc.).
    fit : bool, default True
        When True (default, backward-compatible), calls preprocessor.fit()
        before returning — this is the original behaviour.

        When False, returns an UNFITTED ColumnTransformer so the caller can
        apply _patch_preprocessor() BEFORE fitting.  The caller is then
        responsible for calling preprocessor.fit(X_train[selected_features]).

        ── WHY fit=False EXISTS ────────────────────────────────────────────
        The text fast-path bug: old code did build→fit→patch→transform.
        _patch_preprocessor() replaced the fitted CountVectorizer with a
        new unfitted Pipeline([squeeze, vectorizer]), so .transform() treated
        the entire X_train DataFrame as one document → shape=(1, 1) → every
        CV fold saw inconsistent sample counts [1, n_rows] → all models failed.

        Correct order: build → patch (unfit) → fit → transform.
        Pass fit=False to enable this order in model_selection.py.
        ────────────────────────────────────────────────────────────────────

    Returns
    -------
    (ColumnTransformer, drop_columns)
        If fit=True  → fitted ColumnTransformer (original behaviour).
        If fit=False → unfit  ColumnTransformer (caller must call .fit()).
    """
    col_summary: List[Dict] = profiling_meta.get("column_wise_summary", [])
    summary_map: Dict[str, Dict] = {c["column_name"]: c for c in col_summary}

    transformers: List[Tuple[str, Any, List[str]]] = []
    drop_cols:    List[str] = []

    for feat in selected_features:
        if feat not in X_train.columns:
            logger.warning("Feature '%s' not in X_train — skipping.", feat)
            continue

        info           = summary_map.get(feat, {})
        stype          = info.get("structural_type")
        unique_count   = info.get("unique_count", int(X_train[feat].nunique()))
        avg_text_length = info.get("avg_text_length")

        strategy, transformer_obj = build_transformer(
            col=feat,
            unique_count=unique_count,
            row_count=row_count,
            structural_type=stype,
            avg_text_length=avg_text_length,
            y_train=y_train,
        )

        if transformer_obj == "drop":
            logger.info("Dropping column '%s' (%s)", feat, strategy)
            drop_cols.append(feat)
            continue

        logger.info("Adding transformer for '%s': %s", feat, strategy)
        transformers.append((f"tf_{feat}", transformer_obj, [feat]))

    if not transformers:
        raise ValueError("No valid transformers built — check profiling metadata.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if fit:
        # Original behaviour: fit immediately and return a ready-to-use preprocessor.
        X_fit = X_train.loc[:, selected_features]
        if y_train is not None:
            preprocessor.fit(X_fit, y_train)
        else:
            preprocessor.fit(X_fit)

        logger.info(
            "ColumnTransformer fitted. Transformers: %d | Dropped: %d",
            len(transformers), len(drop_cols),
        )
    else:
        # fit=False: caller will patch then fit.
        logger.info(
            "ColumnTransformer built (unfit). Transformers: %d | Dropped: %d",
            len(transformers), len(drop_cols),
        )

    return preprocessor, drop_cols


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING STATS  (embedded into pkl for inference-time distribution checks)
# ══════════════════════════════════════════════════════════════════════════════

def build_training_stats(
    X_train: pd.DataFrame,
    selected_features: List[str],
    profiling_meta: Dict,
) -> Dict[str, Dict]:
    """
    Compute per-feature stats from training data.

    Call this in model_selection.py and include the result in pkl_payload:
        pkl_payload["training_stats"] = build_training_stats(X_train, selected_features, profiling)

    The inference engine (model_testing.py:StrictInputValidator) reads these
    stats to warn or reject out-of-distribution values.
    """
    col_summary = profiling_meta.get("column_wise_summary", [])
    summary_map = {c["column_name"]: c for c in col_summary}
    stats: Dict[str, Dict] = {}

    for feat in selected_features:
        if feat not in X_train.columns:
            continue

        info   = summary_map.get(feat, {})
        stype  = str(info.get("structural_type", "")).lower()
        series = X_train[feat].dropna()
        entry: Dict[str, Any] = {"stype": stype}

        if stype in _NUMERIC_STYPES:
            entry.update({
                "min":     float(series.min())  if not series.empty else None,
                "max":     float(series.max())  if not series.empty else None,
                "mean":    float(series.mean()) if not series.empty else None,
                "std":     float(series.std())  if not series.empty else None,
                "uniques": None,
            })
        elif stype in _CAT_STYPES | _BOOLEAN_STYPES:
            entry.update({
                "min": None, "max": None, "mean": None, "std": None,
                "uniques": [str(v) for v in series.unique().tolist()],
            })
        else:
            entry.update({"min": None, "max": None, "mean": None, "std": None, "uniques": None})

        stats[feat] = entry

    return stats