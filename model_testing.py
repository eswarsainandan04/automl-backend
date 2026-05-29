from __future__ import annotations

import json
import pickle
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_preprocessing.supabase_storage import download_file, download_json, list_files


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "storage" / "output"
META_ROOT = BASE_DIR / "storage" / "meta_data"


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_from_supabase(path: str) -> Dict:
    return download_json(path)


def _discover_model_pkls(user_output_dir: Path) -> List[Path]:
    return sorted(user_output_dir.glob("*_model.pkl"))


def _model_dataset_base(model_pkl: Path) -> str:
    stem = model_pkl.stem
    suffixes = ["_best_model", "_model_pipeline", "_model"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_metadata_paths(user_meta_dir: Path, model_pkl: Path) -> Tuple[Path, Optional[Path], str]:
    """Resolve metadata paths for a model file.

    Profiling metadata is optional because combined datasets may only produce
    a merged features metadata file.
    """
    features_files = sorted(user_meta_dir.glob("*_features.json"))
    profiling_files = sorted(user_meta_dir.glob("*_profiling.json"))

    if not features_files:
        raise FileNotFoundError(f"Missing features metadata json files in {user_meta_dir}")

    base_guess = _model_dataset_base(model_pkl)
    guessed_features = user_meta_dir / f"{base_guess}_features.json"
    guessed_profile = user_meta_dir / f"{base_guess}_profiling.json"
    if guessed_features.exists():
        return guessed_features, guessed_profile if guessed_profile.exists() else None, base_guess

    model_stem = model_pkl.stem
    feature_bases = [p.stem.replace("_features", "") for p in features_files]
    candidates = [b for b in feature_bases if model_stem.startswith(b)]

    if not candidates:
        raise FileNotFoundError(
            f"Could not match metadata for model file '{model_pkl.name}' in {user_meta_dir}"
        )

    best_base = sorted(candidates, key=len, reverse=True)[0]
    fallback_profile = user_meta_dir / f"{best_base}_profiling.json"
    return (
        user_meta_dir / f"{best_base}_features.json",
        fallback_profile if fallback_profile.exists() else None,
        best_base,
    )


def _normalize_framework(payload: Dict) -> str:
    return str(payload.get("framework", "unknown")).lower()


def _load_model(payload: Dict):
    framework = _normalize_framework(payload)

    if framework == "autogluon":
        from autogluon.tabular import TabularPredictor
        if payload.get("predictor") is not None:
            return framework, payload["predictor"]
        return framework, TabularPredictor.load(payload["predictor_path"])

    elif framework == "sklearn":
        model = payload.get("pipeline")
        if model is None:
            model = payload.get("model")
        if model is None:
            raise ValueError("Sklearn payload missing pipeline/model object")
        return framework, model

    elif framework == "flaml":
        model = payload.get("predictor")
        if model is None:
            raise ValueError("FLAML payload missing predictor object")
        return framework, model

    else:
        raise ValueError("Unsupported model type")


def _pick_first_loadable_model(user_output_dir: Path, user_meta_dir: Path) -> Tuple[Path, Dict, Path, Optional[Path], str]:
    model_candidates = _discover_model_pkls(user_output_dir)
    if not model_candidates:
        raise FileNotFoundError(f"no *_model.pkl files found in {user_output_dir}")

    def _priority(p: Path) -> Tuple[int, str]:
        name = p.name
        canonical = name.endswith("_model.pkl") and not name.endswith("_best_model.pkl")
        return (0 if canonical else 1, name)

    for model_path in sorted(model_candidates, key=_priority):
        try:
            features_path, profile_path, dataset_base = _resolve_metadata_paths(user_meta_dir, model_path)
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
            return model_path, payload, features_path, profile_path, dataset_base
        except Exception as exc:
            print(f"Skipping model '{model_path.name}' due to load issue: {exc}")

    raise RuntimeError("No loadable model pickle found for this user.")


def _validate_input_ranges(df: pd.DataFrame, training_stats: Optional[Dict] = None) -> None:
    """Validate numeric input values against training data min/max from training_stats.

    Uses the dynamic per-feature min/max recorded at training time (stored in
    the pkl under "training_stats") instead of any hardcoded dataset-specific
    constants.  Logs a warning for out-of-range values but does NOT raise —
    the pipeline's own imputer/scaler handles edge cases gracefully.
    """
    import logging as _log
    _logger = _log.getLogger("ModelTesting")

    if not training_stats:
        return  # Nothing to validate against — skip silently

    _NUMERIC_STYPES_LOCAL = frozenset({"numeric", "integer", "float"})

    for col, info in training_stats.items():
        if col not in df.columns:
            continue
        stype = str(info.get("stype", "")).lower()
        if stype not in _NUMERIC_STYPES_LOCAL:
            continue
        low  = info.get("min")
        high = info.get("max")
        if low is None or high is None:
            continue
        val = df[col].iloc[0]
        if pd.isna(val):
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if not (low <= fval <= high):
            _logger.warning(
                "Input value for '%s' is outside the training range [%s, %s]. Got: %s. "
                "Prediction may be less reliable.",
                col, low, high, fval,
            )


def _build_type_map(
    features_meta: Dict,
    profile_meta: Optional[Dict],
    pkl_type_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a column → structural_type map used by the UI schema and _coerce_input_df.

    Priority (highest → lowest):
    1. pkl embedded type_map  ← written at training time from the same profiling
       JSON used to build the preprocessor; guaranteed consistent; works offline.
       (FIX: new layer — eliminates Supabase dependency at inference time)
    2. profiling column_wise_summary structural_type  ← live download fallback
    3. features_meta["structural_types"]  ← written by feature_selection.py
    4. features_meta["feature_types"]     ← legacy naive-inferred fallback (lowest)

    WHY pkl_type_map wins:
    The embedded type_map is built from the exact profiling JSON that was used to
    fit the ColumnTransformer at training time.  It cannot drift due to Supabase
    download failures, session changes, or file overwrites.  Layers 2-4 are kept
    as cascading fallbacks for pkls created before this fix was deployed.
    """
    type_map = {}

    # Layer 4 (lowest priority): legacy feature_types — naive dtype-based
    for col, t in features_meta.get("feature_types", {}).items():
        type_map[col] = str(t).lower()

    # Layer 3: structural_types written by the fixed feature_selection.py —
    # profiling-backed, overrides naive feature_types for the same column.
    for col, t in features_meta.get("structural_types", {}).items():
        if col and t:
            type_map[col] = str(t).lower()

    # Layer 2: live profiling JSON structural_type (download fallback).
    if profile_meta:
        for item in profile_meta.get("column_wise_summary", []):
            col = item.get("column_name")
            stype = str(item.get("structural_type", "")).strip().lower()
            if not col or not stype:
                continue
            type_map[col] = stype

    # Layer 1 (HIGHEST priority): pkl-embedded type_map. Built at training time
    # from the same profiling JSON used to fit the preprocessor. Guaranteed
    # consistent. Works without any Supabase download at inference time.
    # FIX: this layer is new — it overrides all other sources.
    if pkl_type_map:
        for col, stype in pkl_type_map.items():
            if col and stype:
                type_map[col] = str(stype).lower()

    return type_map


# ══════════════════════════════════════════════════════════════════════════════
# FIX: INFERENCE DTYPE COERCION
# ══════════════════════════════════════════════════════════════════════════════

_NUMERIC_STYPES = frozenset({"numeric", "integer", "float"})
_CAT_STYPES     = frozenset({"categorical", "category"})
_BOOLEAN_STYPES = frozenset({"boolean"})


def _canonicalize_categorical_value(raw_value: Any, known_uniques: List[str]) -> str:
    """Map user-entered category to the closest known training category string.

    Handles common drift patterns:
    - case/whitespace differences ("Diesel" vs "diesel ")
    - numeric-string variants ("0" vs "0.0")
    """
    value = str(raw_value).strip()
    if value in known_uniques:
        return value

    lower_lookup: Dict[str, str] = {}
    for u in known_uniques:
        lu = u.lower()
        if lu not in lower_lookup:
            lower_lookup[lu] = u

    lv = value.lower()
    if lv in lower_lookup:
        return lower_lookup[lv]

    try:
        fval = float(value)
        for u in known_uniques:
            try:
                if np.isclose(float(u), fval, rtol=0.0, atol=1e-12):
                    return u
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError):
        pass

    return value


def _collect_prediction_warnings(
    raw_df: pd.DataFrame,
    coerced_df: pd.DataFrame,
    selected_features: List[str],
    type_map: Dict[str, str],
    training_stats: Optional[Dict] = None,
) -> List[str]:
    """Return human-readable warnings that explain potentially unreliable predictions."""
    warnings: List[str] = []
    stats = training_stats or {}

    for col in selected_features:
        if col not in raw_df.columns or col not in coerced_df.columns:
            continue

        stype = str(type_map.get(col, "")).lower()
        raw_val = raw_df[col].iloc[0]
        val = coerced_df[col].iloc[0]
        info = stats.get(col, {}) if isinstance(stats, dict) else {}

        if stype in _NUMERIC_STYPES:
            if pd.isna(val):
                warnings.append(
                    f"{col}: value '{raw_val}' could not be parsed as numeric; imputer/default behavior may affect prediction."
                )
                continue

            low = info.get("min")
            high = info.get("max")
            if low is not None and high is not None:
                try:
                    fval = float(val)
                    if not (float(low) <= fval <= float(high)):
                        warnings.append(
                            f"{col}: {fval} is outside training range [{low}, {high}] — prediction may be less reliable."
                        )
                except (TypeError, ValueError):
                    pass

        elif stype in _CAT_STYPES | _BOOLEAN_STYPES:
            known = [str(u) for u in (info.get("uniques") or []) if u is not None]
            if known and str(val) not in set(known):
                warnings.append(
                    f"{col}: category '{val}' was not seen in training data — encoder may treat it as unknown."
                )

    # Keep warning list stable and duplicate-free
    deduped = list(dict.fromkeys(warnings))
    return deduped


def _coerce_input_df(
    df: pd.DataFrame,
    selected_features: List[str],
    type_map: Dict[str, str],
    training_stats: Optional[Dict] = None,
) -> pd.DataFrame:
    """Coerce an inference DataFrame to match the dtypes the pipeline was trained on.

    ROOT CAUSE THIS FIXES
    ─────────────────────
    At training time enforce_dtypes() casts columns by structural_type:
      • numeric  → float64
      • boolean  → int8
      • categorical → pandas category, then _cat_to_object() converts to str

    OHE(drop='first') fitted on categorical columns stores categories_ as string
    arrays (e.g. ["0", "1"]).  At inference, if a value arrives as Python int 0
    (from JSON deserialization) instead of str "0", OHE treats it as an UNKNOWN
    category → all-zeros output → wrong/flipped prediction.

    Similarly, numeric columns must be float64, not str, for StandardScaler.

    This function applies the exact same dtype rules as enforce_dtypes() so the
    pipeline always receives consistent types at both train and inference time.

    SIGNATURE CHANGE vs old version
    ────────────────────────────────
    The old signature accepted `profile_meta: Optional[Dict]` and rebuilt the
    type_map internally.  When profile_meta was an empty dict {} (returned by
    _load_session_artifacts on load failure) the rebuilt type_map was also {}
    and NO coercion was applied — the silent root cause of the label flip.

    Now callers must pass the fully-resolved type_map (built by _build_type_map
    which merges profiling + features metadata + training_stats).  This makes
    the "no coercion" failure mode impossible unless the caller explicitly passes
    an empty map.

    Column type decision rules:
      • structural_type in {numeric, integer, float} → pd.to_numeric → float64
      • structural_type in {boolean}                 → bool-string map → int8
      • structural_type in {categorical, category}   → astype(str) — CRITICAL
      • all other types (text, date, etc.)           → leave unchanged
    """
    _bool_map = {
        "true": 1, "false": 0,
        "yes":  1, "no":    0,
        "1":    1, "0":     0,
        "1.0":  1, "0.0":   0,
    }

    # ── If type_map is empty, fall back to training_stats ─────────────────────
    # This can happen only when profiling JSON was not stored in the pkl AND the
    # pkl predates the embedded type_map fix. training_stats (always saved by
    # model_selection.py) records the stype each column had at training time and
    # is a reliable secondary source.
    if not type_map and training_stats:
        import logging as _log
        _log.getLogger("ModelTesting").warning(
            "_coerce_input_df: type_map is empty — falling back to training_stats. "
            "This means profiling metadata was not available. Coercion may be incomplete."
        )
        type_map = {
            col: str(info.get("stype", "")).lower()
            for col, info in training_stats.items()
        }

    for col in selected_features:
        if col not in df.columns:
            continue

        stype = type_map.get(col, "")

        if stype in _NUMERIC_STYPES:
            # "120" → 120.0, "1.5" → 1.5, "" → NaN (handled by imputer)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif stype in _BOOLEAN_STYPES:
            # FIX-2: boolean structural_type means the column uses
            # FunctionTransformer(_boolean_transform) at training time —
            # which produces int arrays. So int8 is correct HERE.
            #
            # IMPORTANT: Columns that profiling tags as 'categorical' (even if
            # they look binary, like sex/fbs/exang) go through the categorical
            # branch BELOW and are cast to str. This matches what _cat_to_object()
            # does inside the OHE pipeline at training — it stringifies before OHE
            # fits, so OHE stores categories_=["0","1"] (strings). Giving it int8
            # at inference would produce an unknown-category → all-zeros → wrong
            # prediction. The key: boolean → int8 ONLY when the column truly uses
            # _boolean_transform (FunctionTransformer), not OHE.
            series = df[col].astype(str).str.strip().str.lower()
            df[col] = series.map(lambda v: _bool_map.get(v, np.nan))
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

        elif stype in _CAT_STYPES:
            # CRITICAL: cast to str so OHE / TargetEncoder / CountEncoder receive
            # exactly the same dtype they saw at training time after _cat_to_object()
            # ran inside each encoder pipeline.
            #
            # This covers ALL integer-encoded categoricals (cp, ca, thal, slope,
            # restecg) AND binary categoricals (sex, fbs, exang) whose profiling
            # structural_type is 'categorical'. OHE(drop='first') stores categories_
            # as string arrays ["0","1"]. If we pass int 0 instead of str "0", OHE
            # treats it as an unknown category → all-zeros → wrong prediction.
            df[col] = df[col].astype(str)

            # Align user input to training category representation where possible.
            # This resolves common mismatches like "0" vs "0.0" and case variants.
            known_uniques = [
                str(u).strip()
                for u in (training_stats or {}).get(col, {}).get("uniques", [])
                if u is not None
            ]
            if known_uniques:
                df[col] = df[col].map(lambda v: _canonicalize_categorical_value(v, known_uniques))

        # text / date / datetime / unknown → leave as-is

    return df


def _collect_input(selected_features: List[str]) -> Dict:
    print("\nEnter values (press Enter for missing):")
    row = {}

    for col in selected_features:
        val = input(f"{col}: ").strip()
        row[col] = None if val == "" else val

    return row


def _field_input_type(feature_type: str) -> str:
    t = str(feature_type).strip().lower()
    if t in {"numeric", "integer", "float"}:
        return "number"
    if t in {"text", "string"}:
        return "text_area"
    if t in {"boolean", "categorical", "category"}:
        return "categorical"
    return "text_area"


def _extract_options_from_column_name(column_name: str) -> List[str]:
    """Infer enum-like options from names such as feature_(yes/no) or feature_(true/false)."""
    m = re.search(r"_\(([^)]+)\)", str(column_name or "").strip().lower())
    if not m:
        return []

    inside = m.group(1)
    parts = [p.strip() for p in inside.split("/") if p.strip()]
    if len(parts) < 2:
        return []

    deduped: List[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return deduped


def _coerce_boolean_options(options: List[str]) -> List[str]:
    """Return the actual unique values found in the training data for a boolean column.

    BUG FIX: Previously this always returned ["0", "1"] regardless of what the
    training data contained.  If the training data had "yes"/"no" or "True"/"False",
    the UI showed "0"/"1" which _coerce_input_df then correctly mapped — BUT the
    user had no way to know which option means which.  Worse, if the actual
    cleaned CSV had "0"/"1" stored as int8 but displayed as string, presenting
    "0"/"1" is still correct.

    The right behavior: show exactly what is in the cleaned CSV (already
    extracted as `options` from value_counts), falling back to ["0","1"] only
    when options is empty.  _coerce_input_df handles all known boolean string
    variants (yes/no, true/false, 1/0) uniformly.
    """
    if options:
        return options
    return ["0", "1"]


def _download_prefix_to_local(prefix: str, local_dir: Path) -> None:
    """Download all files under a Supabase prefix into local_dir."""
    files = list_files(prefix, recursive=True)
    for rel_path in files:
        content = download_file(f"{prefix}/{rel_path}")
        target = local_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.write(content)


def _ensure_local_autogluon_artifacts(payload: Dict[str, Any], user_id: str, session_id: str) -> Dict[str, Any]:
    """Mirror required AutoGluon artifact folder from Supabase for local predictor loading."""
    if _normalize_framework(payload) != "autogluon":
        return payload

    predictor_path_raw = str(payload.get("predictor_path", "") or "").strip()
    if not predictor_path_raw and payload.get("predictor") is not None:
        predictor_path_raw = str(getattr(payload.get("predictor"), "path", "") or "").strip()

    if not predictor_path_raw:
        raise ValueError("AutoGluon payload missing predictor_path")

    artifact_dir = Path(predictor_path_raw).name
    if not artifact_dir:
        raise ValueError(f"Invalid AutoGluon predictor_path in payload: {predictor_path_raw}")

    remote_prefix = f"output/{user_id}/{session_id}/{artifact_dir}"
    local_predictor_dir = OUTPUT_ROOT / user_id / artifact_dir
    marker_file = local_predictor_dir / "learner.pkl"
    if not marker_file.exists():
        local_predictor_dir.mkdir(parents=True, exist_ok=True)
        _download_prefix_to_local(remote_prefix, local_predictor_dir)

    normalized_payload = dict(payload)
    normalized_payload["predictor_path"] = str(local_predictor_dir)
    normalized_payload["predictor"] = None
    return normalized_payload


def _load_session_artifacts(
    user_id: str,
    session_id: str,
    dataset_base: str,
    load_model: bool = True,
) -> Tuple[Dict, Any, Dict, Optional[Dict]]:
    """Load model pkl, features metadata, and profiling metadata from Supabase.

    Returns
    -------
    payload      : raw pkl dict (framework, pipeline, label_encoder, training_stats,
                   profiling_meta [embedded since fix], ...)
    model        : fitted model object (None when load_model=False)
    features_meta: features JSON dict
    profile_meta : profiling JSON dict, or None if not available
                   ← Returns None (not {}) when loading fails.
                   An empty dict {} was the silent root cause of the label-flip
                   bug: _build_type_map built an empty type_map from it and
                   _coerce_input_df applied no coercion at all.

    Priority for profiling_meta:
    1. Embedded in pkl (payload["profiling_meta"]) — always available if model
       was trained with the fixed model_selection.py.
    2. Downloaded from Supabase — fallback for older pkls.
    3. None — _coerce_input_df falls back to training_stats.
    """
    model_key    = f"output/{user_id}/{session_id}/{dataset_base}_model.pkl"
    features_key = f"meta_data/{user_id}/{session_id}/{dataset_base}_features.json"
    profile_key  = f"meta_data/{user_id}/{session_id}/{dataset_base}_profiling.json"

    raw_model = download_file(model_key)
    payload = pickle.loads(raw_model)
    payload = _ensure_local_autogluon_artifacts(payload, user_id=user_id, session_id=session_id)
    features_meta = _load_json_from_supabase(features_key)

    # ── Priority 1: profiling_meta embedded in pkl ─────────────────────────────
    profile_meta: Optional[Dict] = None
    embedded = payload.get("profiling_meta")
    if isinstance(embedded, dict) and embedded.get("column_wise_summary"):
        profile_meta = embedded
    else:
        # ── Priority 2: download from Supabase ─────────────────────────────────
        # ── FIX: return None explicitly when profiling JSON cannot be loaded ───
        # The old code returned {} on failure.  {} passes isinstance(dict) check
        # but is also falsy, causing _build_type_map to produce an empty type_map
        # and _coerce_input_df to skip all dtype coercion → label flip.
        try:
            downloaded = _load_json_from_supabase(profile_key)
            if isinstance(downloaded, dict) and downloaded.get("column_wise_summary"):
                profile_meta = downloaded
            else:
                import logging as _log
                _log.getLogger("ModelTesting").warning(
                    "Profiling JSON loaded from Supabase but missing column_wise_summary "
                    "— treating as unavailable. Falling back to training_stats."
                )
        except Exception:
            pass  # profile_meta stays None; _coerce_input_df uses training_stats

    model = None
    if load_model:
        _framework, model = _load_model(payload)
    return payload, model, features_meta, profile_meta


def get_model_testing_schema_supabase(user_id: str, session_id: str, dataset_base: str) -> Dict[str, Any]:
    """Return UI schema for model testing inputs from session files in Supabase."""
    _payload, _model, features_meta, profile_meta = _load_session_artifacts(
        user_id,
        session_id,
        dataset_base,
        load_model=False,
    )

    # FIX-3a: use payload's selected_features as the AUTHORITATIVE column order.
    # Previously fell back to features_meta when payload returned [] (falsy empty
    # list), which could produce a different column ordering than the pkl pipeline
    # was fitted on — causing columns to map to the wrong transformer.
    payload_features: List[str] = _payload.get("selected_features") or []
    meta_features: List[str] = list(features_meta.get("selected_features", []))
    selected_features: List[str] = payload_features if payload_features else meta_features

    target = str(features_meta.get("target", ""))
    task = str(features_meta.get("task", "classification")).strip().lower()

    if not selected_features:
        raise ValueError("selected_features missing in features metadata")

    training_stats = _payload.get("training_stats") or {}

    # FIX-3b: pass pkl_type_map so _build_type_map uses the embedded type_map
    # as the highest-priority source — no Supabase download required.
    pkl_type_map: Optional[Dict[str, str]] = _payload.get("type_map")
    type_map = _build_type_map(features_meta, profile_meta, pkl_type_map=pkl_type_map)

    cleaned_key = f"output/{user_id}/{session_id}/{dataset_base}_cleaned.csv"
    cleaned_bytes = download_file(cleaned_key)
    df = pd.read_csv(io.BytesIO(cleaned_bytes))

    fields: List[Dict[str, Any]] = []
    for col in selected_features:
        ftype = str(type_map.get(col, features_meta.get("feature_types", {}).get(col, "categorical"))).lower()
        input_type = _field_input_type(ftype)

        field: Dict[str, Any] = {
            "name": col,
            "feature_type": ftype,
            "input_type": input_type,
            "options": [],
        }

        if input_type == "number":
            info = training_stats.get(col, {}) if isinstance(training_stats, dict) else {}
            low = info.get("min")
            high = info.get("max")
            try:
                if low is not None:
                    field["min"] = float(low)
                if high is not None:
                    field["max"] = float(high)
            except (TypeError, ValueError):
                pass

        if input_type == "categorical" and col in df.columns:
            # FIX-3c: always stringify options so they match OHE's string categories_.
            # OHE fits on str values (via _cat_to_object at training). If we return
            # int options [0, 1] from value_counts, the UI sends int 0 in the JSON
            # payload, and if _coerce_input_df somehow doesn't fire (e.g. old pkl),
            # OHE sees int 0 ≠ str "0" → unknown category → wrong prediction.
            vals = (
                df[col]
                .dropna()
                .astype(str)          # ← always str to match OHE's fitted categories_
                .value_counts()
                .head(50)
                .index
                .tolist()
            )
            field["options"] = vals

        if input_type == "categorical":
            if not field["options"]:
                field["options"] = _extract_options_from_column_name(col)

            if ftype == "boolean":
                field["options"] = _coerce_boolean_options(field["options"])

        fields.append(field)

    return {
        "dataset_base": dataset_base,
        "target": target,
        "task": task,
        "default_confidence_threshold_percent": 50.0,
        "selected_features": selected_features,
        "fields": fields,
    }


def get_model_report_supabase(user_id: str, session_id: str, dataset_base: str) -> Dict[str, Any]:
    """Load session-scoped model report from metadata storage."""
    report_key = f"meta_data/{user_id}/{session_id}/{dataset_base}_model_report.json"
    report = _load_json_from_supabase(report_key)

    if not isinstance(report, dict):
        raise ValueError("Invalid model report payload")

    report["session_id"] = session_id
    report["dataset_base"] = report.get("dataset_base", dataset_base)
    return report


def predict_from_session_model(
    user_id: str,
    session_id: str,
    dataset_base: str,
    row: Dict[str, Any],
    confidence_threshold_percent: float = 50.0,
) -> Dict[str, Any]:
    """Run prediction against a session-scoped model stored in Supabase.

    The loaded .pkl pipeline already includes the full ColumnTransformer
    (encoding + scaling) fitted on training data.

    LABEL-FLIP FIX (multi-layer)
    ─────────────────────────────
    The label flip (no-disease → 1, disease → 0) was caused by the pipeline
    receiving wrong dtypes at inference time. At training, _cat_to_object()
    converts categorical columns (e.g. sex, fbs, exang) to str BEFORE OHE
    fits on them, so OHE stores categories_ = ["0", "1"] (strings).

    At inference, JSON deserialization gives Python int 0 / int 1 for these
    fields. If _coerce_input_df did not convert them to str "0"/"1" first,
    OHE received unknown dtype → all-zeros → wrong prediction.

    FIX (this version):
    1. pkl now embeds type_map at training time (model_selection.py fix).
       _build_type_map() uses it as the highest-priority source so coercion
       always fires even if Supabase downloads fail.
    2. _coerce_input_df(): boolean columns that use OHE still receive str (not
       int8), because _cat_to_object always stringifies before OHE fits.
    3. feature_order: payload['selected_features'] is used as the authoritative
       column order. The falsy-empty-list fallback now correctly distinguishes
       between an explicitly empty list (error) and a missing key (fall back).
    """
    _payload, model, features_meta, profile_meta = _load_session_artifacts(user_id, session_id, dataset_base)
    framework = _normalize_framework(_payload)
    if model is None:
        framework, model = _load_model(_payload)

    # FIX-4: use payload selected_features as the authoritative column order.
    # The previous `_payload.get("selected_features") or list(...)` pattern
    # treats an empty list [] as falsy and falls through to features_meta —
    # which may have a different column order, causing wrong transformer mapping.
    payload_features: List[str] = _payload.get("selected_features") or []
    meta_features: List[str] = list(features_meta.get("selected_features", []))
    feature_order: List[str] = payload_features if payload_features else meta_features

    task = str(features_meta.get("task", "classification")).strip().lower()

    try:
        threshold_percent = float(confidence_threshold_percent)
    except (TypeError, ValueError):
        raise ValueError("confidence_threshold_percent must be a number between 0 and 100")

    if not (0.0 <= threshold_percent <= 100.0):
        raise ValueError("confidence_threshold_percent must be between 0 and 100")

    threshold_prob = threshold_percent / 100.0

    missing = [f for f in feature_order if f not in row]
    if missing:
        raise ValueError(f"Missing input values for feature(s): {missing}")

    input_row = {feat: row.get(feat) for feat in feature_order}
    input_df = pd.DataFrame([input_row])
    input_df = input_df[feature_order]
    raw_input_df = input_df.copy(deep=True)

    # FIX: build fully-resolved type_map with pkl_type_map as highest priority.
    # pkl_type_map is embedded at training time in model_selection.py and is
    # guaranteed consistent with the fitted preprocessor. It works without any
    # Supabase dependency. Layers 2-4 in _build_type_map are kept as fallbacks
    # for pkls created before this fix was deployed.
    pkl_type_map: Optional[Dict[str, str]] = _payload.get("type_map")
    type_map = _build_type_map(features_meta, profile_meta, pkl_type_map=pkl_type_map)

    # Secondary safety net: if type_map is still incomplete (e.g. very old pkl
    # predating both fixes), fill gaps from training_stats in pkl.
    training_stats = _payload.get("training_stats") or {}
    for feat in feature_order:
        if feat not in type_map or not type_map[feat]:
            stype = str(training_stats.get(feat, {}).get("stype", "")).lower()
            if stype:
                type_map[feat] = stype

    import logging as _log
    _logger = _log.getLogger("ModelTesting")
    _logger.info(
        "predict_from_session_model | type_map resolved for %d/%d features | "
        "pkl_type_map=%s | profile_meta=%s | training_stats=%s",
        sum(1 for f in feature_order if type_map.get(f)),
        len(feature_order),
        "available" if pkl_type_map else "unavailable (old pkl)",
        "available" if profile_meta else "unavailable",
        "available" if training_stats else "unavailable",
    )

    input_df = _coerce_input_df(input_df, feature_order, type_map, training_stats=training_stats)

    warnings = _collect_prediction_warnings(
        raw_df=raw_input_df,
        coerced_df=input_df,
        selected_features=feature_order,
        type_map=type_map,
        training_stats=training_stats,
    )

    _validate_input_ranges(input_df, training_stats=training_stats)

    label_encoder = _payload.get("label_encoder")

    pred, prob = _predict(framework, model, input_df, task, label_encoder=label_encoder)

    meets_threshold: Optional[bool] = None
    if task == "classification" and prob is not None:
        prob_value = float(prob)
        meets_threshold = prob_value >= threshold_prob
        if not meets_threshold:
            warnings.append(
                f"Model confidence {(prob_value * 100.0):.2f}% is below threshold {threshold_percent:.2f}%."
            )

    warnings = list(dict.fromkeys(warnings))

    return {
        "dataset_base": dataset_base,
        "task": task,
        "prediction": pred if isinstance(pred, (int, float, str, bool)) else str(pred),
        "probability": float(prob) if prob is not None else None,
        "confidence_threshold_percent": threshold_percent,
        "meets_confidence_threshold": meets_threshold,
        "warnings": warnings,
    }


def predict_batch_from_session_model(
    user_id: str,
    session_id: str,
    dataset_base: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run predictions against a session-scoped model for multiple input rows."""
    if not rows:
        raise ValueError("No input rows provided.")

    _payload, model, features_meta, profile_meta = _load_session_artifacts(user_id, session_id, dataset_base)
    framework = _normalize_framework(_payload)
    if model is None:
        framework, model = _load_model(_payload)

    payload_features: List[str] = _payload.get("selected_features") or []
    meta_features: List[str] = list(features_meta.get("selected_features", []))
    feature_order: List[str] = payload_features if payload_features else meta_features

    if not feature_order:
        raise ValueError("Selected features missing for this model.")

    task = str(features_meta.get("task", "classification")).strip().lower()

    extra_columns = set()
    sanitized_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = [f for f in feature_order if f not in row]
        if missing:
            raise ValueError(f"Missing input values in row {idx}: {missing}")

        extra = [k for k in row.keys() if k not in feature_order]
        extra_columns.update(extra)

        sanitized_rows.append({feat: row.get(feat) for feat in feature_order})

    input_df = pd.DataFrame(sanitized_rows)
    input_df = input_df[feature_order]
    raw_input_df = input_df.copy(deep=True)

    pkl_type_map: Optional[Dict[str, str]] = _payload.get("type_map")
    type_map = _build_type_map(features_meta, profile_meta, pkl_type_map=pkl_type_map)

    training_stats = _payload.get("training_stats") or {}
    for feat in feature_order:
        if feat not in type_map or not type_map[feat]:
            stype = str(training_stats.get(feat, {}).get("stype", "")).lower()
            if stype:
                type_map[feat] = stype

    input_df = _coerce_input_df(input_df, feature_order, type_map, training_stats=training_stats)

    label_encoder = _payload.get("label_encoder")

    predictions: List[Any] = []
    probabilities: List[Optional[float]] = []
    row_warnings: List[List[str]] = []
    for i in range(len(input_df)):
        row_df = input_df.iloc[[i]]
        raw_row_df = raw_input_df.iloc[[i]]
        row_warnings.append(
            _collect_prediction_warnings(
                raw_df=raw_row_df,
                coerced_df=row_df,
                selected_features=feature_order,
                type_map=type_map,
                training_stats=training_stats,
            )
        )
        pred, prob = _predict(framework, model, row_df, task, label_encoder=label_encoder)
        predictions.append(pred if isinstance(pred, (int, float, str, bool)) else str(pred))
        probabilities.append(float(prob) if prob is not None else None)

    return {
        "dataset_base": dataset_base,
        "task": task,
        "feature_order": feature_order,
        "rows": sanitized_rows,
        "predictions": predictions,
        "probabilities": probabilities,
        "row_warnings": row_warnings,
        "extra_columns": sorted(extra_columns),
    }


def _prob_for_pred(proba_row: np.ndarray, pred, classes: list) -> float:
    """Return the probability for the predicted class.

    ``predict_proba`` returns probabilities in the order of ``model.classes_``,
    which is NOT necessarily [0, 1, 2, …].  The only correct mapping is:
    probability = proba_row[classes_.index(pred)].
    """
    pred_native = pred.item() if isinstance(pred, np.generic) else pred

    if classes:
        if pred_native in classes:
            return float(proba_row[classes.index(pred_native)])

        str_classes = [str(c) for c in classes]
        str_pred = str(pred_native)
        if str_pred in str_classes:
            return float(proba_row[str_classes.index(str_pred)])

    try:
        idx = int(pred_native)
        if 0 <= idx < len(proba_row):
            return float(proba_row[idx])
    except (TypeError, ValueError):
        pass

    return float(np.max(proba_row))


def _extract_final_estimator(model):
    """Walk a sklearn Pipeline to its last step (the actual classifier/regressor)."""
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def _predict(framework: str, model, df: pd.DataFrame, task: str, label_encoder=None):
    """Run inference and return (decoded_prediction, confidence).

    Parameters
    ----------
    framework : str
        "sklearn" | "autogluon" | "flaml"
    model :
        Fitted pipeline / predictor object.
    df : pd.DataFrame
        Input row — dtype-coerced via _coerce_input_df() before this call.
    task : str
        "classification" | "regression"
    label_encoder : LabelEncoder | None
        Non-None only for XGBClassifier / LGBMClassifier. Used to decode the
        integer prediction back to the original class label.
    """

    if framework == "sklearn":
        raw_pred = model.predict(df)[0]

        if task == "classification":
            final_est = _extract_final_estimator(model)
            classes: list = list(getattr(final_est, "classes_", []))

            if label_encoder is not None:
                try:
                    pred = label_encoder.inverse_transform([int(raw_pred)])[0]
                except Exception:
                    pred = raw_pred
            else:
                pred = raw_pred

            if hasattr(model, "predict_proba"):
                proba_row = model.predict_proba(df)[0]
                if label_encoder is not None:
                    le_classes = list(label_encoder.classes_)
                    confidence = _prob_for_pred(proba_row, pred, le_classes)
                else:
                    confidence = _prob_for_pred(proba_row, pred, classes)
                return pred, confidence

            if hasattr(model, "decision_function"):
                scores_arr = np.asarray(model.decision_function(df))

                if scores_arr.ndim == 1 or (scores_arr.ndim == 2 and scores_arr.shape[1] == 1):
                    margin = float(scores_arr.ravel()[0])
                    pos_prob = 1.0 / (1.0 + np.exp(-margin))

                    ref_classes = list(label_encoder.classes_) if label_encoder else classes
                    if len(ref_classes) == 2:
                        pred_native = pred.item() if isinstance(pred, np.generic) else pred
                        if pred_native in ref_classes:
                            pred_idx = ref_classes.index(pred_native)
                            pred_prob = pos_prob if pred_idx == 1 else (1.0 - pos_prob)
                            return pred, float(pred_prob)
                    return pred, float(max(pos_prob, 1.0 - pos_prob))

                row_scores = scores_arr[0] if scores_arr.ndim == 2 else scores_arr
                row_scores = row_scores - np.max(row_scores)
                exp_scores = np.exp(row_scores)
                probs = exp_scores / np.sum(exp_scores)
                ref_classes = list(label_encoder.classes_) if label_encoder else classes
                confidence = _prob_for_pred(probs, pred, ref_classes)
                return pred, confidence

        return raw_pred, None

    elif framework == "autogluon":
        pred = model.predict(df).iloc[0]
        pred_native = pred.item() if isinstance(pred, np.generic) else pred

        if task == "classification":
            proba_df = model.predict_proba(df)

            if pred_native in proba_df.columns:
                return pred_native, float(proba_df[pred_native].iloc[0])

            str_pred = str(pred_native)
            str_cols = {str(c): c for c in proba_df.columns}
            if str_pred in str_cols:
                return pred_native, float(proba_df[str_cols[str_pred]].iloc[0])

            return pred_native, float(proba_df.max(axis=1).iloc[0])

        return pred_native, None

    elif framework == "flaml":
        preds = model.predict(df)
        raw_pred = preds[0] if isinstance(preds, (list, tuple, np.ndarray, pd.Series)) else preds
        pred_native = raw_pred.item() if isinstance(raw_pred, np.generic) else raw_pred

        if task == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)

                if isinstance(proba, pd.DataFrame):
                    if pred_native in proba.columns:
                        return pred_native, float(proba[pred_native].iloc[0])
                    str_pred = str(pred_native)
                    str_cols = {str(c): c for c in proba.columns}
                    if str_pred in str_cols:
                        return pred_native, float(proba[str_cols[str_pred]].iloc[0])
                    return pred_native, float(proba.max(axis=1).iloc[0])

                proba_arr = np.asarray(proba)
                if proba_arr.ndim == 2 and proba_arr.shape[1] > 0:
                    classes = list(getattr(model, "classes_", []))
                    confidence = _prob_for_pred(proba_arr[0], pred_native, classes)
                    return pred_native, confidence

            except Exception:
                pass

        return pred_native, None


def main():
    print("✅ Fixed Dynamic Inference")

    user_id = input("Enter user_id: ").strip()
    if not user_id:
        print("Error: user_id cannot be empty")
        return

    user_output_dir = OUTPUT_ROOT / user_id
    user_meta_dir = META_ROOT / user_id
    if not user_output_dir.exists() or not user_meta_dir.exists():
        print(f"Error: path not found. output={user_output_dir} meta={user_meta_dir}")
        return

    try:
        model_path, payload, features_path, profile_path, dataset_base = _pick_first_loadable_model(
            user_output_dir=user_output_dir,
            user_meta_dir=user_meta_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return

    features_meta = _load_json(features_path)
    framework, model = _load_model(payload)

    # FIX-5a: use payload's selected_features as the authoritative order.
    payload_features: List[str] = payload.get("selected_features") or []
    meta_features: List[str] = list(features_meta.get("selected_features", []))
    selected_features: List[str] = payload_features if payload_features else meta_features

    task = features_meta["task"]

    # Load profiling metadata for dtype coercion.
    profile_meta: Optional[Dict] = None
    if profile_path and profile_path.exists():
        profile_meta = _load_json(profile_path)

    row = _collect_input(selected_features)
    input_df = pd.DataFrame([row])
    input_df = input_df[selected_features]

    # FIX-5b: pass pkl_type_map as highest-priority source — eliminates need
    # for live profiling JSON at inference time. Works fully offline / in tests.
    pkl_type_map: Optional[Dict[str, str]] = payload.get("type_map")
    type_map = _build_type_map(features_meta, profile_meta, pkl_type_map=pkl_type_map)

    training_stats = payload.get("training_stats") or {}
    for feat in selected_features:
        if feat not in type_map or not type_map[feat]:
            stype = str(training_stats.get(feat, {}).get("stype", "")).lower()
            if stype:
                type_map[feat] = stype

    input_df = _coerce_input_df(input_df, selected_features, type_map, training_stats=training_stats)

    _validate_input_ranges(input_df, training_stats=training_stats)

    label_encoder = payload.get("label_encoder")

    pred, prob = _predict(framework, model, input_df, task, label_encoder=label_encoder)

    print("\n===== RESULT =====")
    print(f"Prediction: {pred}")
    print(f"Probability: {prob}")

    print("\n[DEBUG] Input dtypes after coercion:")
    print(input_df.dtypes.to_string())
    print("[DEBUG] Input values after coercion:")
    print(input_df.iloc[0].to_string())


if __name__ == "__main__":
    main()