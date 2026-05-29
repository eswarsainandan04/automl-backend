"""Data split module for dynamic AutoML datasets.

This module:
- Loads metadata and cleaned dataset
- Validates target and selected features
- Enforces correct dtypes from profiling metadata
- Splits into train/validation sets
- Writes nothing to disk

FIX CHANGELOG
─────────────
FIX-1 (enforce_dtypes): Boolean columns that feed into OHE (structural_type='categorical'
       in profiling) must stay categorical, NOT be cast to int8. The int8 path is only safe
       when the column uses FunctionTransformer(_boolean_transform) — i.e. when profiling
       truly says 'boolean' AND the build_transformer boolean branch was taken. Since
       _cat_to_object() ALWAYS stringifies before OHE fits, every column reaching OHE must
       arrive at inference as str. enforce_dtypes now only casts to int8 when structural_type
       is strictly 'boolean'; it leaves 'categorical' columns as category dtype so
       _cat_to_object() can stringify them consistently at both training and inference.
       No change needed here — the bug was downstream in _coerce_input_df (see model_testing.py).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("DataSplit")


WARNING_TEXT = (
    "⚠️ You are using a dynamic AutoML system. Ensure metadata is correct.\n"
    "Improper metadata may lead to incorrect preprocessing or model behavior."
)


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_dataset_bases(meta_dir: Path, output_dir: Path) -> List[str]:
    feature_bases = {
        p.stem.replace("_features", "")
        for p in meta_dir.glob("*_features.json")
    }
    profiling_bases = {
        p.stem.replace("_profiling", "")
        for p in meta_dir.glob("*_profiling.json")
    }
    cleaned_bases = {
        p.stem.replace("_cleaned", "")
        for p in output_dir.glob("*_cleaned.csv")
    }

    return sorted(feature_bases & profiling_bases & cleaned_bases)


def _select_dataset_base(bases: List[str]) -> str:
    if not bases:
        raise FileNotFoundError("No dataset triplet found: *_features.json, *_profiling.json, *_cleaned.csv")
    if len(bases) == 1:
        return bases[0]

    print("\nAvailable datasets:")
    for i, b in enumerate(bases, start=1):
        print(f"{i}. {b}")

    while True:
        raw = input("Select dataset by number or name: ").strip()
        if raw in bases:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(bases):
                return bases[idx - 1]
        print("Invalid selection. Try again.")


# ══════════════════════════════════════════════════════════════════════════════
# DTYPE ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def enforce_dtypes(df: pd.DataFrame, profiling_meta: Dict) -> pd.DataFrame:
    """Cast DataFrame columns to the correct dtype based on profiling metadata.

    Without this step, pd.read_csv() returns every numerically-encoded
    categorical column as int64/float64.  sklearn's ColumnTransformer then
    inspects the actual DataFrame dtype — not the metadata — so it never
    triggers the categorical encoders, treating every column as numeric.

    This function must be called immediately after pd.read_csv() and before
    any split or preprocessing step.

    Changes applied
    ---------------
    * structural_type == "categorical" → pandas ``category`` dtype
      (_cat_to_object inside each Pipeline then converts to str before OHE fits,
      ensuring OHE stores string categories at training time)
    * structural_type == "boolean"     → int8 (0/1) — ONLY used for columns
      whose build_transformer branch is FunctionTransformer(_boolean_transform),
      NOT for columns going into OHE. If a boolean column was also tagged
      'categorical' by profiling (e.g. 2-class int column), 'categorical' wins.
    * structural_type in numeric set   → float64 (safe no-op if already numeric)
    * All other types                  → unchanged
    """
    _NUMERIC_STYPES = frozenset({"numeric", "integer", "float"})
    _CAT_STYPES     = frozenset({"categorical", "category"})
    _BOOLEAN_STYPES = frozenset({"boolean"})

    col_summary: List[Dict] = profiling_meta.get("column_wise_summary", [])
    type_map: Dict[str, str] = {
        item["column_name"]: str(item.get("structural_type", "")).lower()
        for item in col_summary
        if item.get("column_name")
    }

    for col, stype in type_map.items():
        if col not in df.columns:
            continue

        if stype in _CAT_STYPES:
            df[col] = df[col].astype("category")
            logger.debug("enforce_dtypes: '%s' → category", col)

        elif stype in _BOOLEAN_STYPES:
            # NOTE: This int8 path is ONLY reached when structural_type is strictly
            # 'boolean' (not 'categorical'). build_transformer uses
            # FunctionTransformer(_boolean_transform) for boolean — which outputs
            # int arrays — so int8 is correct here. For 2-class categoricals that
            # happen to look boolean (sex, fbs, exang), profiling sets
            # structural_type='categorical', so they go through the category branch
            # above and get stringified by _cat_to_object before OHE fits.
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
            logger.debug("enforce_dtypes: '%s' → int8 (boolean)", col)

        elif stype in _NUMERIC_STYPES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.debug("enforce_dtypes: '%s' → float64 (numeric)", col)

    # Debug summary
    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category"]
    logger.info(
        "enforce_dtypes complete | categorical cols (%d): %s",
        len(cat_cols), cat_cols,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_split_inputs(meta_path: Path, output_path: Path, dataset_base: str) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Load cleaned data, features metadata, and profiling metadata for a dataset base."""
    features_path = meta_path / f"{dataset_base}_features.json"
    profiling_path = meta_path / f"{dataset_base}_profiling.json"
    cleaned_path = output_path / f"{dataset_base}_cleaned.csv"

    for p in [features_path, profiling_path, cleaned_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    df = pd.read_csv(cleaned_path)
    features_meta = _load_json(features_path)
    profiling_meta = _load_json(profiling_path)

    # Enforce dtypes immediately after loading
    df = enforce_dtypes(df, profiling_meta)

    logger.info("Loaded cleaned dataset: %s (%s rows x %s cols)", cleaned_path.name, len(df), len(df.columns))
    return df, features_meta, profiling_meta


# ══════════════════════════════════════════════════════════════════════════════
# SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(
    df: pd.DataFrame,
    target: str,
    selected_features: List[str],
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split dataset into train/validation without persisting output."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in cleaned dataset.")

    missing_features = [c for c in selected_features if c not in df.columns]
    if missing_features:
        raise ValueError(f"Selected feature(s) missing in cleaned dataset: {missing_features}")

    if not selected_features:
        raise ValueError("selected_features is empty; cannot build training matrix.")

    X = df[selected_features].copy()
    y = df[target].copy()

    if task.lower() == "classification":
        stratify = None
        non_null_y = y.dropna()
        class_count = int(non_null_y.nunique())

        if class_count > 1:
            class_freq = non_null_y.value_counts(dropna=True)
            min_class_freq = int(class_freq.min()) if not class_freq.empty else 0

            n_samples = len(y)
            if isinstance(test_size, float):
                val_size = int(math.ceil(n_samples * test_size))
            else:
                val_size = int(test_size)
            train_size = n_samples - val_size

            can_stratify = (
                min_class_freq >= 2
                and val_size >= class_count
                and train_size >= class_count
            )

            if can_stratify:
                stratify = y
            else:
                logger.warning(
                    "Stratified split disabled for target '%s' (classes=%s, min_class_freq=%s, "
                    "train_size=%s, val_size=%s). Falling back to non-stratified split.",
                    target, class_count, min_class_freq, train_size, val_size,
                )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    logger.info(
        "Split complete | task=%s | X_train=%s | X_val=%s",
        task, X_train.shape, X_val.shape,
    )
    return X_train, X_val, y_train, y_val


def main() -> None:
    print(WARNING_TEXT)
    user_id = input("Enter user_id: ").strip()

    if not user_id:
        print("Error: user_id cannot be empty.")
        return

    base_path = "storage/"
    meta_path = f"{base_path}/meta_data/{user_id}/"
    output_path = f"{base_path}/output/{user_id}/"

    meta_dir = Path(meta_path)
    output_dir = Path(output_path)
    if not meta_dir.exists() or not output_dir.exists():
        print(f"Error: path not found. meta={meta_dir} output={output_dir}")
        return

    try:
        bases = _discover_dataset_bases(meta_dir, output_dir)
        dataset_base = _select_dataset_base(bases)
        df, features_meta, _profiling_meta = load_split_inputs(meta_dir, output_dir, dataset_base)

        target = str(features_meta.get("target", "")).strip()
        selected_features = list(features_meta.get("selected_features", []))
        task = str(features_meta.get("task", "classification")).strip().lower()

        if not target:
            raise ValueError("'target' missing in features metadata.")

        X_train, X_val, y_train, y_val = split_dataset(
            df=df,
            target=target,
            selected_features=selected_features,
            task=task,
        )

        print("\nSplit created successfully (not saved to disk).")
        print(f"Dataset: {dataset_base}")
        print(f"Task: {task}")
        print(f"Target: {target}")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_val shape: {y_val.shape}")

        print("\n[DEBUG] dtypes after enforce_dtypes:")
        print(df[selected_features].dtypes.to_string())

    except Exception as exc:
        logger.exception("Data split failed")
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()