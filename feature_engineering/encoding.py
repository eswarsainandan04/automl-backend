"""
Production-Grade AutoML Encoding Engine
Applies structural-type-based encoding to preprocessed DataFrames.
Uses AutoGluon feature generators where compatible, with sklearn fallbacks.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# AutoGluon availability
try:
    from autogluon.features.generators import (
        CategoryFeatureGenerator,
        DatetimeFeatureGenerator,
        AutoMLPipelineFeatureGenerator,
    )
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
#  Column-Name Resolver
# ──────────────────────────────────────────────────────────────

def _normalize_col_name(name: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .strip("_")
    )


def _resolve_column_map(
    profiling_columns: List[Dict[str, Any]],
    df_columns: List[str],
) -> Dict[str, str]:
    """
    Build a map  profiling_column_name → actual_df_column_name.
    Handles renaming that may have happened during earlier pipeline steps
    (e.g. ``is_active_(true/false)`` in JSON  →  ``is_active`` in CSV).
    """
    df_norm = {_normalize_col_name(c): c for c in df_columns}
    mapping: Dict[str, str] = {}

    for col_meta in profiling_columns:
        prof_name = col_meta["column_name"]

        # Exact match
        if prof_name in df_columns:
            mapping[prof_name] = prof_name
            continue

        # Normalized match
        norm = _normalize_col_name(prof_name)
        if norm in df_norm:
            mapping[prof_name] = df_norm[norm]
            continue

        # Prefix / substring match (longest match wins)
        best, best_len = None, 0
        for df_col in df_columns:
            n = _normalize_col_name(df_col)
            if norm.startswith(n) or n.startswith(norm):
                if len(n) > best_len:
                    best, best_len = df_col, len(n)
        if best:
            mapping[prof_name] = best

    return mapping


# ──────────────────────────────────────────────────────────────
#  Encoding Engine
# ──────────────────────────────────────────────────────────────

class EncodingEngine:
    """
    Structural-type-driven encoding engine for AutoML preprocessing.

    Workflow
    -------
    1. ``fit(df, profiling_json)``   – learn encoders from training data
    2. ``transform(df)``             – apply learned encoders
    3. ``fit_transform(df, json)``   – convenience shortcut

    Structural-type → action:
        identifier  → drop
        numeric     → pass through (no scaling)
        categorical → encode (OneHot / Frequency)
        text        → TF-IDF vectorize
        datetime    → extract year / month / day / weekday
        boolean     → keep as int (0 / 1)
        unknown     → keep as-is
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.fitted = False

        # Fitted objects  ─  stored per column for reuse in ``transform``
        self.fitted_objects: Dict[str, Dict] = {
            "encoders": {},
            "vectorizers": {},
        }

        # Bookkeeping
        self._column_plan: List[Dict[str, Any]] = []
        self._identifier_cols: List[str] = []
        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._text_cols: List[str] = []
        self._datetime_cols: List[str] = []
        self._boolean_cols: List[str] = []
        self._unknown_cols: List[str] = []
        self._time_cols: List[str] = []
        self._date_cols: List[str] = []
        # Encoding summary (written to profiling JSON)
        self.encoding_summary: Dict[str, Any] = {}

    # ──────────────────────  public API  ──────────────────────

    def fit(self, df: pd.DataFrame, profiling_json: Dict[str, Any]) -> "EncodingEngine":
        """
        Learn all encoders from *training* data.
        No data leakage: nothing from ``transform`` leaks back.
        """
        self._build_column_plan(df, profiling_json)
        self._fit_categorical(df)
        self._fit_text(df)
        # datetime & boolean & identifier need no fitting
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted encoders to a DataFrame."""
        if not self.fitted:
            raise RuntimeError("EncodingEngine has not been fitted. Call fit() first.")

        df = df.copy()

        # 1. Drop identifiers
        df = self._transform_identifiers(df)

        # 1b. Drop unknown columns
        df = self._transform_unknowns(df)

        # 2b. Date feature extraction
        df = self._transform_date(df)

        # 2c. Time feature extraction
        df = self._transform_time(df)

        # 2. Datetime feature extraction
        df = self._transform_datetime(df)

        # 3. Text → TF-IDF
        df = self._transform_text(df)

        # 4. Categorical encoding
        df = self._transform_categorical(df)

        # 5. Numeric → pass through / drop high-uniqueness
        df = self._passthrough_numeric(df)

        # 6. Boolean → int
        df = self._transform_boolean(df)

        # 7. Final cleanup – fill remaining NaN in numeric cols
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)

        return df

    def fit_transform(self, df: pd.DataFrame, profiling_json: Dict[str, Any]) -> pd.DataFrame:
        """Convenience: fit + transform in one call."""
        self.fit(df, profiling_json)
        return self.transform(df)

    # ──────────────────────  plan builder  ────────────────────

    def _build_column_plan(self, df: pd.DataFrame, profiling_json: Dict[str, Any]):
        """Read structural_type from profiling JSON and map to df columns."""
        columns_meta = profiling_json.get("column_wise_summary", [])
        total_rows = profiling_json.get("number_of_rows", len(df))
        col_map = _resolve_column_map(columns_meta, df.columns.tolist())

        # Reset lists
        for lst in (
            self._identifier_cols,
            self._numeric_cols,
            self._categorical_cols,
            self._text_cols,
            self._datetime_cols,
            self._date_cols,      # ✅ Added
            self._time_cols,      # ✅ Added
            self._boolean_cols,
            self._unknown_cols,
            self._column_plan,
        ):
            lst.clear()

        for col_meta in columns_meta:
            prof_name = col_meta["column_name"]
            df_col = col_map.get(prof_name)

            if df_col is None or df_col not in df.columns:
                continue

            structural = col_meta.get("structural_type", "unknown")

            entry = {
                "profiling_name": prof_name,
                "df_col": df_col,
                "structural_type": structural,
                "semantic_type": col_meta.get("semantic_type", ""),
                "null_percentage": col_meta.get("null_percentage", 0.0),
                "unique_count": col_meta.get("unique_count", 0),
                "total_rows": total_rows,
            }

            self._column_plan.append(entry)

            {
                "identifier": self._identifier_cols,
                "numeric": self._numeric_cols,
                "categorical": self._categorical_cols,
                "text": self._text_cols,
                "datetime": self._datetime_cols,
                "date": self._date_cols,
                "time": self._time_cols,
                "boolean": self._boolean_cols,
            }.get(structural, self._unknown_cols).append(df_col)

        # Track unknown columns for dropping
        self._unknown_cols = [
            entry["df_col"]
            for entry in self._column_plan
            if entry["structural_type"] == "unknown"
        ]
        
        
        
    def _transform_unknowns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._unknown_cols if c in df.columns]
        for col in cols:
            self.encoding_summary[col] = {
                "method": "Dropped",
                "reason": "structural_type = unknown",
            }
        if cols:
            df = df.drop(columns=cols)
        return df
    # ──────────────────  IDENTIFIER (drop)  ───────────────────

    def _transform_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self._identifier_cols if c in df.columns]
        for col in cols:
            self.encoding_summary[col] = {
                "method": "Dropped",
                "reason": "structural_type = identifier",
            }
        if cols:
            df = df.drop(columns=cols)
        return df

    # ──────────────────  NUMERIC (pass through)  ─────────────

    # Semantic types that represent real measurements / features.
    # These should NEVER be dropped even if uniqueness is high.
    _FEATURE_SEMANTIC_TYPES = frozenset({
        "currency", "salary", "percentage", "float",
        "distance", "weight", "volume", "area", "speed",
        "temperature", "pressure", "energy", "power",
        "capacity", "density", "angle",
        "latitude", "longitude", "geo_coordinate",
        "duration", "timestamp", "boolean", "version", "ratio",
    })

    def _passthrough_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Numeric policy:
        - NO scaling
        - NO normalization
        - Drop identifier-like numeric columns
        - Force numeric dtype
        """

        cols_to_drop = []

        for entry in self._column_plan:
            if entry["structural_type"] != "numeric":
                continue

            col = entry["df_col"]
            if col not in df.columns:
                continue

            # Force numeric cast ONLY
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop identifier-like numerics (phone, account, GST)
            total_rows = entry.get("total_rows", len(df))
            unique_count = entry.get("unique_count", df[col].nunique())
            uniqueness_ratio = unique_count / total_rows if total_rows else 0

            digit_len = (
                df[col]
                .dropna()
                .astype(int, errors="ignore")
                .astype(str)
                .str.len()
            )

            if uniqueness_ratio > 0.9 and digit_len.median() >= 10:
                cols_to_drop.append(col)
                self.encoding_summary[col] = {
                    "method": "Dropped",
                    "reason": "Identifier-like numeric (high uniqueness, long digits)",
                }
            else:
                self.encoding_summary[col] = {
                    "method": "NumericPassThrough",
                    "scaled": False,
                }

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    # ──────────────────  CATEGORICAL (encode)  ────────────────

    def _fit_categorical(self, df: pd.DataFrame):
        for entry in self._column_plan:
            if entry["structural_type"] != "categorical":
                continue
            col = entry["df_col"]
            if col not in df.columns:
                continue

            unique = entry["unique_count"]

            if unique <= 10:
                # One-hot: store the known categories
                categories = sorted(df[col].dropna().unique().tolist())
                self.fitted_objects["encoders"][col] = {
                    "type": "onehot",
                    "categories": categories,
                }
                self.encoding_summary[col] = {
                    "method": "OneHotEncoding",
                    "new_columns": [f"{col}_{cat}" for cat in categories[1:]],
                    "reason": f"structural_type = categorical, unique_count = {unique}",
                }

            elif unique <= 100:
                # Frequency encoding (target encoding requires y – not available here)
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self.fitted_objects["encoders"][col] = {
                    "type": "frequency",
                    "freq_map": freq_map,
                }
                self.encoding_summary[col] = {
                    "method": "FrequencyEncoding",
                    "reason": f"structural_type = categorical, unique_count = {unique}",
                }

            else:
                # >100 unique  →  frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self.fitted_objects["encoders"][col] = {
                    "type": "frequency",
                    "freq_map": freq_map,
                }
                self.encoding_summary[col] = {
                    "method": "FrequencyEncoding",
                    "reason": f"structural_type = categorical, unique_count = {unique} (high cardinality)",
                }

    def _transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, enc_info in self.fitted_objects["encoders"].items():
            if col not in df.columns:
                continue

            if enc_info["type"] == "onehot":
                dummies = pd.get_dummies(
                    df[col], prefix=col, prefix_sep="_", drop_first=True, dtype=int
                )
                # Ensure columns from fit are present
                expected = [f"{col}_{cat}" for cat in enc_info["categories"][1:]]
                for ec in expected:
                    if ec not in dummies.columns:
                        dummies[ec] = 0
                dummies = dummies[[c for c in expected if c in dummies.columns]]

                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

                # Update summary with actual new columns
                self.encoding_summary[col]["new_columns"] = list(dummies.columns)

            elif enc_info["type"] == "frequency":
                freq_map = enc_info["freq_map"]
                df[col] = df[col].map(freq_map).fillna(0.0)

        return df

    # ──────────────────  TEXT (TF-IDF)  ───────────────────────

    def _fit_text(self, df: pd.DataFrame):
        for entry in self._column_plan:
            if entry["structural_type"] != "text":
                continue
            col = entry["df_col"]
            if col not in df.columns:
                continue

            text_data = df[col].fillna("").astype(str)

            vectorizer = TfidfVectorizer(
                max_features=100,
                min_df=1,
                ngram_range=(1, 2),
                strip_accents="unicode",
                lowercase=True,
            )
            vectorizer.fit(text_data)

            self.fitted_objects["vectorizers"][col] = vectorizer

    def _transform_text(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, vectorizer in self.fitted_objects["vectorizers"].items():
            if col not in df.columns:
                continue

            text_data = df[col].fillna("").astype(str)
            tfidf_matrix = vectorizer.transform(text_data)
            feature_names = [
                f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])
            ]
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(), columns=feature_names, index=df.index
            )

            df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

            self.encoding_summary[col] = {
                "method": "TF-IDF",
                "max_features": 100,
                "features_created": len(feature_names),
                "new_columns": feature_names,
                "reason": "structural_type = text",
            }

        return df

    # ──────────────────  DATETIME (extract)  ──────────────────

    def _transform_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._datetime_cols:
            if col not in df.columns:
                continue

            # Detect Unix timestamps (numeric columns)
            if pd.api.types.is_numeric_dtype(df[col]):
                dt_series = pd.to_datetime(df[col], unit="s", errors="coerce")
            else:
                dt_series = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

                # Retry with dayfirst if many NaTs
                if dt_series.isna().sum() > len(dt_series) * 0.5:
                    dt_series = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

            new_cols = {
                f"{col}_year": dt_series.dt.year,
                f"{col}_month": dt_series.dt.month,
                f"{col}_day": dt_series.dt.day,
                f"{col}_weekday": dt_series.dt.dayofweek,
                f"{col}_hour": dt_series.dt.hour,
                f"{col}_minute": dt_series.dt.minute,
            }

            for name, series in new_cols.items():
                df[name] = series.fillna(0).astype(int)

            df = df.drop(columns=[col])

            self.encoding_summary[col] = {
                "method": "DatetimeFeatureExtraction",
                "features_extracted": list(new_cols.keys()),
                "reason": "structural_type = datetime",
            }

        return df

    # ──────────────────  DATE (extract)  ──────────────────

    def _transform_date(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._date_cols:
            if col not in df.columns:
                continue

            dt_series = pd.to_datetime(df[col], errors="coerce")

            new_cols = {
                f"{col}_year": dt_series.dt.year,
                f"{col}_month": dt_series.dt.month,
                f"{col}_day": dt_series.dt.day,
                f"{col}_weekday": dt_series.dt.dayofweek,
                f"{col}_is_weekend": dt_series.dt.dayofweek.isin([5, 6]).astype(int),
            }

            for name, series in new_cols.items():
                df[name] = series.fillna(0).astype(int)

            df = df.drop(columns=[col])

            self.encoding_summary[col] = {
                "method": "DateFeatureExtraction",
                "features_extracted": list(new_cols.keys()),
                "reason": "structural_type = date",
            }

        return df

    # ──────────────────  TIME (extract)  ──────────────────

    def _transform_time(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._time_cols:
            if col not in df.columns:
                continue

            time_series = pd.to_datetime(df[col], errors="coerce")

            new_cols = {
                f"{col}_hour": time_series.dt.hour,
                f"{col}_minute": time_series.dt.minute,
                f"{col}_second": time_series.dt.second,
                f"{col}_is_morning": time_series.dt.hour.between(5, 11).astype(int),
                f"{col}_is_afternoon": time_series.dt.hour.between(12, 17).astype(int),
                f"{col}_is_evening": time_series.dt.hour.between(18, 22).astype(int),
                f"{col}_is_night": time_series.dt.hour.between(23, 4).astype(int),
            }

            for name, series in new_cols.items():
                df[name] = series.fillna(0).astype(int)

            df = df.drop(columns=[col])

            self.encoding_summary[col] = {
                "method": "TimeFeatureExtraction",
                "features_extracted": list(new_cols.keys()),
                "reason": "structural_type = time",
            }

        return df

    # ──────────────────  BOOLEAN  ─────────────────────────────
    
    def _transform_boolean(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._boolean_cols:
            if col not in df.columns:
                continue

            # Normalise to int 0/1
            mapping = {
                "true": 1, "false": 0,
                "yes": 1, "no": 0,
                "1": 1, "0": 0,
                1: 1, 0: 0,
                True: 1, False: 0,
            }
            df[col] = (
                df[col]
                .map(lambda v: mapping.get(v if not isinstance(v, str) else v.lower().strip(), v))
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

            self.encoding_summary[col] = {
                "method": "BooleanCast",
                "reason": "structural_type = boolean → int 0/1",
            }

        return df


# ──────────────────────────────────────────────────────────────
#  Profiling JSON Updater
# ──────────────────────────────────────────────────────────────

def _save_encoding_summary(
    profiling_path: Path,
    engine: EncodingEngine,
    original_cols: List[str],
    final_cols: List[str],
):
    """Append ``encoding_summary`` section to an existing profiling JSON."""
    if not profiling_path.exists():
        return

    with open(profiling_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    methods_used = sorted(
        set(info["method"] for info in engine.encoding_summary.values())
    )
    data["encoding_summary"] = {
        "encoding_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_column_count": len(original_cols),
        "final_column_count": len(final_cols),
        "columns_dropped": sum(
            1 for v in engine.encoding_summary.values() if v["method"] == "Dropped"
        ),
        "columns_encoded": sum(
            1 for v in engine.encoding_summary.values() if v["method"] != "Dropped"
        ),
        "encoding_methods_used": methods_used,
        "column_wise_encoding": engine.encoding_summary,
    }

    with open(profiling_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
#  CLI runner
# ──────────────────────────────────────────────────────────────

def process_user_files(userid: str):
    """Process all ``*_cleaned.csv`` files for a given user."""
    input_dir = Path(f"storage/output/{userid}")
    meta_dir = Path(f"storage/meta_data/{userid}")

    if not input_dir.exists():
        print(f"Error: output directory not found  →  {input_dir}")
        return

    cleaned_files = sorted(input_dir.glob("*_cleaned.csv"))
    if not cleaned_files:
        print(f"No cleaned CSV files found in {input_dir}")
        return

    print(f"\nFound {len(cleaned_files)} cleaned file(s) for '{userid}':")
    for f in cleaned_files:
        print(f"  - {f.name}")

    for file_path in cleaned_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {file_path.name}")
        print(f"{'=' * 60}")

        # Load CSV
        df = pd.read_csv(file_path)
        original_columns = df.columns.tolist()
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Locate profiling JSON
        base_name = file_path.stem.replace("_cleaned", "")
        profiling_path = meta_dir / f"{base_name}_profiling.json"

        if not profiling_path.exists():
            print(f"Profiling file not found: {profiling_path}  – skipping")
            continue

        with open(profiling_path, "r", encoding="utf-8") as f:
            profiling_json = json.load(f)

        # Encode
        engine = EncodingEngine()
        df_encoded = engine.fit_transform(df, profiling_json)
        final_columns = df_encoded.columns.tolist()

        # Save encoded CSV
        output_path = input_dir / f"{base_name}_encoded.csv"
        try:
            df_encoded.to_csv(output_path, index=False)
        except PermissionError:
            output_path = input_dir / f"{base_name}_encoded_new.csv"
            df_encoded.to_csv(output_path, index=False)
            print(f"  (original file locked, saved to alternate name)")
        print(f"\nSaved  →  {output_path}")
        print(f"Shape  →  {df_encoded.shape}")

        # Update profiling JSON
        _save_encoding_summary(profiling_path, engine, original_columns, final_columns)
        print(f"Encoding summary saved to  →  {profiling_path.name}")

        # Print summary
        print(f"\nEncoding Summary:")
        type_counts: Dict[str, int] = {}
        for info in engine.encoding_summary.values():
            m = info["method"]
            type_counts[m] = type_counts.get(m, 0) + 1
        for method, count in sorted(type_counts.items()):
            print(f"  {method}: {count} column(s)")

    print(f"\n{'=' * 60}")
    print(f"All files processed for '{userid}'")
    print(f"{'=' * 60}")


def main():
    print("=" * 60)
    print("AutoML Encoding Engine")
    print("=" * 60)

    userid = input("\nEnter username: ").strip()
    if not userid:
        print("Error: username cannot be empty")
        return

    process_user_files(userid)


if __name__ == "__main__":
    main()
