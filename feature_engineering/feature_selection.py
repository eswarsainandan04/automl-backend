"""
feature_selection.py
====================
AutoML Preprocessing Module — Feature Selection Engine

Part of: backend/data_preprocessing/

Pipeline position:
    profiling -> column_handler -> missing_values -> encoding
    -> scaling -> dimension_reduction -> [THIS MODULE] -> model training

Input  : storage/output/{userid}/*_scaling.csv
Output : same file overwritten in-place
Meta   : storage/meta_data/{userid}/*_profiling.json
         gets a new "feature_selection_summary" block appended

Correction rules applied (v2.0):
    1. Target column is separated from X before ANY selection step.
       It is NEVER passed into variance / correlation filters.
       After selection, X_selected and y are recombined.

    2. Datasets with fewer than MIN_ROWS (50) rows are skipped entirely.
       Variance and correlation statistics are unreliable on tiny datasets.

    3. Constant-column removal (VarianceThreshold=0.0) runs FIRST,
       before the configurable variance threshold step.

    4. Correlation threshold raised to 0.98.
       When two columns exceed the threshold, the one with LOWER variance
       is dropped (keeps the more informative feature).

    5. Pipeline order:
           a. Remove constant columns  (threshold=0.0)
           b. Variance threshold       (threshold=VARIANCE_THRESHOLD)
           c. Correlation filter       (threshold=0.98)
       PCA / dimensionality reduction is NOT performed here.

    6. Metadata block "feature_selection_summary" is fully populated
       including per-strategy details and skipped-reason when applicable.

    7. Target column and identifier columns (flagged in profiling JSON)
       are NEVER touched by any selection step.

Author  : AutoML Pipeline
Version : 2.0.0
"""

import os
import sys
import json
import glob
import warnings
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_selection import VarianceThreshold as SKVarianceThreshold

# Fix encoding for Windows consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ===========================================================================
# Storage root
# ===========================================================================

STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage"
)


# ===========================================================================
# Configuration
# ===========================================================================

class FeatureSelectionConfig:
    """Central configuration — edit these constants to tune behaviour."""

    # ── Dataset size guard ───────────────────────────────────────────────
    # Skip feature selection entirely when the dataset has fewer rows.
    MIN_ROWS = 50

    # ── Step 1: Constant column removal ──────────────────────────────────
    # Always runs first — removes columns with ZERO variance (all same value).
    ENABLE_CONSTANT_REMOVAL = True

    # ── Step 2: Low-variance filter ───────────────────────────────────────
    # Columns whose variance is at or below this value are dropped.
    ENABLE_VARIANCE_FILTER = True
    VARIANCE_THRESHOLD     = 0.01

    # ── Step 3: Correlation filter ────────────────────────────────────────
    # Raised from 0.95 to 0.98 to avoid removing too many features.
    # When a pair exceeds the threshold, the column with LOWER variance is dropped.
    ENABLE_CORRELATION_FILTER = True
    CORRELATION_THRESHOLD     = 0.98
    CORRELATION_METHOD        = "pearson"   # "pearson" | "spearman" | "kendall"

    # ── Safety guard ─────────────────────────────────────────────────────
    # Never drop below this many feature columns total (prevents over-selection).
    MIN_FEATURES_AFTER_SELECTION = 3

    VERSION = "2.0.0"


# ===========================================================================
# Utility helpers
# ===========================================================================

def _now_iso() -> str:
    """Return current timestamp as ISO-8601 string."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read CSV with UTF-8 -> latin-1 -> cp1252 fallback. Returns None on failure."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            warnings.warn(f"[feature_selection] Cannot read '{path}': {exc}")
            return None
    warnings.warn(f"[feature_selection] All encodings exhausted for '{path}'.")
    return None


def _load_json(path: str) -> Dict[str, Any]:
    """Load JSON safely; return empty dict on any error."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"[feature_selection] Cannot parse JSON '{path}': {exc}")
        return {}


def _save_json(data: Dict[str, Any], path: str) -> None:
    """Write dict to JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)
    except OSError as exc:
        warnings.warn(f"[feature_selection] Cannot write JSON '{path}': {exc}")


def _resolve_meta_path(csv_filepath: str, meta_dir: str) -> str:
    """
    Derive the _profiling.json path from a *_scaling.csv path.

    Example
    -------
    .../output/user1/netflix_titles_scaling.csv
        ->  .../meta_data/user1/netflix_titles_profiling.json
    """
    base = os.path.basename(csv_filepath)    # netflix_titles_scaling.csv
    stem = base.replace("_scaling.csv", "")  # netflix_titles
    return os.path.join(meta_dir, f"{stem}_profiling.json")


def _get_identifier_columns(metadata: Dict[str, Any]) -> List[str]:
    """
    Return column names flagged as 'identifier' in the profiling JSON.
    These must never be removed by any selection step.
    """
    return [
        col["column_name"]
        for col in metadata.get("column_wise_summary", [])
        if col.get("structural_type") == "identifier"
    ]


# ===========================================================================
# Step 1 — Constant Column Remover
# ===========================================================================

class ConstantColumnRemover:
    """
    Remove columns that carry zero information — every value is identical.

    Uses VarianceThreshold(threshold=0.0) which removes a column only when
    its variance is exactly zero (all values the same).

    This step runs BEFORE the configurable variance filter so that truly
    constant columns are always eliminated regardless of VARIANCE_THRESHOLD.

    Parameters
    ----------
    protected : list of column names that must never be dropped
                (target column + identifier columns)
    """

    def __init__(self, protected: List[str]) -> None:
        self.protected  = set(protected)
        self.dropped_: List[str] = []
        self.kept_: List[str]    = []

    def fit_transform(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Drop zero-variance columns from the feature DataFrame X.

        Parameters
        ----------
        X : feature DataFrame (target column already removed)

        Returns
        -------
        (X_updated, summary_dict)
        """
        if X.empty or X.shape[1] == 0:
            return X, _empty_summary("ConstantRemoval")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Never touch protected columns
        candidate_cols = [c for c in numeric_cols if c not in self.protected]

        if not candidate_cols:
            return X, _empty_summary("ConstantRemoval")

        matrix   = X[candidate_cols].fillna(0).values
        selector = SKVarianceThreshold(threshold=0.0)
        selector.fit(matrix)

        support       = selector.get_support()
        self.kept_    = [c for c, keep in zip(candidate_cols, support) if keep]
        self.dropped_ = [c for c, keep in zip(candidate_cols, support) if not keep]

        if self.dropped_:
            X = X.drop(columns=self.dropped_)
            print(f"    ConstantRemoval  : dropped {len(self.dropped_):>4} constant column(s)")
            for c in self.dropped_:
                print(f"      - {c}")
        else:
            print("    ConstantRemoval  : no constant columns found")

        return X, {
            "method": "ConstantRemoval",
            "threshold": 0.0,
            "dropped_count": len(self.dropped_),
            "dropped_columns": self.dropped_,
            "kept_count": len(self.kept_),
        }


# ===========================================================================
# Step 2 — Variance Threshold Filter
# ===========================================================================

class VarianceFilter:
    """
    Drop numeric columns whose variance is at or below VARIANCE_THRESHOLD.

    Near-constant features (variance just above zero) add noise rather than
    signal and inflate dimensionality.

    Parameters
    ----------
    threshold : variance below-or-equal threshold for dropping
    protected : list of column names that must never be dropped
    """

    def __init__(
        self,
        threshold: float   = FeatureSelectionConfig.VARIANCE_THRESHOLD,
        protected: List[str] = None,
    ) -> None:
        self.threshold = threshold
        self.protected = set(protected or [])
        self.dropped_: List[str] = []
        self.kept_: List[str]    = []

    def fit_transform(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply variance threshold on feature DataFrame X.

        Parameters
        ----------
        X : feature DataFrame (target already removed, constants already gone)

        Returns
        -------
        (X_updated, summary_dict)
        """
        numeric_cols    = X.select_dtypes(include=[np.number]).columns.tolist()
        candidate_cols  = [c for c in numeric_cols if c not in self.protected]

        if not candidate_cols:
            return X, _empty_summary("VarianceThreshold")

        matrix   = X[candidate_cols].fillna(0).values
        selector = SKVarianceThreshold(threshold=self.threshold)
        selector.fit(matrix)

        support       = selector.get_support()
        self.kept_    = [c for c, keep in zip(candidate_cols, support) if keep]
        self.dropped_ = [c for c, keep in zip(candidate_cols, support) if not keep]

        if self.dropped_:
            X = X.drop(columns=self.dropped_)
            print(
                f"    VarianceFilter   : dropped {len(self.dropped_):>4} column(s) "
                f"(variance <= {self.threshold})"
            )
            for c in self.dropped_:
                print(f"      - {c}")
        else:
            print(f"    VarianceFilter   : no columns dropped (threshold={self.threshold})")

        return X, {
            "method": "VarianceThreshold",
            "threshold": self.threshold,
            "dropped_count": len(self.dropped_),
            "dropped_columns": self.dropped_,
            "kept_count": len(self.kept_),
        }


# ===========================================================================
# Step 3 — Correlation Filter
# ===========================================================================

class CorrelationFilter:
    """
    Remove one column from every highly-correlated pair.

    Threshold raised to 0.98 (was 0.95) to avoid removing too many features.

    Tie-breaking rule
    -----------------
    When columns A and B correlate above the threshold, the column with
    the LOWER variance is dropped (the more informative one is kept).

    Parameters
    ----------
    threshold : absolute Pearson correlation upper limit
    method    : "pearson" | "spearman" | "kendall"
    protected : columns that must never be dropped
    """

    def __init__(
        self,
        threshold: float   = FeatureSelectionConfig.CORRELATION_THRESHOLD,
        method: str        = FeatureSelectionConfig.CORRELATION_METHOD,
        protected: List[str] = None,
    ) -> None:
        self.threshold = threshold
        self.method    = method
        self.protected = set(protected or [])
        self.dropped_: List[str] = []
        self.kept_: List[str]    = []

    def fit_transform(
        self, X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply correlation filter on feature DataFrame X.

        Parameters
        ----------
        X : feature DataFrame (target already removed)

        Returns
        -------
        (X_updated, summary_dict)
        """
        numeric_cols   = X.select_dtypes(include=[np.number]).columns.tolist()
        candidate_cols = [c for c in numeric_cols if c not in self.protected]

        if len(candidate_cols) < 2:
            return X, _empty_summary("CorrelationFilter")

        sub    = X[candidate_cols].fillna(0)
        variances = sub.var()

        # Absolute correlation matrix
        corr_matrix = sub.corr(method=self.method).abs()

        # Upper triangle only (no self-correlation, no duplicates)
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )

        to_drop: set = set()

        # Iterate every cell in upper triangle
        for row_name in upper.index:
            for col_name in upper.columns:
                corr_val = upper.loc[row_name, col_name]
                if pd.isna(corr_val):
                    continue
                if corr_val > self.threshold:
                    # Both columns haven't been dropped yet
                    if row_name in to_drop or col_name in to_drop:
                        continue
                    # Never drop protected columns
                    row_protected = row_name in self.protected
                    col_protected = col_name in self.protected

                    if row_protected and col_protected:
                        continue  # can't drop either
                    elif row_protected:
                        to_drop.add(col_name)
                    elif col_protected:
                        to_drop.add(row_name)
                    else:
                        # Drop the one with LOWER variance (keep more informative)
                        if variances.get(row_name, 0) >= variances.get(col_name, 0):
                            to_drop.add(col_name)
                        else:
                            to_drop.add(row_name)

        self.dropped_ = [c for c in candidate_cols if c in to_drop]
        self.kept_    = [c for c in candidate_cols if c not in to_drop]

        if self.dropped_:
            X = X.drop(columns=self.dropped_)
            print(
                f"    CorrelationFilter : dropped {len(self.dropped_):>4} column(s) "
                f"(|{self.method} corr| > {self.threshold})"
            )
            for c in self.dropped_:
                print(f"      - {c}")
        else:
            print(
                f"    CorrelationFilter : no columns dropped "
                f"(threshold={self.threshold})"
            )

        return X, {
            "method": "CorrelationFilter",
            "correlation_method": self.method,
            "threshold": self.threshold,
            "dropped_count": len(self.dropped_),
            "dropped_columns": self.dropped_,
            "kept_count": len(self.kept_),
        }


# ===========================================================================
# Shared helper
# ===========================================================================

def _empty_summary(method: str) -> Dict[str, Any]:
    """Return a no-op summary dict for a strategy that had nothing to do."""
    return {
        "method": method,
        "dropped_count": 0,
        "dropped_columns": [],
        "kept_count": 0,
        "skipped": True,
        "reason": "no eligible columns",
    }


# ===========================================================================
# Core Engine
# ===========================================================================

class FeatureSelectionEngine:
    """
    Orchestrates feature selection for every *_scaling.csv dataset
    belonging to a given user.

    Corrected pipeline per dataset
    --------------------------------
    0.  Size guard          – skip entirely if rows < MIN_ROWS
    1.  Separate target     – X = df.drop(target); y = df[target]
    2.  Collect identifiers – read from profiling JSON; never drop these
    3.  ConstantRemoval     – VarianceThreshold(0.0) on X only
    4.  VarianceFilter      – VarianceThreshold(VARIANCE_THRESHOLD) on X
    5.  CorrelationFilter   – |corr| > 0.98, keep higher-variance col, on X
    6.  Recombine           – df_final = concat(X_selected, y)
    7.  save_dataset()      – overwrite *_scaling.csv
    8.  update_metadata()   – write feature_selection_summary to JSON

    Usage
    -----
    engine = FeatureSelectionEngine()
    engine.run()
    """

    def __init__(self) -> None:
        self.user_id: str          = ""
        self.output_dir: str       = ""
        self.meta_dir: str         = ""
        self.target_col: Optional[str] = None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Prompt for user-id + target column, then process every dataset."""
        self._print_banner()
        self._prompt_user_id()
        self._resolve_directories()
        self.process_all_datasets()

    def process_all_datasets(self) -> None:
        """Discover and process every *_scaling.csv for the current user."""
        pattern = os.path.join(self.output_dir, "*_scaling.csv")
        files   = sorted(glob.glob(pattern))

        if not files:
            print(f"\n[WARNING] No *_scaling.csv files found in:\n  {self.output_dir}")
            return

        success = 0
        errors  = 0

        for filepath in files:
            filename = os.path.basename(filepath)
            print(f"\n{'=' * 60}")
            print(f"  Processing : {filename}")
            print(f"{'=' * 60}")
            try:
                self._process_single_dataset(filepath)
                success += 1
            except Exception as exc:
                errors += 1
                warnings.warn(f"[feature_selection] Failed on '{filename}': {exc}")
                traceback.print_exc()

        print(f"\n{'=' * 60}")
        print(f"  Feature selection complete.")
        print(f"  Success : {success}   Errors : {errors}")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Per-dataset pipeline
    # ------------------------------------------------------------------

    def _process_single_dataset(self, filepath: str) -> None:
        """Run the corrected selection pipeline for one scaled CSV file."""

        # ── Load ────────────────────────────────────────────────────────
        df = _safe_read_csv(filepath)
        if df is None or df.empty:
            print("  [SKIP] Could not load dataset or dataset is empty.")
            return

        original_shape = df.shape
        print(f"\n  Original shape : {original_shape[0]} rows x {original_shape[1]} cols")

        # ── Load profiling metadata ──────────────────────────────────────
        meta_path = _resolve_meta_path(filepath, self.meta_dir)
        metadata  = _load_json(meta_path)

        # ── 0. Size guard ────────────────────────────────────────────────
        if df.shape[0] < FeatureSelectionConfig.MIN_ROWS:
            print(
                f"\n  [SKIP] Dataset has only {df.shape[0]} rows "
                f"(minimum: {FeatureSelectionConfig.MIN_ROWS}). "
                "Feature selection skipped — statistics unreliable on small datasets."
            )
            self._write_skipped_metadata(
                metadata, meta_path, original_shape,
                reason="feature_selection_skipped_small_dataset",
            )
            return

        # ── 1. Separate target column ────────────────────────────────────
        #    RULE: target must NEVER enter any selection filter.
        target_col = self.target_col
        y: Optional[pd.Series] = None

        if target_col:
            if target_col in df.columns:
                y  = df[target_col].copy()
                X  = df.drop(columns=[target_col]).copy()
                print(f"  Target column  : '{target_col}' separated from features")
            else:
                print(
                    f"  [WARNING] Target column '{target_col}' not found in "
                    f"{os.path.basename(filepath)} — treating as unsupervised"
                )
                target_col = None
                X = df.copy()
        else:
            print("  Target column  : not set — unsupervised selection only")
            X = df.copy()

        # ── 2. Collect protected columns ─────────────────────────────────
        #    Identifier columns from profiling JSON are never removed.
        identifier_cols = _get_identifier_columns(metadata)
        # Build the full protected set: target + identifiers
        protected = set(identifier_cols)
        if target_col:
            protected.add(target_col)

        if identifier_cols:
            print(f"  Protected cols : {identifier_cols} (identifier, never dropped)")

        # ── Accumulate strategy log ───────────────────────────────────────
        strategy_log: List[Dict] = []
        print()

        # ── 3. Remove constant columns (threshold = 0.0) ─────────────────
        if FeatureSelectionConfig.ENABLE_CONSTANT_REMOVAL:
            cr = ConstantColumnRemover(protected=list(protected))
            X, cr_summary = cr.fit_transform(X)
            strategy_log.append(cr_summary)

        # Guard: stop if too few features remain
        if not self._enough_features(X, strategy_log, "after constant removal"):
            df_final = self._recombine(X, y, target_col)
            self._finish(df_final, filepath, metadata, meta_path,
                         original_shape, strategy_log)
            return

        # ── 4. Low-variance filter (threshold = VARIANCE_THRESHOLD) ──────
        if FeatureSelectionConfig.ENABLE_VARIANCE_FILTER:
            vf = VarianceFilter(
                threshold=FeatureSelectionConfig.VARIANCE_THRESHOLD,
                protected=list(protected),
            )
            X, vf_summary = vf.fit_transform(X)
            strategy_log.append(vf_summary)

        # Guard
        if not self._enough_features(X, strategy_log, "after variance filter"):
            df_final = self._recombine(X, y, target_col)
            self._finish(df_final, filepath, metadata, meta_path,
                         original_shape, strategy_log)
            return

        # ── 5. Correlation filter (threshold = 0.98) ─────────────────────
        if FeatureSelectionConfig.ENABLE_CORRELATION_FILTER:
            cf = CorrelationFilter(
                threshold=FeatureSelectionConfig.CORRELATION_THRESHOLD,
                method=FeatureSelectionConfig.CORRELATION_METHOD,
                protected=list(protected),
            )
            X, cf_summary = cf.fit_transform(X)
            strategy_log.append(cf_summary)

        # ── 6. Recombine X_selected with y ───────────────────────────────
        df_final = self._recombine(X, y, target_col)

        # ── 7 & 8. Save + update metadata ────────────────────────────────
        self._finish(df_final, filepath, metadata, meta_path,
                     original_shape, strategy_log)

    # ------------------------------------------------------------------
    # Recombine helper
    # ------------------------------------------------------------------

    def _recombine(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        target_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Attach the target column back to the right side of X.

        If no target was separated, returns X unchanged.

        Parameters
        ----------
        X          : feature DataFrame after selection
        y          : target Series (or None)
        target_col : name of the target column

        Returns
        -------
        Recombined DataFrame with target as the last column.
        """
        if y is not None and target_col:
            df_final = pd.concat([X.reset_index(drop=True),
                                   y.reset_index(drop=True)], axis=1)
            print(f"\n  Target '{target_col}' recombined -> final shape: "
                  f"{df_final.shape[0]} rows x {df_final.shape[1]} cols")
            return df_final
        return X

    # ------------------------------------------------------------------
    # Guard helper
    # ------------------------------------------------------------------

    def _enough_features(
        self,
        X: pd.DataFrame,
        strategy_log: List[Dict],
        stage: str,
    ) -> bool:
        """
        Return True if X still has enough numeric feature columns.
        Prints a warning and returns False if the minimum is breached.
        """
        n_numeric = len(X.select_dtypes(include=[np.number]).columns)
        if n_numeric < FeatureSelectionConfig.MIN_FEATURES_AFTER_SELECTION:
            print(
                f"\n  [GUARD] Only {n_numeric} numeric feature(s) remain {stage}. "
                "Stopping further selection to preserve data integrity."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Finish step (save + metadata)
    # ------------------------------------------------------------------

    def _finish(
        self,
        df_final: pd.DataFrame,
        filepath: str,
        metadata: Dict[str, Any],
        meta_path: str,
        original_shape: Tuple[int, int],
        strategy_log: List[Dict],
    ) -> None:
        """Save the final dataset and write the metadata summary block."""
        new_shape     = df_final.shape
        total_dropped = original_shape[1] - new_shape[1]

        print(f"\n  New shape      : {new_shape[0]} rows x {new_shape[1]} cols")
        print(f"  Columns removed: {total_dropped}")

        self.save_dataset(df_final, filepath)
        self.update_metadata(
            metadata, meta_path, original_shape, new_shape,
            strategy_log, total_dropped
        )

    # ------------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------------

    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Overwrite the *_scaling.csv with the feature-selected DataFrame.

        Parameters
        ----------
        df       : final DataFrame (X_selected + y if target existed)
        filepath : path to the scaled CSV (will be overwritten in-place)
        """
        try:
            df.to_csv(filepath, index=False, encoding="utf-8")
            print(f"  Saved  ->  {os.path.basename(filepath)}")
        except OSError as exc:
            warnings.warn(
                f"[feature_selection] Cannot save to '{filepath}': {exc}"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def update_metadata(
        self,
        metadata: Dict[str, Any],
        meta_path: str,
        original_shape: Tuple[int, int],
        new_shape: Tuple[int, int],
        strategy_log: List[Dict],
        total_dropped: int,
    ) -> None:
        """
        Write a ``feature_selection_summary`` block to the profiling JSON.

        Schema
        ------
        {
            "applied": true,
            "version": "2.0.0",
            "target_column_used": "price",
            "original_feature_count": 45,
            "selected_feature_count": 38,
            "columns_removed": 7,
            "strategies_applied": ["ConstantRemoval", "VarianceThreshold",
                                    "CorrelationFilter"],
            "strategy_details": [...],
            "timestamp": "2026-01-10T10:20:30"
        }
        """
        strategies_run = [
            s["method"] for s in strategy_log
            if not s.get("skipped", False)
        ]

        metadata["feature_selection_summary"] = {
            "applied": total_dropped > 0,
            "version": FeatureSelectionConfig.VERSION,
            "target_column_used": self.target_col if self.target_col else None,
            "original_feature_count": original_shape[1],
            "selected_feature_count": new_shape[1],
            "columns_removed": total_dropped,
            "strategies_applied": strategies_run,
            "strategy_details": strategy_log,
            "timestamp": _now_iso(),
        }

        _save_json(metadata, meta_path)
        print(f"  Metadata updated  ->  {os.path.basename(meta_path)}")

    def _write_skipped_metadata(
        self,
        metadata: Dict[str, Any],
        meta_path: str,
        original_shape: Tuple[int, int],
        reason: str,
    ) -> None:
        """Write a 'not applied' metadata block with the skip reason."""
        metadata["feature_selection_summary"] = {
            "applied": False,
            "version": FeatureSelectionConfig.VERSION,
            "target_column_used": self.target_col if self.target_col else None,
            "original_feature_count": original_shape[1],
            "selected_feature_count": original_shape[1],
            "columns_removed": 0,
            "strategies_applied": [],
            "strategy_details": [],
            "skipped_reason": reason,
            "timestamp": _now_iso(),
        }
        _save_json(metadata, meta_path)
        print(f"  Metadata updated  ->  {os.path.basename(meta_path)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        print("\n" + "=" * 60)
        print("   FEATURE SELECTION ENGINE  v2.0")
        print("=" * 60)

    def _prompt_user_id(self) -> None:
        """Prompt for user-id and optional target column name."""
        self.user_id = input("\nEnter user ID: ").strip()
        if not self.user_id:
            raise ValueError("User ID cannot be empty.")
        print(f"\nUser ID: {self.user_id}")

        print(
            "\nEnter the target column name for supervised feature selection."
        )
        print("(Press Enter to skip — only unsupervised strategies will run)")
        target = input("Target column name: ").strip()
        if target:
            self.target_col = target
            print(f"Target column  : '{self.target_col}'")
        else:
            self.target_col = None
            print("Target column  : not set — unsupervised selection only")

    def _resolve_directories(self) -> None:
        """Validate and store storage directories for the configured user."""
        self.output_dir = os.path.join(STORAGE_ROOT, "output",    self.user_id)
        self.meta_dir   = os.path.join(STORAGE_ROOT, "meta_data", self.user_id)

        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(
                f"Output directory not found for user '{self.user_id}':\n"
                f"  {self.output_dir}"
            )
        os.makedirs(self.meta_dir, exist_ok=True)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    engine = FeatureSelectionEngine()
    engine.run()