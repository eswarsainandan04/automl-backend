"""
dimension_reduction.py
======================
AutoML Preprocessing Module — Dimensionality Reduction Engine

Part of: backend/data_preprocessing/

This module automatically detects and reduces high-dimensional feature spaces
in scaled CSV datasets produced by the scaling pipeline.

Handles THREE distinct feature group types:

  1. TF-IDF groups   — columns named  <col>_tfidf_<N>
                       Created by EncodingEngine._transform_text()
                       Reduced with TruncatedSVD (sparse-friendly)

  2. One-hot groups  — columns named  <col>_<value>
                       Created by EncodingEngine._transform_categorical()
                       Reduced with TruncatedSVD

  3. Numeric groups  — plain numeric columns (structural_type = numeric /
                       float / integer / percentage / currency / ratio …)
                       Reduced with PCA when total numeric cols > threshold

Reads profiling JSON to identify each column's structural_type and
encoding_summary to understand which columns came from one-hot encoding.

Author  : AutoML Pipeline
Version : 2.0.0
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

# Fix encoding for Windows consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ===========================================================================
# Storage root — mirrors the convention used in profiling.py / encoding.py
# ===========================================================================

STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "storage"
)


# ===========================================================================
# Tunable constants
# ===========================================================================

# ── TF-IDF / One-hot (sparse) reduction ─────────────────────────────────
ONE_HOT_GROUP_MIN_COLS    = 5   # minimum members to form a group
TFIDF_REDUCTION_THRESHOLD = 30  # apply SVD only when group > this many cols
ONEHOT_REDUCTION_THRESHOLD = 30 # apply SVD only when group > this many cols
MAX_SVD_COMPONENTS        = 20  # hard ceiling on SVD output dimensions

# ── Numeric (dense) reduction ────────────────────────────────────────────
NUMERIC_REDUCTION_THRESHOLD = 20    # apply PCA when numeric cols > this
NUMERIC_MAX_PCA_COMPONENTS  = 15    # hard ceiling on PCA output dimensions
NUMERIC_VARIANCE_TARGET     = 0.95  # keep components for 95 % variance

# ── Structural types treated as "numeric" ────────────────────────────────
# Matches the types assigned by structural_type_detector.py
NUMERIC_STRUCTURAL_TYPES = frozenset({
    "numeric", "integer", "float",
    "percentage", "ratio", "currency", "salary",
    "distance", "weight", "volume", "area", "speed",
    "temperature", "pressure", "energy", "power",
    "capacity", "density", "angle",
})

# Substring that marks a TF-IDF column produced by EncodingEngine
TFIDF_SUFFIX_PATTERN = "_tfidf_"   # e.g. description_tfidf_0


# ===========================================================================
# Utility helpers
# ===========================================================================

def _now_iso() -> str:
    """Return current timestamp as ISO-8601 string."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV with UTF-8 -> latin-1 -> cp1252 fallback.
    Returns None on total failure.
    """
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            warnings.warn(f"[dimension_reduction] Cannot read '{path}': {exc}")
            return None
    warnings.warn(f"[dimension_reduction] All encodings exhausted for '{path}'.")
    return None


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file safely; return empty dict on any error."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"[dimension_reduction] Could not parse JSON '{path}': {exc}")
        return {}


def _save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dict to JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, ensure_ascii=False)
    except OSError as exc:
        warnings.warn(f"[dimension_reduction] Cannot write JSON '{path}': {exc}")


# ===========================================================================
# Core Engine
# ===========================================================================

class DimensionReductionEngine:
    """
    Orchestrates dimensionality reduction for all *_scaling.csv datasets
    belonging to a given user.

    Pipeline per dataset
    --------------------
    1. load_dataset()           – read CSV
    2. (load profiling JSON)    – read _profiling.json for column context
    3. detect_tfidf_groups()    – find <col>_tfidf_<N> column clusters
    4. detect_onehot_groups()   – find one-hot clusters using encoding summary
    5. detect_numeric_columns() – gather plain numeric feature columns
    6. apply_svd()              – TruncatedSVD on tfidf / onehot groups
    7. apply_pca_numeric()      – PCA on numeric block
    8. save_dataset()           – overwrite *_scaling.csv
    9. update_metadata()        – write dimension_reduction_summary to JSON

    Usage
    -----
    engine = DimensionReductionEngine()
    engine.run()
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.user_id: str    = ""
        self.output_dir: str = ""
        self.meta_dir: str   = ""

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Prompt for user-id then process every scaled dataset."""
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

        for filepath in files:
            filename = os.path.basename(filepath)
            print(f"\n{'=' * 55}")
            print(f"  Processing: {filename}")
            print(f"{'=' * 55}")
            try:
                self._process_single_dataset(filepath)
            except Exception as exc:
                warnings.warn(
                    f"[dimension_reduction] Failed on '{filename}': {exc}"
                )

        print(f"\n{'=' * 55}")
        print("  All datasets processed.")
        print(f"{'=' * 55}\n")

    # ------------------------------------------------------------------
    # Per-dataset pipeline
    # ------------------------------------------------------------------

    def _process_single_dataset(self, filepath: str) -> None:
        """Run the full reduction pipeline for one scaled CSV file."""

        # 1. Load dataset ------------------------------------------------
        df = self.load_dataset(filepath)
        if df is None:
            print("  [SKIP] Could not load dataset.")
            return

        if df.empty or df.shape[1] < 5:
            print(f"  [SKIP] Too few columns ({df.shape[1]}) for reduction.")
            return

        original_shape = df.shape
        print(f"\n  Original shape : {original_shape[0]} rows x {original_shape[1]} cols")

        # 2. Load profiling metadata -------------------------------------
        meta_path = self._resolve_meta_path(filepath)
        metadata  = _load_json(meta_path)

        # 3. Detect TF-IDF groups ----------------------------------------
        tfidf_groups = self.detect_tfidf_groups(df)

        # 4. Detect one-hot groups (metadata-aware) ----------------------
        onehot_groups = self.detect_onehot_groups(df, metadata)

        # 5. Detect plain numeric columns --------------------------------
        #    Exclude columns already assigned to a tfidf / onehot group
        already_grouped: set = set()
        for g in tfidf_groups.values():
            already_grouped.update(g)
        for g in onehot_groups.values():
            already_grouped.update(g)

        numeric_cols = self.detect_numeric_columns(df, metadata, already_grouped)

        # Print detection summary ----------------------------------------
        print("\n  -- Feature-group detection --")

        if tfidf_groups:
            print("  TF-IDF groups:")
            for prefix, cols in tfidf_groups.items():
                print(f"    {prefix:<30} {len(cols):>4} columns")
        else:
            print("  TF-IDF groups       : none detected")

        if onehot_groups:
            print("  One-hot groups:")
            for prefix, cols in onehot_groups.items():
                print(f"    {prefix:<30} {len(cols):>4} columns")
        else:
            print("  One-hot groups      : none detected")

        print(f"  Numeric columns     : {len(numeric_cols)} columns")
        print()

        # Accumulate reduction log for metadata --------------------------
        reduction_log: List[Dict] = []
        any_reduction = False

        # 6. SVD on TF-IDF and one-hot groups ----------------------------
        all_sparse_groups: Dict[str, List[str]] = {}
        all_sparse_groups.update(tfidf_groups)
        all_sparse_groups.update(onehot_groups)

        if all_sparse_groups:
            df, svd_log = self.apply_svd(df, all_sparse_groups)
            reduction_log.extend(svd_log)
            if svd_log:
                any_reduction = True
        else:
            print("  [INFO] No TF-IDF / one-hot groups meet the reduction threshold.")

        # 7. PCA on numeric columns --------------------------------------
        if len(numeric_cols) > NUMERIC_REDUCTION_THRESHOLD:
            df, pca_log = self.apply_pca_numeric(df, numeric_cols)
            if pca_log:
                reduction_log.extend(pca_log)
                any_reduction = True
        else:
            print(
                f"  [INFO] Numeric PCA skipped "
                f"({len(numeric_cols)} cols <= threshold {NUMERIC_REDUCTION_THRESHOLD})"
            )

        new_shape = df.shape
        saved_cols = original_shape[1] - new_shape[1]
        print(f"\n  New shape      : {new_shape[0]} rows x {new_shape[1]} cols")
        print(f"  Columns removed: {saved_cols}")

        # 8. Save dataset ------------------------------------------------
        self.save_dataset(df, filepath)

        # 9. Update metadata ---------------------------------------------
        if any_reduction:
            self.update_metadata(
                filepath, metadata, meta_path,
                original_shape, new_shape, reduction_log
            )
        else:
            print("  [INFO] No reduction applied.")
            self._write_no_reduction_metadata(
                metadata, meta_path, original_shape
            )

    # ------------------------------------------------------------------
    # Step 1 -- Load dataset
    # ------------------------------------------------------------------

    def load_dataset(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load a CSV file into a pandas DataFrame.

        Parameters
        ----------
        filepath : str
            Absolute path to the *_scaling.csv file.

        Returns
        -------
        pd.DataFrame or None
        """
        if not os.path.isfile(filepath):
            warnings.warn(f"[dimension_reduction] File not found: {filepath}")
            return None
        return _safe_read_csv(filepath)

    # ------------------------------------------------------------------
    # Step 3 -- Detect TF-IDF groups
    # ------------------------------------------------------------------

    def detect_tfidf_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Find columns created by TF-IDF vectorisation.

        The EncodingEngine names them: <original_col>_tfidf_<index>
        e.g.  description_tfidf_0, description_tfidf_1, ...

        Only groups with more than TFIDF_REDUCTION_THRESHOLD columns are
        returned (SVD is not worthwhile for tiny groups).

        Returns
        -------
        dict  prefix -> [col, col, ...]
        """
        bucket: Dict[str, List[str]] = {}

        for col in df.columns:
            if TFIDF_SUFFIX_PATTERN in col:
                # everything before "_tfidf_" is the source column prefix
                prefix = col[: col.index(TFIDF_SUFFIX_PATTERN)]
                bucket.setdefault(prefix, []).append(col)

        # apply size threshold
        return {
            prefix: cols
            for prefix, cols in bucket.items()
            if len(cols) > TFIDF_REDUCTION_THRESHOLD
        }

    # ------------------------------------------------------------------
    # Step 4 -- Detect one-hot groups (metadata-aware)
    # ------------------------------------------------------------------

    def detect_onehot_groups(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Find one-hot encoded column groups (from scaled CSV).

        Primary source
        --------------
        encoding_summary.column_wise_encoding in the profiling JSON.
        Any column whose method is "OneHotEncoding" contributes its
        new_columns list to the group keyed by the original column name.

        Fallback (when JSON is absent or a column is not listed)
        ---------------------------------------------------------
        Heuristic: columns that are integer-dtype, contain an underscore,
        and share a common prefix with >= ONE_HOT_GROUP_MIN_COLS siblings.

        Only groups exceeding ONEHOT_REDUCTION_THRESHOLD are returned.

        Returns
        -------
        dict  prefix -> [col, col, ...]
        """
        groups: Dict[str, List[str]] = {}
        df_cols_set = set(df.columns)

        # Primary: encoding_summary from profiling JSON ------------------
        encoding_summary = (
            metadata
            .get("encoding_summary", {})
            .get("column_wise_encoding", {})
        )

        for orig_col, enc_info in encoding_summary.items():
            if enc_info.get("method") != "OneHotEncoding":
                continue
            new_cols = [
                c for c in enc_info.get("new_columns", [])
                if c in df_cols_set
            ]
            if new_cols:
                groups[orig_col] = new_cols

        # Fallback: name-pattern heuristic --------------------------------
        already_accounted: set = set()
        for cols in groups.values():
            already_accounted.update(cols)

        prefix_map: Dict[str, List[str]] = {}

        for col in df.columns:
            if col in already_accounted:
                continue
            # only integer-dtype (0/1 dummy columns)
            if not pd.api.types.is_integer_dtype(df[col]):
                continue
            # must contain underscore (prefix_value pattern)
            if "_" not in col:
                continue
            # skip TF-IDF columns
            if TFIDF_SUFFIX_PATTERN in col:
                continue
            # skip output columns from a previous reduction run
            if "_svd_" in col or "_pca_" in col:
                continue

            prefix = col.rsplit("_", 1)[0]
            prefix_map.setdefault(prefix, []).append(col)

        for prefix, cols in prefix_map.items():
            if len(cols) >= ONE_HOT_GROUP_MIN_COLS and prefix not in groups:
                groups[prefix] = cols

        # Apply size threshold -------------------------------------------
        return {
            prefix: cols
            for prefix, cols in groups.items()
            if len(cols) > ONEHOT_REDUCTION_THRESHOLD
        }

    # ------------------------------------------------------------------
    # Step 5 -- Detect numeric columns
    # ------------------------------------------------------------------

    def detect_numeric_columns(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        exclude: set,
    ) -> List[str]:
        """
        Identify plain numeric feature columns eligible for PCA.

        Decision logic (in priority order)
        ------------------------------------
        1. If the column appears in column_wise_summary with a
           structural_type inside NUMERIC_STRUCTURAL_TYPES -> include.
        2. If it appears with any OTHER structural_type -> skip.
        3. If not in the profiling JSON at all -> include if it is
           numeric-dtype, not binary-only (0/1), and not an SVD/PCA
           output column.

        Columns in *exclude* (already assigned to an SVD group) are
        always skipped.

        Parameters
        ----------
        df       : scaled DataFrame
        metadata : profiling JSON dict (may be empty)
        exclude  : set of column names already handled by SVD groups

        Returns
        -------
        List of column names suitable for PCA.
        """
        numeric_cols: List[str] = []

        # Build lookup: column_name -> structural_type from JSON
        struct_map: Dict[str, str] = {
            c["column_name"]: c.get("structural_type", "")
            for c in metadata.get("column_wise_summary", [])
        }

        for col in df.columns:
            # Skip columns already assigned to a sparse (SVD) group
            if col in exclude:
                continue
            # Skip output columns from a previous reduction pass
            if "_svd_" in col or "_pca_" in col:
                continue
            # Must be numeric dtype
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            structural_type = struct_map.get(col, "")

            if structural_type:
                # Metadata-driven decision
                if structural_type in NUMERIC_STRUCTURAL_TYPES:
                    numeric_cols.append(col)
                # Any non-numeric structural_type -> skip (boolean, identifier…)
            else:
                # Fallback heuristic: skip binary dummies (only 0 and 1)
                unique_vals = set(df[col].dropna().unique())
                if unique_vals.issubset({0, 1, 0.0, 1.0}):
                    continue
                numeric_cols.append(col)

        return numeric_cols

    # ------------------------------------------------------------------
    # Step 6 -- Apply TruncatedSVD (TF-IDF / one-hot groups)
    # ------------------------------------------------------------------

    def apply_svd(
        self,
        df: pd.DataFrame,
        groups: Dict[str, List[str]],
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply TruncatedSVD to each TF-IDF / one-hot feature group.

        Component count formula
        -----------------------
        k = min(MAX_SVD_COMPONENTS, int(num_features * 0.5))
        k must be >= 1 and < num_features (TruncatedSVD requirement).

        New columns are named: <prefix>_svd_<1-based index>.
        Original group columns are removed from the DataFrame.

        Parameters
        ----------
        df     : scaled DataFrame
        groups : prefix -> list of column names

        Returns
        -------
        (updated_df, reduction_log)
        """
        log: List[Dict] = []

        print("  -- SVD reduction (TF-IDF / one-hot groups) --")

        for prefix, original_cols in groups.items():
            num_features = len(original_cols)
            k = min(MAX_SVD_COMPONENTS, int(num_features * 0.5))
            k = max(k, 1)
            k = min(k, num_features - 1)

            try:
                matrix  = df[original_cols].fillna(0).values
                svd     = TruncatedSVD(n_components=k, random_state=42)
                reduced = svd.fit_transform(matrix)

                var_ratio = float(svd.explained_variance_ratio_.sum())

                new_col_names = [f"{prefix}_svd_{i + 1}" for i in range(k)]
                reduced_df = pd.DataFrame(
                    reduced, columns=new_col_names, index=df.index
                )

                df = df.drop(columns=original_cols)
                df = pd.concat([df, reduced_df], axis=1)

                print(
                    f"    {prefix:<30}  {num_features:>4} -> {k:>3} components"
                    f"  (variance: {var_ratio:.1%})"
                )

                log.append({
                    "group_type": "tfidf_or_onehot",
                    "prefix": prefix,
                    "original_columns": num_features,
                    "reduced_to": k,
                    "method": "TruncatedSVD",
                    "explained_variance": round(var_ratio, 4),
                })

            except Exception as exc:
                warnings.warn(
                    f"[dimension_reduction] SVD failed for '{prefix}': {exc}"
                )
                log.append({
                    "group_type": "tfidf_or_onehot",
                    "prefix": prefix,
                    "original_columns": num_features,
                    "reduced_to": num_features,
                    "method": "TruncatedSVD",
                    "error": str(exc),
                })

        return df, log

    # ------------------------------------------------------------------
    # Step 7 -- Apply PCA on numeric columns
    # ------------------------------------------------------------------

    def apply_pca_numeric(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply PCA to the block of plain numeric feature columns.

        Strategy
        --------
        1. Impute NaN with column median (no data leakage risk for
           unsupervised PCA).
        2. Apply StandardScaler (PCA requires zero-mean / unit-variance).
        3. Determine component count k as the MINIMUM of:
             - NUMERIC_MAX_PCA_COMPONENTS
             - int(n_features * 0.5)
             - the number of components needed to reach
               NUMERIC_VARIANCE_TARGET cumulative explained variance
        4. Original numeric columns are dropped; new columns are named
           numeric_pca_<1-based index>.

        Parameters
        ----------
        df          : scaled DataFrame
        numeric_cols: list of column names to reduce

        Returns
        -------
        (updated_df, reduction_log)
        """
        log: List[Dict] = []
        n_features = len(numeric_cols)

        print("  -- PCA reduction (numeric columns) --")
        print(f"    Input numeric columns : {n_features}")

        try:
            # Median-impute NaN values -----------------------------------
            matrix = df[numeric_cols].copy()
            for col in matrix.columns:
                if matrix[col].isna().any():
                    matrix[col] = matrix[col].fillna(matrix[col].median())
            matrix = matrix.values.astype(float)

            # Standard-scale before PCA ----------------------------------
            scaler        = StandardScaler()
            matrix_scaled = scaler.fit_transform(matrix)

            # Determine the max safe number of components ----------------
            max_k = min(
                NUMERIC_MAX_PCA_COMPONENTS,
                int(n_features * 0.5),
                n_features - 1,
                matrix.shape[0] - 1,   # cannot exceed n_samples - 1
            )
            max_k = max(max_k, 1)

            # Fit PCA with max_k to find variance-optimal component count
            pca_probe = PCA(n_components=max_k, random_state=42)
            pca_probe.fit(matrix_scaled)

            cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
            # Smallest k that reaches the variance target
            k_var = int(np.searchsorted(cumvar, NUMERIC_VARIANCE_TARGET) + 1)
            k     = max(min(k_var, max_k), 1)

            # Final transform with chosen k ------------------------------
            pca_final = PCA(n_components=k, random_state=42)
            reduced   = pca_final.fit_transform(matrix_scaled)
            var_ratio = float(pca_final.explained_variance_ratio_.sum())

            new_col_names = [f"numeric_pca_{i + 1}" for i in range(k)]
            reduced_df = pd.DataFrame(
                reduced, columns=new_col_names, index=df.index
            )

            df = df.drop(columns=numeric_cols)
            df = pd.concat([df, reduced_df], axis=1)

            print(f"    {n_features} numeric columns  ->  {k} PCA components")
            print(f"    Variance explained    : {var_ratio:.1%}")

            log.append({
                "group_type": "numeric",
                "prefix": "numeric_pca",
                "original_columns": n_features,
                "original_column_names": numeric_cols,
                "reduced_to": k,
                "method": "PCA",
                "explained_variance": round(var_ratio, 4),
                "variance_target": NUMERIC_VARIANCE_TARGET,
            })

        except Exception as exc:
            warnings.warn(
                f"[dimension_reduction] PCA failed on numeric block: {exc}"
            )
            log.append({
                "group_type": "numeric",
                "prefix": "numeric_pca",
                "original_columns": n_features,
                "reduced_to": n_features,
                "method": "PCA",
                "error": str(exc),
            })

        return df, log

    # ------------------------------------------------------------------
    # Step 8 -- Save dataset
    # ------------------------------------------------------------------

    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Overwrite the *_scaling.csv with the reduced DataFrame.

        Parameters
        ----------
        df       : reduced DataFrame
        filepath : path to the scaled CSV (will be overwritten)
        """
        try:
            df.to_csv(filepath, index=False, encoding="utf-8")
            print(f"\n  Saved  ->  {os.path.basename(filepath)}")
        except OSError as exc:
            warnings.warn(
                f"[dimension_reduction] Cannot save dataset to '{filepath}': {exc}"
            )

    # ------------------------------------------------------------------
    # Step 9 -- Update profiling metadata
    # ------------------------------------------------------------------

    def update_metadata(
        self,
        csv_filepath: str,
        metadata: Dict[str, Any],
        meta_path: str,
        original_shape: Tuple[int, int],
        new_shape: Tuple[int, int],
        reduction_log: List[Dict],
    ) -> None:
        """
        Inject a dimension_reduction_summary block into the profiling JSON.

        Parameters
        ----------
        csv_filepath   : path to the scaled CSV (display only)
        metadata       : already-loaded profiling JSON dict
        meta_path      : path where the JSON will be saved
        original_shape : (rows, cols) before reduction
        new_shape      : (rows, cols) after reduction
        reduction_log  : list of per-group dicts from apply_svd / apply_pca
        """
        metadata["dimension_reduction_summary"] = {
            "applied": True,
            "original_feature_count": original_shape[1],
            "reduced_feature_count": new_shape[1],
            "columns_removed": original_shape[1] - new_shape[1],
            "groups_processed": reduction_log,
            "timestamp": _now_iso(),
        }
        _save_json(metadata, meta_path)
        print(f"  Metadata updated  ->  {os.path.basename(meta_path)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_no_reduction_metadata(
        self,
        metadata: Dict[str, Any],
        meta_path: str,
        original_shape: Tuple[int, int],
    ) -> None:
        """Record a 'not applied' summary block in the profiling JSON."""
        metadata["dimension_reduction_summary"] = {
            "applied": False,
            "original_feature_count": original_shape[1],
            "reduced_feature_count": original_shape[1],
            "columns_removed": 0,
            "groups_processed": [],
            "timestamp": _now_iso(),
        }
        if meta_path:
            _save_json(metadata, meta_path)

    def _resolve_meta_path(self, csv_filepath: str) -> str:
        """
        Derive the profiling JSON path from the scaled CSV path.

        Example
        -------
        .../output/user1/netflix_titles_scaling.csv
            ->  .../meta_data/user1/netflix_titles_profiling.json
        """
        base = os.path.basename(csv_filepath)      # netflix_titles_scaling.csv
        stem = base.replace("_scaling.csv", "")    # netflix_titles
        return os.path.join(self.meta_dir, f"{stem}_profiling.json")

    def _print_banner(self) -> None:
        """Print the engine header."""
        print("\n" + "=" * 55)
        print("   DIMENSION REDUCTION ENGINE  v2.0")
        print("=" * 55)

    def _prompt_user_id(self) -> None:
        """Ask for a user-id and store it."""
        self.user_id = input("\nEnter user ID: ").strip()
        if not self.user_id:
            raise ValueError("User ID cannot be empty.")
        print(f"\nUser ID: {self.user_id}")

    def _resolve_directories(self) -> None:
        """Validate and set storage directories for the configured user."""
        self.output_dir = os.path.join(STORAGE_ROOT, "output",    self.user_id)
        self.meta_dir   = os.path.join(STORAGE_ROOT, "meta_data", self.user_id)

        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(
                f"Output directory not found for user '{self.user_id}':\n"
                f"  {self.output_dir}"
            )
        # meta_data dir may not exist yet -- create it
        os.makedirs(self.meta_dir, exist_ok=True)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    engine = DimensionReductionEngine()
    engine.run()