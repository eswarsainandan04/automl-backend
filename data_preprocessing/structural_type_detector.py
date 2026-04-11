import json
import os
import sys
from typing import Dict, Any, List

# Allow imports from the backend root when run directly
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from .supabase_storage import download_json, list_files, upload_json
except ImportError:
    from supabase_storage import download_json, list_files, upload_json


class StructuralTypeDetector:
    """
    Detects structural types for columns based on profiling metadata.
    """
    
    STRUCTURAL_TYPES = [
        "numeric",
        "categorical",
        "text",
        "datetime",
        "identifier",
        "unknown"
    ]
    
    DATETIME_SEMANTIC_TYPES = [
        "date", "time", "datetime", "timestamp", 
        "year", "month", "day", "week", "quarter",
        "fiscal_year", "timezone"
    ]
    
    NUMERIC_DTYPES = ["integer", "float", "numeric", "numeric_string"]
    
    NUMERIC_SEMANTIC_TYPES = [
        "integer", "float", "percentage", "ratio", 
        "currency", "salary", "angle", "area", "capacity",
        "density", "distance", "duration", "energy",
        "power", "pressure", "speed", "temperature",
        "volume", "weight"
    ]

    # Numeric columns with very limited unique values are treated as categorical.
    NUMERIC_CATEGORICAL_MAX_UNIQUE = 20
    NUMERIC_CATEGORICAL_MAX_RATIO = 0.20

    IDENTIFIER_SEMANTIC_TYPES = [
        "email",
        "url",
        "network_addresses",
        "file_path",
        "files",
        "version",
    ]
    IDENTIFIER_KEYWORDS = [
        "id",
        "_id",
        "no",
        "_no",
        "_number",
        "code",
        "uuid",
        "guid",
        "phone",
        "mobile",
        "telephone",
        "landline",
        "cell",
        "contact",
        "ssn",
        "passport",
        "account",
        "account_number",
        "transaction_id",
        "order_id",
        "customer",
        "customer_id",
        "user_id",
        "member_id",
        "ref_no"
    ]
    
    def detect(self, column_metadata: Dict[str, Any], total_rows: int) -> str:

        column_name = column_metadata.get("column_name", "")
        inferred_dtype = str(column_metadata.get("inferred_dtype", "")).strip().lower()
        unique_count = column_metadata.get("unique_count", 0)
        sample_values = column_metadata.get("sample_values", [])
        semantic_type = column_metadata.get("semantic_type", "")
        semantic_confidence = column_metadata.get("semantic_confidence", 0.0)
        unique_ratio = unique_count / total_rows if total_rows else 0

        # Normalize common string dtype aliases emitted by different profiling flows.
        if inferred_dtype in ("str", "string", "object", "varchar"):
            inferred_dtype = "text"

        if semantic_type == "time":
            return "time"
        if semantic_type == "date":
            return "date"

        # STEP 1 — Semantic-type based Identifier
        # (email, url, network_addresses, file_path, files, version are always identifiers)
        if semantic_type in self.IDENTIFIER_SEMANTIC_TYPES:
            return "identifier"

        # STEP 2 — Boolean-like values are treated as categorical only.
        if self._is_boolean(semantic_type, inferred_dtype, sample_values, unique_count):
            return "categorical"

        # STEP 3 — Name-based Identifier ONLY when unique ratio > 95%
        # (name match alone is not sufficient without high uniqueness)
        if self._name_based_identifier(column_name) and unique_ratio > 0.95:
            return "identifier"

        # STEP 3.5 — Sequential identifier: monotonic int sequences (1,2,3…)
        #             or prefixed-code sequences (EUR001, EUR002…)
        if self._is_sequential_identifier(column_metadata, total_rows):
            return "identifier"

        # STEP 4 — Datetime
        if self._is_datetime(semantic_type):
            return "datetime"

        # STEP 5 — Numeric / Numeric-Categorical
        if self._is_numeric(inferred_dtype, semantic_type):
            # Year-like columns (e.g., 1995, 2004) should remain numeric
            # even when unique count is low.
            if self._is_year_like_numeric(sample_values):
                return "numeric"
            if self._is_numeric_categorical(unique_count, total_rows):
                return "categorical"
            return "numeric"

        # STEP 6 — Statistical Identifier (before text classification)
        if self._is_identifier(column_metadata, total_rows):
            return "identifier"

        # STEP 7 — Text Handling
        if inferred_dtype == "text":

            avg_length = self._calculate_average_length(sample_values)

            unique_ratio = unique_count / total_rows if total_rows else 0

            # 1️⃣ Low cardinality → categorical (strong rule)
            if unique_count <= 50 and unique_ratio <= 0.5:
                return "categorical"

            # 2️⃣ Long strings → text
            if avg_length > 30:
                return "text"

            # 3️⃣ High uniqueness → text
            if unique_ratio > 0.9:
                return "text"

            return "categorical"

        return "unknown"
    


    def _is_boolean(self, semantic_type: str, inferred_dtype: str,
                    sample_values: List[Any], unique_count: int) -> bool:
        """
        Strict boolean detection.
        Only classify as boolean if:
        - unique_count <= 2
        - values are strictly binary patterns
        """

        # Strong semantic hint
        if semantic_type == "boolean":
            return True

        # Must have at most 2 unique values
        if unique_count > 2:
            return False

        if not sample_values:
            return False

        cleaned_values = set()

        for val in sample_values:
            if val is None:
                continue
            val_str = str(val).strip().lower()
            if val_str in ["null", "none", "nan", ""]:
                continue
            cleaned_values.add(val_str)

        valid_boolean_sets = [
            {"0", "1"},
            {"true", "false"},
            {"yes", "no"},
            {"t", "f"},
            {"y", "n"},
            {"0"},
            {"1"},
        ]

        for pattern in valid_boolean_sets:
            if cleaned_values.issubset(pattern):
                return True

        return False

        
        return False
    
    def _is_datetime(self, semantic_type: str) -> bool:
        """Check if column is datetime type."""
        return semantic_type in self.DATETIME_SEMANTIC_TYPES
    
    def _is_numeric(self, inferred_dtype: str, semantic_type: str) -> bool:
        """Check if column is numeric type."""
        if inferred_dtype in self.NUMERIC_DTYPES:
            return True
        if semantic_type in self.NUMERIC_SEMANTIC_TYPES:
            return True
        return False

    def _is_numeric_categorical(self, unique_count: int, total_rows: int) -> bool:
        """
        Check if a numeric column should be treated as categorical.
        This is intended for low-cardinality numeric columns like ratings/status
        codes (e.g., values in a small set such as 1, 2, 3).
        """
        if total_rows <= 0:
            return False

        unique_ratio = unique_count / total_rows

        return (
            unique_count <= self.NUMERIC_CATEGORICAL_MAX_UNIQUE
            and unique_ratio <= self.NUMERIC_CATEGORICAL_MAX_RATIO
        )

    def valid_year(self, value: Any) -> bool:
        """Return True when value is a valid 4-digit year."""
        if value is None:
            return False

        val_str = str(value).strip()
        if not val_str or val_str.lower() in {"null", "none", "nan"}:
            return False

        try:
            val_float = float(val_str)
        except ValueError:
            return False

        # Reject non-integer numbers such as 1995.5
        if val_float != int(val_float):
            return False

        year = int(val_float)
        # Keep strict 4-digit format and practical year range.
        return len(str(abs(year))) == 4 and 1000 <= year <= 2100

    def _is_year_like_numeric(self, sample_values: List[Any]) -> bool:
        """Detect columns where all observed values look like valid years."""
        values = []
        for v in sample_values:
            if v is None:
                continue
            val_str = str(v).strip()
            if not val_str or val_str.lower() in {"null", "none", "nan"}:
                continue
            values.append(v)

        if not values:
            return False

        return all(self.valid_year(v) for v in values)
    
    
    def _is_identifier(self, column_metadata: Dict[str, Any], total_rows: int) -> bool:
        if total_rows == 0:
            return False

        column_name = column_metadata.get("column_name", "")
        inferred_dtype = column_metadata.get("inferred_dtype", "")
        semantic_type = column_metadata.get("semantic_type", "")
        unique_count = column_metadata.get("unique_count", 0)
        sample_values = column_metadata.get("sample_values", [])

        # ❌ 1. Never classify numeric / datetime columns as identifier
        if inferred_dtype in self.NUMERIC_DTYPES:
            return False

        if semantic_type in self.NUMERIC_SEMANTIC_TYPES:
            return False

        if semantic_type in self.DATETIME_SEMANTIC_TYPES:
            return False

        unique_ratio = unique_count / total_rows

        # Require > 95% uniqueness AND name-based keyword match
        if unique_ratio <= 0.95:
            return False

        if not self._name_based_identifier(column_name):
            return False

        avg_length = self._calculate_average_length(sample_values)

        # Long natural text should not be identifier
        if avg_length > 30:
            return False

        # Optional monotonic check
        if self._is_monotonic(sample_values):
            return True

        # High uniqueness short structured values
        if avg_length <= 20:
            return True

        return False
    
    def _is_sequential_identifier(self, column_metadata: Dict[str, Any], total_rows: int) -> bool:
        """
        Detect identifier columns that contain sequential values even when
        the column name does not contain 'id':
          - Integer sequences:            222100201, 222100202, 222100203
          - Prefixed code sequences:      HCL24224501, HCL24224502, HCL24224503
          - Mixed alphanumeric sequences: 22HP1A1231,  22HP1A1232,  22HP1A1233
        Requires unique_ratio >= 0.90 to avoid flagging sparse numeric columns.
        """
        if total_rows == 0:
            return False

        unique_count = column_metadata.get("unique_count", 0)
        unique_ratio = unique_count / total_rows

        # Must be highly unique
        if unique_ratio < 0.90:
            return False

        inferred_dtype = column_metadata.get("inferred_dtype", "")
        sample_values  = column_metadata.get("sample_values", [])
        semantic_type  = column_metadata.get("semantic_type", "")

        # Skip proper datetime columns (years, months…)
        if semantic_type in self.DATETIME_SEMANTIC_TYPES:
            return False

        # ── Case 1: integer dtype sequence ───────────────────────────────────
        if inferred_dtype == "integer":
            return self._is_integer_sequence(sample_values)

        # ── Case 2: text/varchar — may be digit strings or prefixed sequences ─
        if inferred_dtype in ("text", "varchar", "numeric_string"):
            str_vals = [str(v).strip() for v in sample_values if v is not None]

            # Sub-case 2a: all values are pure digit strings (large IDs stored as text)
            if str_vals and all(sv.isdigit() for sv in str_vals):
                return self._is_integer_sequence(sample_values)

            # Sub-case 2b: mixed alphanumeric prefixed sequence
            return self._is_prefixed_sequence(sample_values)

        return False

    def _is_integer_sequence(self, sample_values: List[Any]) -> bool:
        """
        Return True when sample values are integers with a constant positive step.
        e.g.  222100201, 222100202, 222100203  →  True  (step = 1)
              1001, 1005, 1010, 1015            →  True  (step = 5)
        """
        try:
            vals = []
            for v in sample_values:
                if v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                f = float(s)
                if f != int(f):
                    return False
                vals.append(int(f))

            if len(vals) < 3:
                return False

            diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            return len(set(diffs)) == 1 and diffs[0] > 0
        except (TypeError, ValueError):
            return False

    def _is_prefixed_sequence(self, sample_values: List[Any]) -> bool:
        """
        Return True when all sample values share a common non-empty prefix and
        their remaining suffixes are digits forming a strictly incrementing
        arithmetic sequence.

        Handles any fixed-prefix + incrementing-suffix pattern, e.g.:
          HCL24224501, HCL24224502, HCL24224503   (alpha-digit prefix)
          22HP1A1231,  22HP1A1232,  22HP1A1233    (mixed alphanumeric prefix)
          EUR001, EUR002, EUR003                   (classic alpha + zero-padded)
        """
        str_vals = [str(v).strip() for v in sample_values if v is not None]
        if len(str_vals) < 3:
            return False

        # Compute the longest common prefix (LCP)
        lcp = str_vals[0]
        for sv in str_vals[1:]:
            while lcp and not sv.startswith(lcp):
                lcp = lcp[:-1]
            if not lcp:
                return False

        # Walk backward from LCP length to find the split where all suffixes
        # are purely numeric and form an arithmetic sequence.
        for split in range(len(lcp), 0, -1):
            prefix   = str_vals[0][:split]
            suffixes = []
            valid    = True
            for sv in str_vals:
                if not sv.startswith(prefix):
                    valid = False
                    break
                suffix = sv[split:]
                if not suffix or not suffix.isdigit():
                    valid = False
                    break
                suffixes.append(int(suffix))

            if valid and len(suffixes) >= 3:
                diffs = [suffixes[i + 1] - suffixes[i] for i in range(len(suffixes) - 1)]
                if len(set(diffs)) == 1 and diffs[0] > 0:
                    return True

        return False

    def _name_based_identifier(self, column_name: str) -> bool:
        import re

        column_name = column_name.lower()

        # Strong exact matches
        if column_name in self.IDENTIFIER_KEYWORDS:
            return True

        # Match *_id or id_*
        if re.search(r'(^id_|_id$)', column_name):
            return True

        # Match camelCase / no-separator "id" suffix — e.g. "passengerid", "customerid"
        if re.search(r'[a-z]id$', column_name):
            return True

        # Strong identifier patterns
        strong_patterns = [
            r'phone',
            r'mobile',
            r'ssn',
            r'passport',
            r'uuid',
            r'guid'
        ]

        for pattern in strong_patterns:
            if re.search(pattern, column_name):
                return True

        # Match patterns like transaction_id, customer_id, etc.
        if re.search(r'.*_id$', column_name):
            return True

        return False
    
    
    def _is_monotonic(self, sample_values: List[Any]) -> bool:
        """
        Check if values are strictly increasing or decreasing.
        Useful for detecting sequential IDs.
        """
        try:
            values = []

            for v in sample_values:
                if v is None:
                    continue
                val_str = str(v).strip()
                if val_str.upper() in ["NULL", "NONE", "NAN", ""]:
                    continue
                values.append(float(val_str))

            if len(values) < 3:
                return False

            increasing = all(x < y for x, y in zip(values, values[1:]))
            decreasing = all(x > y for x, y in zip(values, values[1:]))

            return increasing or decreasing

        except:
            return False
    
    
    def _classify_text(self, sample_values: List[Any], unique_count: int, total_rows: int) -> str:
        """Classify text column as either 'text' or 'categorical'."""
        # Calculate average length of sample values
        avg_length = self._calculate_average_length(sample_values)
        
        # Calculate unique ratio
        if total_rows == 0:
            unique_ratio = 0
        else:
            unique_ratio = unique_count / total_rows
        
        # Rule: If average length > 30, it's text
        if avg_length > 30:
            return "text"
        
        # Rule: If unique_ratio <= 0.5, it's categorical
        if unique_ratio <= 0.5:
            return "categorical"
        
        # Default to categorical
        return "categorical"
    
    def _calculate_average_length(self, sample_values: List[Any]) -> float:
        """Calculate average length of sample values."""
        if not sample_values:
            return 0.0
        
        total_length = 0
        count = 0
        
        for val in sample_values:
            if val is not None and val != "":
                # Skip NULL representations
                val_str = str(val)
                if val_str.upper() not in ["NULL", "NONE", "NAN"]:
                    total_length += len(val_str)
                    count += 1
        
        if count == 0:
            return 0.0
        
        return total_length / count


class ProfilingEnhancer:
    """
    Enhances profiling JSON (stored in Supabase) with structural type information.
    """

    def __init__(self, user_id: str, session_id: str):
        self._download_json = download_json
        self._list_files    = list_files
        self._upload_json   = upload_json
        self.user_id    = user_id
        self.session_id = session_id
        self.meta_prefix = f"meta_data/{user_id}/{session_id}"
        self.detector = StructuralTypeDetector()

    def enhance_all(self):
        """Detect and write structural_type for every profiling file in the session."""
        try:
            all_files = self._list_files(self.meta_prefix)
        except Exception as exc:
            print(f"Error listing files at {self.meta_prefix}: {exc}")
            return

        profiling_files = [f for f in all_files if f.endswith("_profiling.json")]

        if not profiling_files:
            print(f"No profiling files found at {self.meta_prefix}")
            return

        print(f"Found {len(profiling_files)} profiling file(s):")
        for pf in profiling_files:
            self.enhance_profiling(pf)

    def enhance_profiling(self, profiling_filename: str):
        """
        Download one *_profiling.json from Supabase, add structural_type to
        each column, then upload it back.

        Args:
            profiling_filename: e.g. "myfile_profiling.json"
        """
        remote_path = f"{self.meta_prefix}/{profiling_filename}"

        try:
            profiling_data = self._download_json(remote_path)
        except Exception as exc:
            print(f"Error downloading {remote_path}: {exc}")
            return

        total_rows = profiling_data.get("number_of_rows", 0)
        column_wise_summary = profiling_data.get("column_wise_summary", [])

        for column in column_wise_summary:
            column["structural_type"] = self.detector.detect(column, total_rows)

        try:
            self._upload_json(remote_path, profiling_data)
        except Exception as exc:
            print(f"Error uploading {remote_path}: {exc}")
            return

        base_name = profiling_filename.replace("_profiling.json", "")
        print(f"✓ Enhanced profiling for {base_name}")
        self._print_summary(column_wise_summary)

    def _print_summary(self, columns: List[Dict[str, Any]]):
        """Print summary of structural types detected."""
        type_counts = {}
        for col in columns:
            st = col.get("structural_type", "unknown")
            type_counts[st] = type_counts.get(st, 0) + 1

        print("\n  Structural Type Distribution:")
        for stype, count in sorted(type_counts.items(), key=lambda x: str(x[0])):
            display = stype if stype is not None else "unknown"
            print(f"    - {display}: {count}")


class ConstraintGenerator:
    """
    Generates structural constraints for all datasets in a session:
      - primary_keys : columns detected as structural_type == "identifier"
      - unique_keys  : columns where unique_count == total_rows (excluding PKs)
      - foreign_keys : column names shared across two or more datasets

    Saves result to  meta_data/{user_id}/{session_id}/constraints.json
    and also returns the dict.
    """

    def __init__(self, user_id: str, session_id: str):
        self.user_id    = user_id
        self.session_id = session_id
        self.meta_prefix = f"meta_data/{user_id}/{session_id}"
        self.detector = StructuralTypeDetector()

    def generate(self) -> dict:
        from datetime import datetime, timezone
        from collections import Counter

        try:
            all_files = list_files(self.meta_prefix)
        except Exception:
            all_files = []

        profiling_files = sorted(f for f in all_files if f.endswith("_profiling.json"))

        dataset_pks:  dict[str, list] = {}
        dataset_uks:  dict[str, list] = {}
        dataset_cols: dict[str, set]  = {}

        for pf in profiling_files:
            dataset_name = pf.replace("_profiling.json", "")
            try:
                profiling = download_json(f"{self.meta_prefix}/{pf}")
            except Exception:
                continue

            total_rows  = profiling.get("number_of_rows", 0)
            col_summary = profiling.get("column_wise_summary", [])

            # Re-run detection fresh to override any stale stored values
            for col in col_summary:
                col["structural_type"] = self.detector.detect(col, total_rows)

            pks = [
                c["column_name"] for c in col_summary
                if c.get("structural_type") == "identifier"
            ]
            pk_set = set(pks)

            uks = [
                c["column_name"] for c in col_summary
                if c.get("unique_count") == total_rows
                and total_rows > 0
            ]

            dataset_pks[dataset_name]  = pks
            dataset_uks[dataset_name]  = uks
            dataset_cols[dataset_name] = {c["column_name"] for c in col_summary}

        # Foreign keys: column names present in 2+ datasets
        col_counter: Counter = Counter()
        for cols in dataset_cols.values():
            for col in cols:
                col_counter[col] += 1
        fk_col_names = {col for col, cnt in col_counter.items() if cnt >= 2}

        foreign_keys = [
            {
                "column_name": col,
                "datasets": sorted(ds for ds, cols in dataset_cols.items() if col in cols),
            }
            for col in sorted(fk_col_names)
        ]

        primary_keys = [
            {"dataset_name": ds, "columns": cols}
            for ds, cols in sorted(dataset_pks.items())
            if cols
        ]

        unique_keys = [
            {"dataset_name": ds, "columns": cols}
            for ds, cols in sorted(dataset_uks.items())
            if cols
        ]

        constraints = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "session_id":   self.session_id,
            "primary_keys": primary_keys,
            "unique_keys":  unique_keys,
            "foreign_keys": foreign_keys,
        }

        try:
            upload_json(f"{self.meta_prefix}/constraints.json", constraints)
        except Exception:
            pass

        return constraints


def main():
    """Main execution function."""
    print("=" * 60)
    print("Structural Type Detector")
    print("=" * 60)

    user_id    = input("\nEnter user_id: ").strip()
    session_id = input("Enter session_id: ").strip()

    if not user_id or not session_id:
        print("Error: user_id and session_id cannot be empty")
        return

    enhancer = ProfilingEnhancer(user_id, session_id)

    print("\n" + "=" * 60)
    print("Processing profiling files…")
    print("=" * 60 + "\n")

    enhancer.enhance_all()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)