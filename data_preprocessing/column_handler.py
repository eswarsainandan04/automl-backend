"""
Column Handler Module for AutoML Preprocessing System

This module performs semantic detection and column-level normalization using
pattern-based detection from Python pattern classes.

Author: AutoML Preprocessing System
Date: 2025-12-23
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

try:
    from .supabase_storage import download_file, upload_file, upload_json, download_json, list_files
except ImportError:
    from supabase_storage import download_file, upload_file, upload_json, download_json, list_files

# Import pattern classes
try:
    from .patterns import PATTERNS
except ImportError:
    from patterns import PATTERNS


# ==============================
# UNIT / SYMBOL DETECTION FOR MIXED-TYPE COLUMNS
# ==============================

# Semantic types whose patterns represent physical units or symbols.
# These are checked when a mixed-type column has minority string values
# (e.g. 13 numbers + 1 "6$") to decide if the symbol defines the column.
_UNIT_SEMANTIC_TYPES = frozenset([
    'currency', 'percentage', 'speed', 'energy', 'power', 'pressure',
    'capacity', 'density', 'area', 'distance', 'weight', 'volume',
    'temperature', 'angle', 'salary',
])


def _detect_units_in_strings(string_values: list) -> Optional[dict]:
    """
    Detect a unit/symbol pattern in a list of string values using the
    existing pattern classes from patterns/.

    Iterates over all unit-type pattern classes, calls detect() on the
    string values, and returns the best-matching semantic type if its
    confidence is ≥ 50 %.

    Returns:
        dict with keys {semantic_type, unit, matched_count, total_strings}
        if a pattern matched, else None.
    """
    if not string_values:
        return None

    # Run each unit-type pattern's detect() on the string values
    unit_scores = {}   # sem_type -> confidence  (unit types only)

    for sem_type, pattern_instance in PATTERNS.items():
        if sem_type not in _UNIT_SEMANTIC_TYPES:
            continue
        try:
            conf = pattern_instance.detect(string_values)
            if conf >= 0.5:
                unit_scores[sem_type] = conf
        except Exception:
            continue

    if not unit_scores:
        return None

    # Sort by confidence descending
    ranked = sorted(unit_scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_conf = ranked[0]

    # AMBIGUITY CHECK: if multiple unit patterns tied at the same confidence,
    # the value is ambiguous — reject.
    unit_ties = [t for t, c in ranked if c == best_conf]
    if len(unit_ties) > 1:
        return None

    # CROSS-CHECK against non-unit patterns: if a non-unit pattern also
    # matches at the same or higher confidence, the value is generic
    # (e.g. "3.0M" matches distance *and* varchar/number_systems at 1.0).
    for sem_type, pattern_instance in PATTERNS.items():
        if sem_type in _UNIT_SEMANTIC_TYPES:
            continue
        # Skip very broad patterns that match almost anything
        if sem_type in ('text', 'varchar'):
            continue
        try:
            conf = pattern_instance.detect(string_values)
            if conf >= best_conf:
                # A non-unit pattern matches equally well → ambiguous, reject
                return None
        except Exception:
            continue

    # Derive the unit label from the pattern instance
    pat = PATTERNS[best_type]
    # First run normalize so detected_unit / detected_currency is populated
    try:
        pat.normalize(pd.Series(string_values))
    except Exception:
        pass

    if best_type == 'currency':
        unit_label = getattr(pat, 'detected_currency', None) or 'USD'
    elif best_type == 'percentage':
        unit_label = '%'
    else:
        unit_label = getattr(pat, 'detected_unit', best_type)

    matched_count = max(1, int(best_conf * len(string_values)))

    return {
        'semantic_type': best_type,
        'unit':          unit_label,
        'matched_count': matched_count,
        'total_strings': len(string_values),
    }


# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_mixed_type_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and clean columns with mixed string and numeric data.
    Before cleaning, inspects minority string values for unit/symbol patterns
    (e.g. '$', 'km', '%').  If a consistent unit is found the numeric part is
    extracted and the unit is recorded for later column renaming.  Otherwise,
    the majority type is kept and minority values are replaced with empty string.

    Examples:
        No unit:    [34, 51, 'Unknown', 896]      -> [34, 51, '', 896]
        With unit:  [0, 0, '$5', 0, '$2']          -> [0, 0, '5', 0, '2']
                    unit_detections["Price"] = {semantic_type:'currency', unit:'USD'}

    Args:
        df: Pandas DataFrame

    Returns:
        Tuple of (cleaned DataFrame, unit_detections dict).
        unit_detections maps column_name -> {semantic_type, unit, regex,
        matched_count, total_strings}.
    """
    df_cleaned = df.copy()
    unit_detections = {}
    
    for col in df_cleaned.columns:
        # Only process object/string columns (already numeric columns are fine)
        if df_cleaned[col].dtype == 'object':
            # Get non-null values
            non_null_values = df_cleaned[col].dropna()
            
            if len(non_null_values) == 0:
                continue
            
            # Count numeric vs string values and collect strings for unit detection
            numeric_count = 0
            string_count = 0
            string_values = []
            
            for val in non_null_values:
                val_str = str(val).strip()
                
                # Try to determine if it's numeric
                try:
                    # Remove common numeric separators
                    test_val = val_str.replace(',', '').replace(' ', '')
                    float(test_val)
                    numeric_count += 1
                except (ValueError, AttributeError):
                    string_count += 1
                    string_values.append(val_str)
            
            # Check if there's a mix (both types present)
            if numeric_count > 0 and string_count > 0:
                total_count = numeric_count + string_count
                numeric_pct = (numeric_count / total_count) * 100
                string_pct = (string_count / total_count) * 100
                
                print(f"     [MIXED TYPE DETECTED] {col}: numeric={numeric_count} ({numeric_pct:.1f}%), string={string_count} ({string_pct:.1f}%)")
                
                # Determine majority type
                keep_numeric = numeric_count >= string_count
                
                if keep_numeric:
                    # BEFORE cleaning: try to detect units/symbols in minority strings
                    detected = _detect_units_in_strings(string_values)
                    
                    if detected:
                        # Unit pattern found — use pattern normalize() to extract numbers
                        unit_detections[col] = detected
                        sem_type = detected['semantic_type']
                        pattern_instance = PATTERNS[sem_type]
                        print(f"       [UNIT DETECTED] {sem_type} ({detected['unit']}) "
                              f"in {detected['matched_count']}/{detected['total_strings']} string values")
                        print(f"       [EXTRACTING] Numbers from unit-bearing strings, "
                              f"non-matching strings set to empty")
                        
                        # Use pattern normalize() on the string values to get numbers
                        try:
                            norm_series = pattern_instance.normalize(pd.Series(string_values))
                            # Build a lookup: original string → normalized numeric
                            norm_lookup = {}
                            for orig, normed in zip(string_values, norm_series):
                                if pd.notna(normed):
                                    norm_lookup[orig] = normed
                        except Exception:
                            norm_lookup = {}
                        
                        def extract_unit_value(val, _lookup=norm_lookup):
                            if pd.isna(val):
                                return ''
                            val_str = str(val).strip()
                            # If already numeric, keep it
                            try:
                                test_val = val_str.replace(',', '').replace(' ', '')
                                float(test_val)
                                return val
                            except (ValueError, AttributeError):
                                pass
                            # Try to get normalized value from the pattern
                            if val_str in _lookup:
                                return _lookup[val_str]
                            return ''  # unrecognised string
                        
                        df_cleaned[col] = df_cleaned[col].apply(extract_unit_value)
                    else:
                        # No unit detected — regular cleaning (blank out strings)
                        print(f"       [CLEANING] Keeping numeric values, removing {string_count} string values (set to empty)")
                        
                        # Replace string values with empty string
                        def clean_value(val):
                            if pd.isna(val):
                                return ''  # Convert NaN to empty string for consistency
                            val_str = str(val).strip()
                            try:
                                test_val = val_str.replace(',', '').replace(' ', '')
                                float(test_val)
                                return val  # Keep numeric value
                            except (ValueError, AttributeError):
                                return ''  # Replace string with empty string
                        
                        df_cleaned[col] = df_cleaned[col].apply(clean_value)
                else:
                    print(f"       [CLEANING] Keeping string values, removing {numeric_count} numeric values (set to empty)")
                    
                    # Replace numeric values with empty string
                    def clean_value(val):
                        if pd.isna(val):
                            return ''  # Convert NaN to empty string for consistency
                        val_str = str(val).strip()
                        try:
                            test_val = val_str.replace(',', '').replace(' ', '')
                            float(test_val)
                            return ''  # Replace numeric with empty string
                        except (ValueError, AttributeError):
                            return val  # Keep string value
                    
                    df_cleaned[col] = df_cleaned[col].apply(clean_value)
    
    return df_cleaned, unit_detections


def remove_special_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove special symbols from all string columns:
    | ~ ( ) [ ] { } " ' ` ! ?
    
    Args:
        df: Pandas DataFrame
    
    Returns:
        DataFrame with symbols removed from string columns
    """
    # Define symbols to remove - using hex codes to avoid quote conflicts
    # Removes: | ~ ( ) [ ] { } " ' ` ! ? + -
    symbols_pattern = r"[|~()\[\]{}\"'`!?+]"
    
    # Create a copy to avoid SettingWithCopyWarning
    df_cleaned = df.copy()
    
    # Process each column
    for col in df_cleaned.columns:
        # Only process object/string columns
        if df_cleaned[col].dtype == 'object':
            # Remove symbols from string values
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: re.sub(symbols_pattern, '', str(x)) if pd.notna(x) else x
            )
    
    return df_cleaned


def detect_numeric_format(series: pd.Series) -> str:
    """
    Detect whether a numeric column should use integer or float format.
    If ANY value has a decimal part, the entire column uses float format.
    
    Args:
        series: Pandas Series to analyze
    
    Returns:
        'float' if any value has decimal part, 'int' otherwise
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'float'
    
    # Check if ANY value has a decimal part
    for val in non_null:
        try:
            # Parse the numeric value
            if isinstance(val, str):
                val_str = val.strip()
                # Remove currency symbols, commas, etc.
                val_str = re.sub(r'[^\d.\-]', '', val_str)
                num = float(val_str)
            else:
                num = float(val)
            
            # If ANY value has decimal part, return 'float'
            if num != int(num):
                return 'float'
        except:
            continue
    
    # All values are integers
    return 'int'





def get_sample_values(series: pd.Series, n: int = 5) -> list:
    """
    Extract first n non-null sample values from a Series.
    
    Args:
        series: Pandas Series to sample from
        n: Number of samples to extract
    
    Returns:
        List of sample values (converted to native Python types)
    """
    non_null = series.dropna()
    samples = non_null.head(n).tolist()
    
    # Convert numpy types to native Python types for JSON serialization
    result = []
    for val in samples:
        if isinstance(val, (np.integer, np.floating)):
            result.append(val.item())
        elif isinstance(val, np.bool_):
            result.append(bool(val))
        elif pd.isna(val):
            result.append(None)
        else:
            result.append(str(val))
    
    return result


def infer_normalized_dtype(series: pd.Series) -> str:
    """
    Infer the data type of a pandas Series after normalization.
    
    Args:
        series: Pandas Series to analyze
    
    Returns:
        String representation of the inferred data type
    """
    # Get pandas dtype
    dtype = series.dtype
    
    # Map pandas dtypes to readable types
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    elif pd.api.types.is_float_dtype(dtype):
        return "float"
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    elif pd.api.types.is_categorical_dtype(dtype):
        return "categorical"
    elif pd.api.types.is_object_dtype(dtype):
        # Try to infer if it's numeric stored as string
        non_null = series.dropna()
        if len(non_null) > 0:
            try:
                pd.to_numeric(non_null.head(100), errors='raise')
                return "numeric_string"
            except (ValueError, TypeError):
                return "text"
        return "text"
    else:
        return str(dtype)


# ==============================
# PATTERN LOADING
# ==============================

def load_patterns():
    """
    Load all semantic pattern classes.
    
    Returns:
        Dictionary mapping semantic_type to pattern instance
    """
    print(f"   Loaded {len(PATTERNS)} pattern classes:")
    for semantic_type in PATTERNS.keys():
        print(f"     - {semantic_type}")
    
    return PATTERNS


# ==============================
# METADATA OPERATIONS
# ==============================

def load_profiling_metadata(userid: str, sessionid: str, filename: str) -> Dict[str, Any]:
    """
    Load existing profiling metadata from Supabase storage.
    """
    base_name = os.path.splitext(filename)[0]
    path = f"meta_data/{userid}/{sessionid}/{base_name}_profiling.json"
    try:
        return download_json(path)
    except Exception as e:
        print(f"    Failed to load profiling metadata: {e}")
        return None


def save_profiling_metadata(metadata: Dict[str, Any], userid: str, sessionid: str, filename: str):
    """
    Save updated profiling metadata to Supabase storage.
    """
    base_name = os.path.splitext(filename)[0]
    path = f"meta_data/{userid}/{sessionid}/{base_name}_profiling.json"
    upload_json(path, metadata)


# ==============================
# SEMANTIC DETECTION
# ==============================

def detect_mixed_types(column: pd.Series, patterns: Dict) -> Tuple[bool, List[str]]:
    """
    Detect if a column contains mixed incompatible formats.
    
    Examples of mixed types:
    - Column with both unix timestamps AND datetime strings
    - Column with both dates AND numbers
    - Column with multiple currency formats
    
    Args:
        column: Pandas Series to analyze
        patterns: Dictionary of pattern instances
    
    Returns:
        Tuple of (is_mixed, detected_types_list)
    """
    non_null = column.dropna()
    if len(non_null) == 0:
        return False, []
    
    # Sample values
    sample_values = non_null.head(min(100, len(non_null))).tolist()
    
    # Detect what percentage of values match each pattern
    pattern_matches = {}
    
    for semantic_type, pattern_instance in patterns.items():
        try:
            # Count how many values match this pattern
            matched_count = 0
            for val in sample_values:
                # Quick individual value check
                confidence = pattern_instance.detect([val])
                if confidence > 0.5:
                    matched_count += 1
            
            if matched_count > 0:
                match_percentage = matched_count / len(sample_values)
                pattern_matches[semantic_type] = match_percentage
        except:
            continue
    
    # Check for incompatible pattern combinations
    detected_types = [t for t, pct in pattern_matches.items() if pct > 0.2]
    
    # Define incompatible pattern groups
    incompatible_groups = [
        {'timestamp', 'datetime', 'date', 'time'},  # Temporal types are OK together if consistent
        {'year', 'month', 'day', 'week'},  # Temporal components
    ]
    
    # Check for truly mixed patterns (e.g., dates AND numbers)
    temporal_types = {'timestamp', 'datetime', 'date', 'time'}
    numeric_types = {'currency', 'price', 'salary', 'revenue', 'expense'}
    
    has_temporal = any(t in detected_types for t in temporal_types)
    has_numeric = any(t in detected_types for t in numeric_types)
    
    # Special case: number_systems can coexist with integer/year/time (binary 0101 might match year pattern)
    # Don't mark as mixed if number_systems is detected
    if 'number_systems' in detected_types:
        # Check if number_systems has significant match percentage
        if pattern_matches.get('number_systems', 0) >= 0.6:
            # Number systems has high confidence - not mixed, just ambiguous
            return False, []
    
    # Mixed if has BOTH temporal and numeric, or multiple temporal formats with low individual confidence
    if has_temporal and has_numeric:
        return True, detected_types
    
    # Check if multiple temporal types but none has >80% match (indicates format mixing)
    temporal_in_column = [t for t in detected_types if t in temporal_types]
    if len(temporal_in_column) > 1:
        max_temporal_confidence = max(pattern_matches.get(t, 0) for t in temporal_in_column)
        if max_temporal_confidence < 0.8:
            return True, detected_types
    
    return False, detected_types


def infer_basic_type(column: pd.Series) -> str:
    """
    Infer basic data type when no semantic pattern matches.
    Returns one of: 'integer', 'float', 'varchar', 'text'
    
    Args:
        column: Pandas Series to analyze
        
    Returns:
        Basic type string ('integer', 'float', 'varchar', 'text')
    """
    non_null = column.dropna()
    if len(non_null) == 0:
        return 'text'
    
    # Try to parse as numeric
    try:
        numeric = pd.to_numeric(non_null, errors='coerce')
        non_null_numeric = numeric.dropna()
        
        # Check if most values converted successfully
        if len(non_null_numeric) / len(non_null) > 0.8:
            # Numeric - check if integer or float
            has_decimal = False
            for val in non_null_numeric:
                if val != int(val):
                    has_decimal = True
                    break
            
            return 'float' if has_decimal else 'integer'
    except:
        pass
    
    # Not numeric - check if varchar (short strings) or text (longer strings)
    avg_length = non_null.astype(str).str.len().mean()
    
    # Varchar typically < 255 chars, text is longer
    if avg_length < 100:
        return 'varchar'
    else:
        return 'text'


def detect_semantic(column: pd.Series, patterns: Dict, 
                   confidence_threshold: float = 0.5) -> Tuple[str, float, Optional[str]]:
    """
    Detect semantic type of a column using pattern classes.
    CRITICAL: Uses RAW values only - never cleaned/transformed values.
    
    Detection logic:
    1. If column matches pattern ≥50% → assign that semantic type and normalize
    2. If no pattern matches ≥50% → fall back to basic type (int/float/varchar/text), do NOT normalize
    3. If column has mixed patterns → mark as 'mixed', do NOT normalize
    
    Args:
        column: Pandas Series to analyze (RAW values)
        patterns: Dictionary of pattern instances
        confidence_threshold: Minimum confidence (50%) to assign semantic type
    
    Returns:
        Tuple of (semantic_type, confidence, note)
    """
    # PRIORITY CHECK: Detect mixed types FIRST
    is_mixed, detected_types = detect_mixed_types(column, patterns)
    if is_mixed:
        return "mixed", 1.0, f"Mixed formats detected: {', '.join(detected_types)}"
    
    # Sample non-null values (RAW only)
    non_null = column.dropna()
    if len(non_null) == 0:
        return "unknown", 0.0, "No non-null values"
    
    # Take sample (max 100 values for performance)
    # IMPORTANT: Keep as a Series so patterns can use column context via `values.name`
    sample_size = min(100, len(non_null))
    sample_series = non_null.head(sample_size)
    sample_values = sample_series.tolist()
    
    # Test against all patterns
    pattern_scores = {}
    
    for semantic_type, pattern_instance in patterns.items():
        try:
            confidence = pattern_instance.detect(sample_series)
            pattern_scores[semantic_type] = confidence
        except Exception as e:
            print(f"        Error detecting {semantic_type}: {e}")
            pattern_scores[semantic_type] = 0.0
    
    # Find best match
    if not pattern_scores:
        return "unknown", 0.0, "No patterns available"
    
    sorted_scores = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
    best_semantic, best_confidence = sorted_scores[0]
    priority_override_applied = False  # Track if we applied a priority override
    
    # ============================================================
    # MANDATORY PRIORITY OVERRIDES (Dynamic Semantic Rules)
    # These overrides are FINAL and cannot be changed by ambiguity resolution
    # ============================================================
    
    # Priority 1: YEAR vs CURRENCY - 4-digit values between 1900-2100
    # Check ALL competing patterns, not just year vs currency
    if 'year' in pattern_scores and pattern_scores['year'] > 0.5:
        # Check if values are 4-digit years without currency symbols
        has_currency_symbol = any(
            '$' in str(v) or 'EUR' in str(v) or 'GBP' in str(v) or 'INR' in str(v) or 'JPY' in str(v) or 'USD' in str(v)
            for v in sample_values if v is not None
        )
        if not has_currency_symbol:
            # Check if values are 4-digit numbers in year range (1900-2100)
            year_like_values = []
            for v in sample_values:
                if v is not None:
                    v_str = str(v).strip()
                    # Must be exactly 4 digits
                    if v_str.isdigit() and len(v_str) == 4:
                        year_val = int(v_str)
                        if 1900 <= year_val <= 2100:
                            year_like_values.append(year_val)
            
            # If >70% of values are valid years, force YEAR semantic type
            if len(year_like_values) / len(sample_values) > 0.7:
                best_semantic = 'year'
                best_confidence = max(pattern_scores['year'], 0.85)
                priority_override_applied = True
                print(f"       [OVERRIDE] YEAR priority applied (detected {len(year_like_values)} year values)")
    
    # Priority 2: RATIO vs TIME - a:b or a/b pattern override
    if not priority_override_applied and 'ratio' in pattern_scores and pattern_scores['ratio'] > 0.5:
        # Check for ratio patterns like 3:05, 2:03, 4:06
        # CRITICAL: Ratio should have numbers on both sides that could be any value
        # Time is typically HH:MM format with MM < 60
        ratio_like_count = 0
        time_like_count = 0
        
        for v in sample_values:
            if v is not None:
                v_str = str(v).strip()
                # Match a:b or a/b pattern
                if re.match(r'^\d+:\d+$', v_str):
                    parts = v_str.split(':')
                    if len(parts) == 2:
                        left, right = int(parts[0]), int(parts[1])
                        # If right side is > 59, it's likely a ratio, not time
                        # Or if the pattern doesn't fit time constraints
                        if right > 59 or left > 23:
                            ratio_like_count += 1
                        else:
                            # Could be time (0-23:0-59)
                            time_like_count += 1
                elif re.match(r'^\d+[/-]\w+$', v_str):  # Like "2-Jan"
                    ratio_like_count += 1
        
        # If majority are ratio-like, override
        if ratio_like_count > time_like_count:
            best_semantic = 'ratio'
            best_confidence = pattern_scores['ratio']
            priority_override_applied = True
            print(f"       [OVERRIDE] RATIO priority applied (ratio={ratio_like_count}, time={time_like_count})")
    
    # Priority 3: DISCOUNT/TAX/INTEREST/COMMISSION - keyword detection
    # Check for specific keywords in column name or values
    column_name_lower = column.name.lower() if hasattr(column, 'name') and column.name else ''
    
    if not priority_override_applied:
        if 'discount' in column_name_lower and 'discount' in pattern_scores:
            if pattern_scores['discount'] > 0.4:
                best_semantic = 'discount'
                best_confidence = pattern_scores['discount']
                priority_override_applied = True
                print(f"       [OVERRIDE] DISCOUNT keyword applied")
        elif 'tax' in column_name_lower and 'tax' in pattern_scores:
            if pattern_scores['tax'] > 0.4:
                best_semantic = 'tax'
                best_confidence = pattern_scores['tax']
                priority_override_applied = True
                print(f"       [OVERRIDE] TAX keyword applied")
        elif 'interest' in column_name_lower and 'interest_rate' in pattern_scores:
            if pattern_scores['interest_rate'] > 0.4:
                best_semantic = 'interest_rate'
                best_confidence = pattern_scores['interest_rate']
                priority_override_applied = True
                print(f"       [OVERRIDE] INTEREST_RATE keyword applied")
        elif 'commission' in column_name_lower and 'commission' in pattern_scores:
            if pattern_scores['commission'] > 0.4:
                best_semantic = 'commission'
                best_confidence = pattern_scores['commission']
                priority_override_applied = True
                print(f"       [OVERRIDE] COMMISSION keyword applied")
    
    # Priority 4: BOOLEAN vs INTEGER - columns with only 0/1 values
    # Check if column has both TRUE-like and FALSE-like values (e.g. 0 AND 1)
    if not priority_override_applied and 'boolean' in pattern_scores and pattern_scores['boolean'] > 0.8:
        # Check if values are ONLY boolean keywords AND contain BOTH sides
        unique_vals = set()
        for v in sample_values:
            if v is not None and not pd.isna(v):
                try:
                    val_str = str(v).strip().lower()
                    unique_vals.add(val_str)
                except:
                    continue
        
        # Boolean keywords recognised by the pattern
        boolean_true_keywords = {'1', 'true', 'yes', 'on', 'active', 'enabled'}
        boolean_false_keywords = {'0', 'false', 'no', 'off', 'inactive', 'disabled'}
        boolean_value_keywords = boolean_true_keywords | boolean_false_keywords
        
        # ALL unique values must be boolean keywords AND column must have
        # both a true-like and a false-like value (not just all 0s).
        has_true_side = bool(unique_vals & boolean_true_keywords)
        has_false_side = bool(unique_vals & boolean_false_keywords)
        
        if (len(unique_vals) > 0
                and unique_vals.issubset(boolean_value_keywords)
                and has_true_side and has_false_side):
            best_semantic = 'boolean'
            best_confidence = max(pattern_scores['boolean'], 0.9)
            priority_override_applied = True
            print(f"       [OVERRIDE] BOOLEAN priority applied (only boolean values: {unique_vals})")
    
    # Priority 5: MEASUREMENT vs VARCHAR - measurement patterns override varchar
    # VARCHAR matches any alphanumeric (like "1L", "500 ml") but measurement patterns are more specific
    if not priority_override_applied:
        measurement_types = [
            'distance', 'weight', 'volume', 'area', 'speed', 'temperature',
            'pressure', 'energy', 'power', 'capacity', 'density', 'angle'
        ]
        
        # Check if best match is varchar but a measurement pattern also matches well
        if best_semantic == 'varchar':
            for meas_type in measurement_types:
                if meas_type in pattern_scores and pattern_scores[meas_type] >= 0.6:
                    # Override varchar with measurement type
                    best_semantic = meas_type
                    best_confidence = pattern_scores[meas_type]
                    priority_override_applied = True
                    print(f"       [OVERRIDE] {meas_type.upper()} priority over VARCHAR (confidence: {best_confidence:.2f})")
                    break
    
    # Priority 6: SALARY vs VARCHAR - salary pattern overrides varchar
    # VARCHAR matches any alphanumeric including salary strings, but salary is more specific
    if not priority_override_applied:
        if best_semantic == 'varchar' and 'salary' in pattern_scores and pattern_scores['salary'] >= 0.6:
            # Override varchar with salary
            best_semantic = 'salary'
            best_confidence = pattern_scores['salary']
            priority_override_applied = True
            print(f"       [OVERRIDE] SALARY priority over VARCHAR (confidence: {best_confidence:.2f})")
    
    # Priority 7: NUMBER_SYSTEMS vs INTEGER/VARCHAR - number systems (hex, binary, octal) override basic types
    # INTEGER/VARCHAR match number system values, but number_systems is more specific
    if not priority_override_applied:
        if best_semantic in ['integer', 'varchar'] and 'number_systems' in pattern_scores and pattern_scores['number_systems'] >= 0.6:
            # Override with number_systems
            best_semantic = 'number_systems'
            best_confidence = pattern_scores['number_systems']
            priority_override_applied = True
            print(f"       [OVERRIDE] NUMBER_SYSTEMS priority over {best_semantic.upper()} (confidence: {best_confidence:.2f})")
    
    # Check confidence threshold (must be >= 0.5 = 50%)
    if best_confidence < confidence_threshold:
        # No pattern matched with ≥50% confidence
        # Fall back to basic data types (int/float/varchar/text) - will NOT be normalized
        basic_type = infer_basic_type(column)
        return basic_type, 0.0, f"No semantic pattern matched (best: {best_semantic} {best_confidence:.0%}), using basic type"
    
    # Pattern matched with ≥50% confidence - will be normalized
    return best_semantic, best_confidence, None


# ==============================
# COLUMN NORMALIZATION
# ==============================

def normalize_column(column: pd.Series, semantic_type: str, patterns: Dict, numeric_format: str = 'float') -> pd.Series:
    """
    Normalize column using pattern class normalize method.
    
    Args:
        column: Pandas Series to normalize
        semantic_type: Detected semantic type
        patterns: Dictionary of pattern instances
        numeric_format: Preferred numeric format ('int' or 'float')
    
    Returns:
        Normalized Pandas Series
    """
    if semantic_type in patterns:
        try:
            pattern_instance = patterns[semantic_type]
            normalized = pattern_instance.normalize(column)
            
            # Apply numeric format preference for financial patterns
            # NOTE: Removed 'salary' from this list because salary pattern already returns integers
            if numeric_format == 'int' and semantic_type in ['currency', 'price', 'cost', 'income', 
                                                               'revenue', 'expense', 'tax', 'budget', 'profit', 
                                                               'loss', 'commission']:
                # Convert to int format (remove decimals if all zeros)
                def to_int_if_whole(val):
                    if pd.isna(val):
                        return val
                    try:
                        val_str = str(val)
                        # Extract number
                        num_match = re.search(r'([\d,]+)\.(\d+)', val_str)
                        if num_match:
                            whole = num_match.group(1)
                            decimal = num_match.group(2)
                            # If decimal is all zeros, remove it
                            if int(decimal) == 0:
                                # Reconstruct with currency symbol
                                prefix = val_str[:val_str.index(whole)]
                                suffix = val_str[val_str.index(decimal) + len(decimal):]
                                return f"{prefix}{whole}{suffix}"
                        return val
                    except:
                        return val
                
                normalized = normalized.apply(to_int_if_whole)
            
            return normalized
        except Exception as e:
            print(f"        Error normalizing {semantic_type}: {e}")
            return column
    
    # For unknown semantic types, return as-is
    return column


def rename_column_with_semantic(column_name: str, semantic_type: str) -> str:
    """
    Rename column to include semantic type suffix.
    
    Args:
        column_name: Original column name
        semantic_type: Detected semantic type
    
    Returns:
        New column name with semantic suffix
    """
    # Skip if semantic type is unknown or ambiguous
    if semantic_type in ['unknown', 'ambiguous']:
        return column_name
    
    # Add semantic suffix
    return f"{column_name}_({semantic_type.upper()})"


# ==============================
# MAIN PROCESSING
# ==============================

def process_dataset(df: pd.DataFrame, filename: str, userid: str,
                   sessionid: str, patterns: Dict) -> pd.DataFrame:
    """
    Process a single dataset: detect semantics, normalize columns, and update metadata.
    
    Args:
        df: Pandas DataFrame to process
        filename: Dataset filename
        userid: User ID
        sessionid: Session ID
        patterns: Dictionary of pattern instances
    
    Returns:
        Normalized DataFrame
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {filename}")
    print(f"{'-' * 70}")
    
    # Load existing profiling metadata
    print("   Loading profiling metadata...")
    metadata = load_profiling_metadata(userid, sessionid, filename)
    
    if metadata is None:
        print("   No profiling metadata found. Run profiling.py first.")
        return df
    
    # Remove duplicate column names (keep first occurrence)
    duplicate_name_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_name_cols:
        print(f"   [DUPLICATE NAMES] Found {len(duplicate_name_cols)} duplicate column name(s): {duplicate_name_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
        print(f"   [DUPLICATE NAMES] Removed duplicates, keeping first occurrence")
    
    # Remove duplicate column data (columns with identical values)
    print("   Checking for duplicate column data...")
    cols_to_keep = []
    cols_to_remove = []
    seen_data = {}
    
    for col in df.columns:
        # Create a hashable representation of the column data
        col_hash = tuple(df[col].fillna('__NA__').astype(str).values)
        
        if col_hash in seen_data:
            # This column has duplicate data
            cols_to_remove.append(col)
            print(f"   [DUPLICATE DATA] Column '{col}' has same data as '{seen_data[col_hash]}'")
        else:
            # First occurrence of this data
            seen_data[col_hash] = col
            cols_to_keep.append(col)
    
    if cols_to_remove:
        print(f"   [DUPLICATE DATA] Removing {len(cols_to_remove)} column(s) with duplicate data")
        df = df[cols_to_keep].copy()  # Make a proper copy to avoid SettingWithCopyWarning
        
        # Update profiling metadata to remove duplicate columns
        if 'column_wise_summary' in metadata:
            original_count = len(metadata['column_wise_summary'])
            metadata['column_wise_summary'] = [
                col_meta for col_meta in metadata['column_wise_summary']
                if col_meta['column_name'] not in cols_to_remove
            ]
            updated_count = len(metadata['column_wise_summary'])
            print(f"   [METADATA] Updated profiling: {original_count} -> {updated_count} columns")
        
        # Update column count in metadata
        metadata['number_of_columns'] = len(df.columns)
        
        # Save updated profiling metadata
        save_profiling_metadata(metadata, userid, sessionid, filename)
        print(f"   [METADATA] Saved updated profiling.json")
    else:
        print(f"   [DUPLICATE DATA] No duplicate data columns found")
    
    # PREPROCESSING STEP 1: Clean mixed-type columns (string/numeric mix)
    print("   Checking for mixed-type columns...")
    df, unit_detections = clean_mixed_type_columns(df)
    
    # PREPROCESSING STEP 2: Remove special symbols from all columns BEFORE semantic detection
    print("   Removing special symbols from data...")
    df = remove_special_symbols(df)
    
    # Duplicate removal disabled - preserving all rows
    original_rows = len(df)
    # df = df.drop_duplicates()  # DISABLED: Keep all rows including duplicates
    duplicates_removed = 0  # Set to 0 since we're not removing duplicates
    # if duplicates_removed > 0:
    #     print(f"    Removed {duplicates_removed} duplicate rows")
    
    # IMPORTANT: Detect patterns on ORIGINAL data BEFORE cleaning (to preserve symbols like %, $, °)
    print(f"   Detecting semantics for {len(df.columns)} columns...")
    
    # PRE-CHECK: Identify columns with >= 70% missing values BEFORE normalization
    # This prevents normalization from filling/converting NaN values
    columns_with_high_missing = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        total_rows = len(df)
        missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        if missing_percentage >= 70:
            columns_with_high_missing.append(col)
            print(f"     [PRE-CHECK] '{col}' has {missing_percentage:.1f}% missing - will be dropped")
    
    # Store original data for pattern detection
    df_original = df.copy()
    
    # Now clean the data for normalization
    import re
    # Remove all non-ASCII and all punctuation except . , : ; - /
    def clean_value(val):
        if pd.isna(val):
            return val
        s = str(val)
        # Replace en-dash with regular hyphen BEFORE removing non-ASCII
        s = s.replace('–', '-')
        # Remove all non-ASCII
        s = re.sub(r'[^\x20-\x7E]', '', s)
        # Remove all punctuation except . , : ; - / @ % $ (keep symbols for pattern detection)        # Remove extra spaces
        s = ' '.join(s.split())
        return s.strip()
    
    for col in df.columns:
        df[col] = df[col].apply(clean_value)

    # PATTERN EXTRACTION: DISABLED
    # Text patterns like "100 sold", "200 sold" will remain as varchar columns
    # No numeric extraction or column renaming will be performed
    
    # Process each column (using ORIGINAL data for detection, CLEANED data for normalization)

    transformed_columns = []
    skipped_columns = []
    dropped_columns = []  # Track dropped columns separately
    column_rename_map = {}
    new_currency_columns = []  # Track currency columns to add after processing

    for idx, column_summary in enumerate(metadata.get('column_wise_summary', [])):
        column_name = column_summary['column_name']

        # Skip currency indicator columns (they're metadata, not data to process)
        if column_name.endswith('_currency') or column_name.endswith('_(type)') or column_name.endswith('_(country)'):
            continue

        if column_name not in df.columns:
            print(f"      Column '{column_name}' not found in dataset")
            continue

        # Detect semantic type on ORIGINAL data (before cleaning)
        semantic_type, confidence, ambiguity_note = detect_semantic(
            df_original[column_name], patterns
        )
        
        # Override with unit detection from mixed-type cleaning
        if column_name in unit_detections:
            unit_info = unit_detections[column_name]
            semantic_type = unit_info['semantic_type']
            confidence = 1.0
            ambiguity_note = (f"Unit extracted from mixed-type values: "
                              f"{unit_info['unit']} ({unit_info['matched_count']}/"
                              f"{unit_info['total_strings']} strings matched)")
            print(f"       [UNIT OVERRIDE] {column_name}: {semantic_type} ({unit_info['unit']})")
        
        # If unknown, use inferred_dtype as semantic_type
        if semantic_type == 'unknown':
            semantic_type = column_summary.get('inferred_dtype', 'unknown')
        
        # FORCE specific semantic types for columns with clear keywords (even if low confidence)
        # This ensures columns like "budget", "discount", etc. are normalized properly
        column_name_lower = column_name.lower()
        if confidence < 0.6:  # Only for low-confidence columns
            # Currency-related columns
            if any(keyword in column_name_lower for keyword in ['income', 'salary', 'revenue', 'expense', 'price', 'amount', 'amt']) and 'currency' in patterns:
                semantic_type = 'currency'
                confidence = 0.6
                print(f"       [FORCED] Currency keyword detected - forcing currency normalization")
            elif 'budget' in column_name_lower and 'currency' in patterns:
                semantic_type = 'currency'
                confidence = 0.6
                print(f"       [FORCED] Budget keyword detected - forcing currency normalization")
            elif 'discount' in column_name_lower and 'percentage' in patterns:
                semantic_type = 'percentage'
                confidence = 0.6
                print(f"       [FORCED] Discount keyword detected - forcing percentage normalization")
            elif 'fiscal' in column_name_lower and 'year' in column_name_lower and 'fiscal_year' in patterns:
                semantic_type = 'fiscal_year'
                confidence = 0.6
                print(f"       [FORCED] Fiscal year keyword detected - forcing fiscal_year normalization")
            # Numeric columns with low confidence - use integer or float patterns
            elif semantic_type in ['integer', 'float'] and confidence == 0.0:
                # Keep the detected type but boost confidence
                confidence = 0.6
                print(f"       [FORCED] Numeric type detected - forcing {semantic_type} normalization")
        
        # Update metadata
        column_summary['semantic_type'] = semantic_type
        column_summary['semantic_confidence'] = round(confidence, 4)
        if ambiguity_note:
            column_summary['semantic_notes'] = ambiguity_note
        
        # Log detection
        print(f"     {column_name}: {semantic_type} (confidence: {confidence:.2%})")
        
        # Normalize ONLY if:
        # 1. Semantic type matched a pattern (in patterns dict)
        # 2. NOT mixed (multiple conflicting patterns)
        # 3. NOT basic type (int/float/varchar/text from fallback)
        basic_types = ['integer', 'float', 'varchar', 'text']
        is_basic_fallback = (semantic_type in basic_types and confidence == 0.0)
        
        should_normalize = (
            semantic_type not in ['unknown', 'mixed']
            and semantic_type in patterns
            and not is_basic_fallback
        )
        
        if should_normalize:
            pattern_instance = patterns.get(semantic_type)
            
            # If the column's unit was already detected and its numeric values
            # extracted during mixed-type cleaning, skip re-normalization —
            # the values are already clean numbers.
            if column_name in unit_detections:
                print(f"         [SKIP NORMALIZE] {column_name} — values already extracted by unit detection")
                # Convert to numeric dtype
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                # Preview
                try:
                    preview_vals = df[column_name].dropna().unique()[:5]
                    preview_str = ', '.join(str(v) for v in preview_vals)
                    print(f"         [VALUES] [{preview_str}]")
                except Exception:
                    pass
            else:
                print(f"         [NORMALIZING] {column_name} as {semantic_type}")
                # Detect numeric format preference (int vs float) BEFORE normalization
                numeric_format = detect_numeric_format(df[column_name])
                
                # Normalize column using pattern class
                pattern_instance = patterns[semantic_type]
                # IMPORTANT: Use ORIGINAL data for normalization (preserves symbols like $, €, ₹, °, etc.)
                df[column_name] = normalize_column(df_original[column_name], semantic_type, patterns, numeric_format)

                # ── POST-NORMALIZATION CLEANUP ──
                # After normalization, some values the pattern could NOT
                # handle remain as their original strings (e.g. "Varies
                # with device" after number_systems normalises "19M"→19000000).
                # Detect these leftover strings and blank them out so the
                # column becomes cleanly numeric.
                # GUARD: only trigger for semantic types that are supposed to
                # produce numeric output.  Types like version, date, text,
                # varchar legitimately stay as strings — never wipe those.
                _NUMERIC_SEMANTIC_TYPES = frozenset({
                    'integer', 'float', 'number_systems', 'currency',
                    'percentage', 'distance', 'weight', 'volume',
                    'temperature', 'speed', 'energy', 'power', 'pressure',
                    'capacity', 'density', 'area', 'angle', 'salary',
                    'ratio', 'duration',
                })
                if df[column_name].dtype == 'object' and semantic_type in _NUMERIC_SEMANTIC_TYPES:
                    # Count how many values are already numeric after normalize
                    non_null = df[column_name].dropna()
                    numeric_ok = 0
                    for v in non_null:
                        try:
                            float(str(v).replace(',', '').replace(' ', ''))
                            numeric_ok += 1
                        except (ValueError, TypeError):
                            pass
                    # Only clean if we have a genuine mix (some numeric, some not)
                    if numeric_ok > 0 and numeric_ok < len(non_null):
                        leftover_count = 0
                        def _post_norm_clean(val):
                            nonlocal leftover_count
                            if pd.isna(val) or str(val).strip() == '':
                                return val
                            try:
                                float(str(val).replace(',', '').replace(' ', ''))
                                return val          # already numeric
                            except (ValueError, TypeError):
                                leftover_count += 1
                                return ''           # blank leftover string
                        df[column_name] = df[column_name].apply(_post_norm_clean)
                        if leftover_count > 0:
                            print(f"         [POST-NORM CLEANUP] Blanked {leftover_count} "
                                  f"non-numeric values that pattern could not normalize")

            # Unicode-safe preview for Windows consoles (prevents UnicodeEncodeError)
            try:
                preview_vals = df[column_name].dropna().unique()[:5]
                preview_str = ', '.join(str(v) for v in preview_vals)
                preview_str = preview_str.encode('ascii', errors='backslashreplace').decode('ascii')
                print(f"         [NORMALIZED] Column values: [{preview_str}]")
            except Exception:
                print(f"         [NORMALIZED] Column values: [preview unavailable]")
            
            # CRITICAL: Enforce int/float consistency AFTER normalization
            # For INTEGER semantic type: convert to Int64 (nullable integer)
            # For FLOAT semantic type: convert to float with 2 decimal places
            # For other types: apply standard int/float detection
            if semantic_type == 'integer':
                # Force integer type
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')
            elif semantic_type == 'float':
                # Force float type with 2 decimal places
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce').round(2)
            elif pd.api.types.is_numeric_dtype(df[column_name]):
                has_decimal = False
                non_null = df[column_name].dropna()
                
                # Check both the values AND the column dtype
                # If column is already float dtype, keep as float
                if pd.api.types.is_float_dtype(df[column_name].dtype):
                    has_decimal = True
                else:
                    # Check individual values
                    for val in non_null:
                        try:
                            float_val = float(val)
                            if float_val != int(float_val):
                                has_decimal = True
                                break
                        except:
                            continue
                
                if has_decimal:
                    # Convert ALL to float with 2 decimal places for consistency
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                    df[column_name] = df[column_name].round(2)
                else:
                    # Convert ALL to int (preserving NaN)
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')
            
            # Check if column should be renamed based on pattern metadata
            new_column_name = column_name
            
            # Salary: handle different scenarios
            if semantic_type == 'salary' and hasattr(pattern_instance, 'scenario'):
                scenario = pattern_instance.scenario
                
                if scenario == 'has_hourly_weekly':
                    # Scenario 0: Contains hourly/weekly rates
                    # Create 2 columns: salary_{currency}, salary_time_unit
                    currency = pattern_instance.target_currency
                    new_column_name = f"{column_name}_{currency}"
                    
                    # Create time unit column
                    time_unit_col_name = f"{column_name}_time_unit"
                    df[time_unit_col_name] = pattern_instance.salary_time_units
                    df_original[time_unit_col_name] = pattern_instance.salary_time_units  # Also add to df_original
                    print(f"       [SALARY HOURLY/WEEKLY] Created {time_unit_col_name} column")
                    print(f"       [SALARY HOURLY/WEEKLY] Renamed to {new_column_name}")
                    
                    # Store info to add to metadata AFTER loop completes
                    new_currency_columns.append({
                        'column_name': time_unit_col_name,
                        'data': pattern_instance.salary_time_units
                    })
                
                elif scenario == 'mixed_both':
                    # Scenario 1: Mixed currency AND mixed time units
                    # Create 3 columns: salary, salary_time_unit, salary_currency
                    new_column_name = column_name  # Keep original name for amount
                    
                    # Create time unit column
                    time_unit_col_name = f"{column_name}_time_unit"
                    df[time_unit_col_name] = pattern_instance.salary_time_units
                    df_original[time_unit_col_name] = pattern_instance.salary_time_units  # Also add to df_original
                    print(f"       [SALARY SCENARIO 1] Created {time_unit_col_name} column")
                    
                    # Create currency column
                    currency_col_name = f"{column_name}_currency"
                    df[currency_col_name] = pattern_instance.salary_currencies
                    df_original[currency_col_name] = pattern_instance.salary_currencies  # Also add to df_original
                    print(f"       [SALARY SCENARIO 1] Created {currency_col_name} column")
                    
                    # Store info to add to metadata AFTER loop completes
                    new_currency_columns.append({
                        'column_name': time_unit_col_name,
                        'data': pattern_instance.salary_time_units
                    })
                    new_currency_columns.append({
                        'column_name': currency_col_name,
                        'data': pattern_instance.salary_currencies
                    })
                    
                elif scenario == 'mixed_currency':
                    # Scenario 2: Mixed currencies but same time unit
                    # Rename: salary_{most_common_currency}_per_{time_unit}
                    currency = pattern_instance.target_currency
                    period = pattern_instance.target_period
                    new_column_name = f"{column_name}_{currency}_per_{period}"
                    print(f"       [SALARY SCENARIO 2] Renamed to {new_column_name}")
                    
                elif scenario == 'mixed_time':
                    # Scenario 3: Same currency but mixed time units
                    # Rename: salary_{currency}_per_{most_common_time_unit}
                    currency = pattern_instance.target_currency
                    period = pattern_instance.target_period
                    new_column_name = f"{column_name}_{currency}_per_{period}"
                    print(f"       [SALARY SCENARIO 3] Renamed to {new_column_name}")
                    
                else:
                    # Uniform: single currency and time unit
                    currency = pattern_instance.target_currency
                    period = pattern_instance.target_period
                    new_column_name = f"{column_name}_{currency}_per_{period}"
                    print(f"       [SALARY UNIFORM] Renamed to {new_column_name}")
            
            # Currency: Rename column to include currency code (like salary pattern)
            if hasattr(pattern_instance, 'currency_codes') and pattern_instance.currency_codes is not None:
                # MIXED CURRENCIES: Create extra column for currency codes
                currency_col_name = f"{column_name}_(currency_type)"
                df[currency_col_name] = pattern_instance.currency_codes
                df_original[currency_col_name] = pattern_instance.currency_codes  # Also add to df_original
                print(f"       [CURRENCY TYPE] Created {currency_col_name} column (mixed currencies)")
                
                # Store info to add to metadata AFTER loop completes
                new_currency_columns.append({
                    'column_name': currency_col_name,
                    'data': pattern_instance.currency_codes
                })
                
            elif hasattr(pattern_instance, 'detected_currency') and pattern_instance.detected_currency:
                # SINGLE CURRENCY: Rename column to include currency code in name
                new_column_name = f"{column_name}_({pattern_instance.detected_currency})"
                print(f"       [CURRENCY TYPE] Renamed column to {new_column_name}")
            
            # Measurement units: rename column with unit suffix (distance, weight, volume, etc.)
            if hasattr(pattern_instance, 'detected_unit') and pattern_instance.detected_unit:
                # MEASUREMENT UNIT: Rename column with unit suffix
                new_column_name = f"{column_name}_({pattern_instance.detected_unit})"
            
            # Boolean format: rename column with format suffix (yes/no, active/inactive, etc.)
            elif hasattr(pattern_instance, 'detected_format') and pattern_instance.detected_format:
                # Check if it's a multi-country phone pattern
                if hasattr(pattern_instance, 'is_multi_country') and pattern_instance.is_multi_country:
                    # Phone with multiple countries: add country column
                    new_column_name = column_name  # Keep original name for phone
                    country_column_name = f"{column_name}_(country)"
                    
                    # Add country column
                    if hasattr(pattern_instance, 'country_data') and pattern_instance.country_data is not None:
                        df[country_column_name] = pattern_instance.country_data
                        print(f"         [ADDED] Country column: {country_column_name}")
                else:
                    # Skip renaming for specific semantic types that don't need format suffix
                    no_rename_types = ['email', 'url', 'gender', 'file_path', 'files', 'geo_coordinate','network_addresses']
                    if semantic_type not in no_rename_types:
                        # Regular format suffix (boolean, single-country phone, etc.)
                        new_column_name = f"{column_name}_({pattern_instance.detected_format})"
            
            # Percentage/Discount/Interest: add _percentage suffix
            elif hasattr(pattern_instance, 'detected_suffix') and pattern_instance.detected_suffix:
                new_column_name = f"{column_name}{pattern_instance.detected_suffix}"
            
            # Geo Coordinate: Split into latitude and longitude columns
            if semantic_type == 'geo_coordinate' and hasattr(pattern_instance, 'latitude_values') and hasattr(pattern_instance, 'longitude_values'):
                if pattern_instance.latitude_values is not None and pattern_instance.longitude_values is not None:
                    # Create latitude and longitude columns
                    lat_col_name = f"{column_name}_latitude"
                    lon_col_name = f"{column_name}_longitude"
                    
                    df[lat_col_name] = pattern_instance.latitude_values
                    df[lon_col_name] = pattern_instance.longitude_values
                    df_original[lat_col_name] = pattern_instance.latitude_values
                    df_original[lon_col_name] = pattern_instance.longitude_values
                    
                    print(f"       [GEO COORDINATE] Split into {lat_col_name} and {lon_col_name}")
                    
                    # Store info to add to metadata AFTER loop completes
                    new_currency_columns.append({
                        'column_name': lat_col_name,
                        'data': pattern_instance.latitude_values
                    })
                    new_currency_columns.append({
                        'column_name': lon_col_name,
                        'data': pattern_instance.longitude_values
                    })
                    
                    # Remove the original geo_coordinate column from DataFrame
                    df.drop(columns=[column_name], inplace=True)
                    df_original.drop(columns=[column_name], inplace=True)
                    
                    # Mark column for removal from metadata
                    column_summary['_to_remove'] = True
                    
                    # Add to transformed list and skip the rest of processing
                    transformed_columns.append(lat_col_name)
                    transformed_columns.append(lon_col_name)
                    continue  # Skip the rest of the loop for this column
            
            # Unit-based renaming from mixed-type extraction
            # (only if pattern-based renaming did not already rename the column)
            if column_name in unit_detections and new_column_name == column_name:
                new_column_name = f"{column_name}_({unit_detections[column_name]['unit']})"
                print(f"       [UNIT RENAME] {column_name} -> {new_column_name}")
            
            # Rename column if needed
            if new_column_name != column_name:
                df.rename(columns={column_name: new_column_name}, inplace=True)
                column_summary['column_name'] = new_column_name
                column_name = new_column_name
            
            # Re-infer dtype after normalization (only if column still exists)
            if column_name in df.columns:
                normalized_dtype = infer_normalized_dtype(df[column_name])
                column_summary['inferred_dtype'] = normalized_dtype
                
                # Update sample_values to reflect normalized format
                normalized_samples = get_sample_values(df[column_name], n=5)
                column_summary['sample_values'] = normalized_samples
            
            transformed_columns.append(column_name)
        else:
            # SKIPPED column - still update sample_values to ensure accuracy
            skipped_columns.append(column_name)
            
            # Update sample_values from current dataframe (fix for corrupted metadata)
            if column_name in df.columns:
                refreshed_samples = get_sample_values(df[column_name], n=5)
                column_summary['sample_values'] = refreshed_samples
            
            if ambiguity_note:
                print(f"         {ambiguity_note}")
    
    # Add new currency columns to metadata AFTER processing is done
    for curr_col_info in new_currency_columns:
        currency_col_name = curr_col_info['column_name']
        
        # Detect if it's a numeric column (hourly salary, lat/lon) or text column (currency type)
        is_numeric = pd.api.types.is_numeric_dtype(df[currency_col_name].dtype)
        inferred_dtype = 'float' if is_numeric and pd.api.types.is_float_dtype(df[currency_col_name].dtype) else ('integer' if is_numeric else 'text')
        
        metadata['column_wise_summary'].append({
            'column_name': currency_col_name,
            'inferred_dtype': inferred_dtype,
            'null_count': int(df[currency_col_name].isna().sum()),
            'null_percentage': round((df[currency_col_name].isna().sum() / len(df) * 100), 2),
            'unique_count': int(df[currency_col_name].nunique(dropna=True)),
            'sample_values': df[currency_col_name].dropna().head(5).tolist() if len(df[currency_col_name].dropna()) > 0 else [],
            'semantic_type': inferred_dtype,
            'semantic_confidence': 1.0
        })
    
    # Remove columns marked for removal (geo_coordinate split columns)
    metadata['column_wise_summary'] = [
        col_meta for col_meta in metadata['column_wise_summary']
        if not col_meta.get('_to_remove', False)
    ]
    
    # FINAL STEP: Drop columns identified in pre-check (>= 70% missing values)
    print(f"\n   Final check: Removing columns with >= 70% missing values...")
    
    if columns_with_high_missing:
        # Only drop columns that still exist (some may have been dropped during semantic detection)
        existing_cols_to_drop = [col for col in columns_with_high_missing if col in df.columns]
        if existing_cols_to_drop:
            # Drop columns from DataFrame
            df.drop(columns=existing_cols_to_drop, inplace=True)
        
            # Remove from metadata
            metadata['column_wise_summary'] = [
                col_meta for col_meta in metadata['column_wise_summary']
                if col_meta['column_name'] not in existing_cols_to_drop
            ]
        
            # Update column count
            metadata['number_of_columns'] = len(df.columns)
            print(f"      Dropped columns: {existing_cols_to_drop}")
        else:
            print(f"      No columns to drop (already removed during semantic detection)")
        
        print(f"      Remaining columns: {list(df.columns)}")
        
        # Add to dropped list
        for col in columns_with_high_missing:
            # Get missing percentage from original data
            missing_pct = (df_original[col].isna().sum() / len(df_original)) * 100 if col in df_original.columns else 100.0
            dropped_columns.append(f"{col} (dropped: {missing_pct:.1f}% missing)")
            print(f"     [DROPPED] '{col}' has {missing_pct:.1f}% missing values")
        
        print(f"     Total dropped: {len(columns_with_high_missing)} column(s)")
    else:
        print(f"     No columns with >= 70% missing values")
    
    # Update metadata with transformation summary
    metadata['transformation_summary'] = {
        'duplicates_removed': duplicates_removed,
        'columns_transformed': len(transformed_columns),
        'columns_skipped': len(skipped_columns),
        'columns_dropped': len(dropped_columns),
        'transformed_column_list': transformed_columns,
        'skipped_column_list': skipped_columns,
        'dropped_column_list': dropped_columns,
        'transformation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save updated metadata
    print(f"\n   Updating profiling metadata...")
    save_profiling_metadata(metadata, userid, sessionid, filename)
    print(f"\n   Transformation complete:")
    print(f"     Transformed: {len(transformed_columns)} columns")
    print(f"     Skipped: {len(skipped_columns)} columns")
    print(f"     Dropped: {len(dropped_columns)} columns")
    
    return df


def load_dataset_from_storage(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from Supabase storage.

    Args:
        filepath: Supabase storage path, e.g. "input/user1/session1/data.csv"

    Returns:
        Pandas DataFrame (cleaned of corrupted rows)
    """
    file_content = download_file(filepath)

    if filepath.lower().endswith('.csv'):
        try:
            try:
                df = pd.read_csv(BytesIO(file_content), encoding='utf-8', on_bad_lines='warn')
            except TypeError:
                df = pd.read_csv(BytesIO(file_content), encoding='utf-8', error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            try:
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding='latin-1', on_bad_lines='warn')
                except TypeError:
                    df = pd.read_csv(BytesIO(file_content), encoding='latin-1', error_bad_lines=False, warn_bad_lines=True)
            except Exception:
                try:
                    df = pd.read_csv(BytesIO(file_content), encoding='cp1252', on_bad_lines='warn')
                except TypeError:
                    df = pd.read_csv(BytesIO(file_content), encoding='cp1252', error_bad_lines=False, warn_bad_lines=True)

        print(f"   Running data integrity checks...")
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            print(f"       Detected {len(unnamed_cols)} unnamed columns (potential column bleed)")
            for col in unnamed_cols:
                non_null_pct = df[col].notna().sum() / len(df) * 100
                print(f"        - {col}: {non_null_pct:.1f}% filled")
                if non_null_pct < 10:
                    df = df.drop(columns=[col])
                    print(f"         Removed {col} (mostly empty)")

        original_count = len(df)
        df = df.dropna(how='all')
        dropped_empty = original_count - len(df)
        if dropped_empty > 0:
            print(f"      Removed {dropped_empty} completely empty rows")

        print(f"       Expected columns: {len(df.columns)}")
        print(f"       Total rows after cleanup: {len(df)}")

    elif filepath.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(BytesIO(file_content))
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    return df


def save_dataset_to_storage(df: pd.DataFrame, userid: str, sessionid: str, filename: str):
    """
    Save cleaned dataset to Supabase storage output directory.
    """
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_cleaned.csv"
    path = f"output/{userid}/{sessionid}/{output_filename}"
    content = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
    upload_file(path, content, content_type="text/csv")
    print(f"   Saved cleaned dataset to: {path}")


def process_user_datasets(userid: str, sessionid: str):
    """
    Main processing function: process all datasets for a user/session.
    """
    print("=" * 70)
    print("COLUMN HANDLER - AutoML Preprocessing System")
    print("=" * 70)
    print(f"User ID:    {userid}")
    print(f"Session ID: {sessionid}")
    print()

    try:
        print(" Loading semantic pattern classes...")
        patterns = load_patterns()
        print()

        if not patterns:
            print(" No patterns found. Cannot proceed.")
            return

        folder = f"input/{userid}/{sessionid}"
        print(f" Searching for datasets in: {folder}")

        try:
            files = list_files(folder)
        except Exception as e:
            print(f"  Error listing files: {e}")
            return

        csv_files = [f for f in files if f.lower().endswith('.csv')]

        if not csv_files:
            print(f"  No CSV files found in: {folder}")
            return

        print(f" Found {len(csv_files)} CSV dataset(s)")

        success_count = 0
        error_count = 0

        for filename in csv_files:
            try:
                file_path = f"input/{userid}/{sessionid}/{filename}"
                print(f"\n   Loading dataset: {filename}")
                df = load_dataset_from_storage(file_path)
                print(f"     Shape: {df.shape[0]} rows  {df.shape[1]} columns")

                cleaned_df = process_dataset(df, filename, userid, sessionid, patterns)
                save_dataset_to_storage(cleaned_df, userid, sessionid, filename)
                success_count += 1

            except Exception as e:
                error_count += 1
                print(f"\n   Error processing {filename}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 70)
        print(" PROCESSING SUMMARY")
        print("=" * 70)
        print(f" Successfully processed: {success_count} dataset(s)")
        if error_count > 0:
            print(f" Errors encountered: {error_count} dataset(s)")
        print()
        print(" Column handling completed!")
        print("=" * 70)

    except Exception as e:
        print(f" Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main entry point for the column handler script.
    """
    print()

    if len(sys.argv) > 2:
        userid    = sys.argv[1]
        sessionid = sys.argv[2]
    elif len(sys.argv) > 1:
        userid    = sys.argv[1]
        sessionid = input("Enter Session ID: ").strip()
    else:
        userid    = input("Enter User ID:    ").strip()
        sessionid = input("Enter Session ID: ").strip()

    if not userid:
        print(" Error: User ID cannot be empty")
        sys.exit(1)
    if not sessionid:
        print(" Error: Session ID cannot be empty")
        sys.exit(1)

    print()
    process_user_datasets(userid, sessionid)


if __name__ == "__main__":
    main()
