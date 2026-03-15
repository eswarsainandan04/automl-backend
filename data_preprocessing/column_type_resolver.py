"""
Column Type Resolver Module for AutoML Preprocessing System

This module resolves varchar/object columns by:
1. Identifying ID-like text → SKIP
2. Identifying long unstructured text → SKIP
3. Extracting numeric + short text → EXTRACT structured components
4. Handling range patterns
5. Re-detecting and normalizing new columns

Author: AutoML Preprocessing System
Date: 2026-01-22
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from io import BytesIO
try:
    from .supabase_storage import download_file, upload_file, download_json, upload_json, list_files
except ImportError:
    from supabase_storage import download_file, upload_file, download_json, upload_json, list_files

# Import pattern classes
try:
    from .patterns import PATTERNS
except ImportError:
    from patterns import PATTERNS

# Import quantulum3 for unit extraction
try:
    from quantulum3 import parser as quantulum_parser
except ImportError:
    print("WARNING: quantulum3 not installed. Install with: pip install quantulum3")
    quantulum_parser = None


# ==============================
# HELPER FUNCTIONS
# ==============================

def is_id_like(column: pd.Series) -> bool:
    """
    Detect if column contains ID-like values (alphanumeric codes).
    
    Characteristics:
    - Alphanumeric without spaces (or with underscores/hyphens only)  
    - High pattern match (>70% values match ID pattern)
    - OR column name contains 'id', 'code', 'reference', 'ref'
    
    Examples:
    - C00021, CUST-001, REF123
    - 22HP1A1201, HCL-12312
    - HCLID0032, USER_99102
    - ORD-A92XZ, EMP-1023
    
    Args:
        column: Pandas Series to analyze
    
    Returns:
        True if column is ID-like, False otherwise
    """
    non_null = column.dropna()
    if len(non_null) == 0:
        return False
    
    # Check column name for ID-related keywords
    col_name_lower = column.name.lower() if hasattr(column, 'name') and column.name else ''
    id_keywords = ['_id', 'code', 'reference', '_ref', 'customer_id', 'user_id', 'employee_id']
    has_id_keyword = any(keyword in col_name_lower for keyword in id_keywords)
    
    # Sample values
    sample = non_null.head(min(100, len(non_null)))
    
    # Check pattern: alphanumeric with optional separators (-, _)
    id_pattern = r'^[A-Za-z0-9][A-Za-z0-9_\-]*$'
    
    matched_count = 0
    for val in sample:
        val_str = str(val).strip()
        
        # Check if matches ID pattern
        if re.match(id_pattern, val_str):
            # Additional check: must have BOTH letters and numbers
            has_letters = bool(re.search(r'[A-Za-z]', val_str))
            has_numbers = bool(re.search(r'[0-9]', val_str))
            
            if has_letters and has_numbers:
                matched_count += 1
    
    # Calculate pattern match percentage
    pattern_match_pct = matched_count / len(sample) if len(sample) > 0 else 0
    
    # ID-like if:
    # 1. >70% values match ID pattern, OR
    # 2. Column name contains ID keyword AND >50% values match pattern
    if pattern_match_pct > 0.7:
        return True
    elif has_id_keyword and pattern_match_pct > 0.5:
        return True
    else:
        return False


def is_long_text(column: pd.Series, avg_length_threshold: int = 50, 
                 avg_words_threshold: int = 5) -> bool:
    """
    Detect if column contains long, unstructured text.
    
    Characteristics:
    - Sentence-like structure
    - High average length (>50 chars)
    - Multiple words (>5 words on average)
    - High uniqueness
    
    Examples:
    - "There are 500 men in a city since 1995 year..."
    - "According to reports, sales increased significantly"
    - "Handled manually by the admin due to issues"
    
    Args:
        column: Pandas Series to analyze
        avg_length_threshold: Minimum average character length
        avg_words_threshold: Minimum average word count
    
    Returns:
        True if column contains long text, False otherwise
    """
    non_null = column.dropna()
    if len(non_null) == 0:
        return False
    
    # Calculate average length
    avg_length = non_null.astype(str).str.len().mean()
    
    # Calculate average word count
    avg_words = non_null.astype(str).str.split().str.len().mean()
    
    # Check if meets thresholds
    if avg_length > avg_length_threshold and avg_words > avg_words_threshold:
        return True
    
    return False


def is_extractable_numeric_text(column: pd.Series) -> bool:
    """
    Detect if column contains extractable numeric + short text.
    
    Characteristics:
    - Contains numeric values
    - Combined with 1-2 words of context
    - Not ID-like
    - Not long text
    
    Examples:
    - "12 Days"
    - "12K+ products sold"
    - "100 users"
    - "10$-20$ per year"
    - "20% discount"
    
    Args:
        column: Pandas Series to analyze
    
    Returns:
        True if extractable, False otherwise
    """
    non_null = column.dropna()
    if len(non_null) == 0:
        return False
    
    # Sample values
    sample = non_null.head(min(100, len(non_null)))
    
    # Pattern: must contain at least one number
    has_numeric_count = 0
    
    for val in sample:
        val_str = str(val).strip()
        
        # Check if contains any numeric characters
        if re.search(r'\d', val_str):
            has_numeric_count += 1
    
    # If >70% of values contain numbers, it's extractable
    return (has_numeric_count / len(sample)) > 0.7


# ==============================
# EXTRACTION FUNCTIONS
# ==============================

def process_numeric_value(value: str) -> float:
    """
    Process numeric values with suffixes and modifiers.
    
    Examples:
    - "10+" → 10
    - "1.5k+" → 1500
    - "5K" → 5000
    - "2.5M" → 2500000
    - "100" → 100
    
    Args:
        value: String value to process
    
    Returns:
        Numeric value as float
    """
    val = value.strip()
    
    # Remove + modifier if present
    if val.endswith('+'):
        val = val[:-1]
    
    # Check for k/K suffix (thousands)
    if val.lower().endswith('k'):
        try:
            num = float(val[:-1])
            return num * 1000
        except ValueError:
            pass
    
    # Check for m/M suffix (millions)
    elif val.lower().endswith('m'):
        try:
            num = float(val[:-1])
            return num * 1000000
        except ValueError:
            pass
    
    # Check for b/B suffix (billions)
    elif val.lower().endswith('b'):
        try:
            num = float(val[:-1])
            return num * 1000000000
        except ValueError:
            pass
    
    # No suffix, just return the number
    try:
        return float(val)
    except ValueError:
        # If conversion fails, return the original string
        return val


def handle_range_affixes(min_part: str, max_part: str) -> Tuple[str, str, Optional[str]]:
    """
    Handle prefix/postfix symbols and units in range values.
    
    Examples:
    - "2", "5k" → "2k", "5k", None
    - "$2", "3" → "$2", "$3", None
    - "2", "6 rs" → "2 rs", "6 rs", None
    - "2$", "4$" with external text → "2$", "4$", None
    
    Args:
        min_part: Minimum value part of range
        max_part: Maximum value part of range
    
    Returns:
        Tuple of (min_final, max_final, extracted_text)
    """
    # Extract numeric and non-numeric parts from both min and max
    min_num_match = re.search(r'(\d+\.?\d*)', min_part)
    max_num_match = re.search(r'(\d+\.?\d*)', max_part)
    
    if not min_num_match or not max_num_match:
        # Can't parse numbers, return as-is
        return min_part, max_part, None
    
    min_num = min_num_match.group(1)
    max_num = max_num_match.group(1)
    
    # Get prefix and postfix for min
    min_prefix = min_part[:min_num_match.start()].strip()
    min_postfix = min_part[min_num_match.end():].strip()
    
    # Get prefix and postfix for max
    max_prefix = max_part[:max_num_match.start()].strip()
    max_postfix = max_part[max_num_match.end():].strip()
    
    # Determine common prefix/postfix to apply to both values
    common_prefix = ""
    common_postfix = ""
    extracted_text = None
    
    # Handle PREFIX (e.g., $2-3 → $2, $3 or rs 2-6 → rs 2, rs 6)
    if min_prefix and not max_prefix:
        # Min has prefix, max doesn't → apply to both
        common_prefix = min_prefix
    elif max_prefix and not min_prefix:
        # Max has prefix, min doesn't → apply to both
        common_prefix = max_prefix
    elif min_prefix == max_prefix:
        # Both have same prefix → use it
        common_prefix = min_prefix
    
    # Handle POSTFIX (e.g., 2-5k → 2k, 5k or 2-6 rs → 2 rs, 6 rs)
    if min_postfix and not max_postfix:
        # Min has postfix, max doesn't
        # Check if it's a short symbol/unit (≤4 chars, like k, $, rs, %, hr)
        if len(min_postfix) <= 4:
            common_postfix = min_postfix
        else:
            # Longer text, likely descriptive → extract separately
            extracted_text = min_postfix
    elif max_postfix and not min_postfix:
        # Max has postfix, min doesn't
        # Check if it's a short symbol/unit
        if len(max_postfix) <= 4:
            common_postfix = max_postfix
        else:
            # Longer text → extract separately
            extracted_text = max_postfix
    elif min_postfix == max_postfix:
        # Both have same postfix
        if len(min_postfix) <= 4:
            common_postfix = min_postfix
        else:
            extracted_text = min_postfix
    
    # Build final values with distributed affixes
    # Format: prefix + number + postfix (with appropriate spacing)
    if common_prefix and common_postfix:
        # Both prefix and postfix
        min_final = f"{common_prefix}{min_num}{common_postfix}"
        max_final = f"{common_prefix}{max_num}{common_postfix}"
    elif common_prefix:
        # Only prefix
        min_final = f"{common_prefix}{min_num}"
        max_final = f"{common_prefix}{max_num}"
    elif common_postfix:
        # Only postfix - check if we need space (for units like "rs", "kg")
        if common_postfix.isalpha() and len(common_postfix) > 1:
            # Alphabetic unit → add space
            min_final = f"{min_num} {common_postfix}"
            max_final = f"{max_num} {common_postfix}"
        else:
            # Symbol → no space
            min_final = f"{min_num}{common_postfix}"
            max_final = f"{max_num}{common_postfix}"
    else:
        # No common affixes
        min_final = min_num
        max_final = max_num
    
    return min_final, max_final, extracted_text


def extract_structured_components(value: str) -> Dict[str, Any]:
    """
    Extract structured components from a varchar value.
    
    Strategy: Split ranges into min/max values preserving original format,
    then let pattern detection & normalize() handle the conversion.
    
    Examples:
    - "2hr - 3hr" → min="2hr", max="3hr" (pattern will normalize to duration)
    - "2K - 5K" → min="2K", max="5K" (pattern will normalize to 2000, 5000)
    - "$2 - $3" → min="$2", max="$3" (pattern will normalize currency)
    - "3% - 4%" → min="3%", max="4%" (pattern will normalize percentage)
    - "2024-25" → min=2024, max=2025 (year range with expansion)
    
    Args:
        value: String value to extract from
    
    Returns:
        Dictionary with extracted components (preserving original format for pattern detection)
    """
    if pd.isna(value) or value == '':
        return {}
    
    val_str = str(value).strip()
    result = {}
    
    # PRIORITY 1: Check for year range pattern FIRST (2024-25, 2014-15, etc.)
    # This must come before generic range pattern to handle abbreviated years
    year_range_pattern = r'^\d{4}\s*-\s*\d{2,4}$'
    if re.match(year_range_pattern, val_str):
        min_year, max_year = parse_year_range(val_str)
        if min_year is not None and max_year is not None:
            result['value_min'] = str(min_year)  # Full year
            result['value_max'] = str(max_year)  # Full year expanded
            return result
    
    # Universal RANGE pattern: capture anything before and after separator
    # Supports: $2-$3, 2hr-3hr, 2K-5K, 3%-4%, 2-3, 2-5k, $2-3, 2-6 rs, etc.
    range_match = re.match(r'^(.+?)\s*[-–]\s*(.+?)(\s+(.+))?$', val_str)
    if range_match:
        min_part = range_match.group(1).strip()
        max_part = range_match.group(2).strip()
        text_part = range_match.group(4).strip() if range_match.group(4) else None
        
        # Only treat as range if both parts contain numbers
        if re.search(r'\d', min_part) and re.search(r'\d', max_part):
            # Apply smart prefix/postfix handling
            min_final, max_final, extracted_text = handle_range_affixes(min_part, max_part)
            
            result['value_min'] = min_final
            result['value_max'] = max_final
            
            # Combine text parts (extracted from affixes + external text)
            if text_part and extracted_text:
                result['text'] = f"{extracted_text} {text_part}"
            elif text_part:
                result['text'] = text_part
            elif extracted_text:
                result['text'] = extracted_text
            
            return result
    
    # "to" based ranges: 2 to 5, $2 to $3, 2 to 5k, etc.
    range_to_match = re.match(r'^(.+?)\s+(?:to|TO)\s+(.+?)(\s+(.+))?$', val_str, re.IGNORECASE)
    if range_to_match:
        min_part = range_to_match.group(1).strip()
        max_part = range_to_match.group(2).strip()
        text_part = range_to_match.group(4).strip() if range_to_match.group(4) else None
        
        if re.search(r'\d', min_part) and re.search(r'\d', max_part):
            # Apply smart prefix/postfix handling
            min_final, max_final, extracted_text = handle_range_affixes(min_part, max_part)
            
            result['value_min'] = min_final
            result['value_max'] = max_final
            
            # Combine text parts
            if text_part and extracted_text:
                result['text'] = f"{extracted_text} {text_part}"
            elif text_part:
                result['text'] = text_part
            elif extracted_text:
                result['text'] = extracted_text
            
            return result
    
    # Single value with text (100 orders, $50 price, 5K users, 3% discount, 5K+ customers)
    # Check for + modifier and handle it
    if re.search(r'\d', val_str):
        # Check if there's a text part after the number+unit
        parts = val_str.split(None, 1)  # Split on first whitespace
        if len(parts) == 2:
            value_part = parts[0]
            text_part = parts[1]
            
            # Process value_part: remove + and convert k/K/M/m suffixes
            processed_value = process_numeric_value(value_part)
            result['value'] = str(processed_value)
            
            result['text'] = text_part  # "orders", "price", "users", "discount", "customers"
        else:
            # No text part, just value
            value_part = val_str
            
            # Process value_part: remove + and convert k/K/M/m suffixes
            processed_value = process_numeric_value(value_part)
            result['value'] = str(processed_value)
        return result
    
    # No extraction possible
    return {}


# ==============================
# RANGE HANDLING
# ==============================

def parse_year_range(value: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse year range values, handling both full and abbreviated formats.
    
    Formats supported:
    - Full: "2024-2025" → (2024, 2025)
    - Abbreviated: "2024-25" → (2024, 2025)
    - Abbreviated: "2014-15" → (2014, 2015)
    - Abbreviated: "1999-00" → (1999, 2000)
    
    Args:
        value: String containing year range
    
    Returns:
        Tuple of (min_year, max_year) or (None, None) if invalid
    """
    if pd.isna(value) or value == '':
        return None, None
    
    val_str = str(value).strip()
    
    # Pattern: year-year (with optional spaces)
    match = re.match(r'^(\d{4})\s*-\s*(\d{2,4})$', val_str)
    if match:
        min_year_str = match.group(1)
        max_year_str = match.group(2)
        
        min_year = int(min_year_str)
        
        # If max year is abbreviated (2 digits), expand it
        if len(max_year_str) == 2:
            # Get century from min_year
            century = (min_year // 100) * 100
            max_year_short = int(max_year_str)
            
            # If short year < min_year last 2 digits, it's next century
            if max_year_short < (min_year % 100):
                max_year = century + 100 + max_year_short
            else:
                max_year = century + max_year_short
        else:
            max_year = int(max_year_str)
        
        return min_year, max_year
    
    return None, None


def detect_year_range_pattern(column: pd.Series) -> Tuple[bool, Optional[Dict]]:
    """
    Detect if a column contains year range patterns.
    
    Supported formats:
    - Full: 2024-2025
    - Abbreviated: 2024-25, 2014-15
    
    Args:
        column: Pandas Series to analyze
    
    Returns:
        Tuple of (is_year_range, range_data_dict)
        range_data_dict contains: {'min_values': [...], 'max_values': [...]}
    """
    non_null = column.dropna()
    if len(non_null) < 1:
        return False, None
    
    # Sample values
    sample = non_null.head(min(100, len(non_null)))
    
    # Year range pattern: YYYY-YY or YYYY-YYYY
    year_range_pattern = r'^\d{4}\s*-\s*\d{2,4}$'
    
    matched_count = 0
    for val in sample:
        val_str = str(val).strip()
        if re.match(year_range_pattern, val_str):
            matched_count += 1
    
    # Require at least 70% of values to match year range pattern
    if matched_count / len(sample) < 0.7:
        return False, None
    
    # Extract min/max years from ALL rows
    min_values = []
    max_values = []
    
    for val in column:
        min_year, max_year = parse_year_range(val)
        min_values.append(min_year)
        max_values.append(max_year)
    
    return True, {
        'min_values': min_values,
        'max_values': max_values
    }


def detect_range_pattern(column: pd.Series) -> Tuple[bool, Optional[Dict]]:
    """
    Detect if a column contains range patterns and extract min/max values.
    
    Supported formats:
    - Hyphen: 10-20, 2.5-3.5, 10 - 20
    - Word: 10 to 20, 5 TO 12.5
    - With units: 5$ - 10$, 20% - 30%, 5k - 10k
    - Ordinals: 10th to 12th
    - Letters: A - Z, Jan-Mar
    - Years: 2024-25, 2024-2025 (handled specially)
    
    Args:
        column: Pandas Series to analyze
    
    Returns:
        Tuple of (has_range_pattern, range_data_dict)
        range_data_dict contains: {'min_values': [...], 'max_values': [...]}
    """
    non_null = column.dropna()
    if len(non_null) < 2:  # Need at least 2 values to detect pattern
        return False, None
    
    # Sample values
    sample = non_null.head(min(100, len(non_null)))
    
    # PRIORITY 1: Check for year ranges FIRST (before temporal check)
    is_year_range, year_data = detect_year_range_pattern(column)
    if is_year_range:
        return True, year_data
    
    # PRIORITY 2: Check if column contains temporal formats (dates, times, datetimes, timestamps)
    temporal_patterns = [
        r'^\d{4}-\d{2}-\d{2}$',  # Date: 2023-01-15
        r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # Datetime: 2024-12-01 09:30:00
        r'^\d{2}:\d{2}:\d{2}',  # Time: 09:30:00
        r'^\d{10,13}$',  # Unix timestamp
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO datetime
    ]
    
    temporal_match_count = 0
    for val in sample:
        val_str = str(val).strip()
        for pattern in temporal_patterns:
            if re.match(pattern, val_str):
                temporal_match_count += 1
                break
    
    # If >50% of values match temporal patterns, do NOT treat as range
    if temporal_match_count / len(sample) > 0.5:
        return False, None
    
    # Range patterns (case-insensitive)
    range_patterns = [
        r'^\s*([^-]+?)\s*-\s*(.+?)\s*$',  # Hyphen-based ranges
        r'^\s*(.+?)\s+(?:to|TO)\s+(.+?)\s*$'  # "to" based ranges
    ]
    
    matched_count = 0
    for val in sample:
        val_str = str(val).strip()
        for pattern in range_patterns:
            if re.match(pattern, val_str):
                matched_count += 1
                break
    
    # Require at least 70% of values to match range pattern
    if matched_count / len(sample) < 0.7:
        return False, None
    
    # Extract min/max values from ALL rows
    min_values = []
    max_values = []
    
    for val in column:
        if pd.isna(val):
            min_values.append(np.nan)
            max_values.append(np.nan)
            continue
        
        val_str = str(val).strip()
        extracted = False
        
        for pattern in range_patterns:
            match = re.match(pattern, val_str)
            if match:
                min_part = match.group(1).strip()
                max_part = match.group(2).strip()
                
                min_values.append(min_part)
                max_values.append(max_part)
                extracted = True
                break
        
        if not extracted:
            min_values.append(np.nan)
            max_values.append(np.nan)
    
    return True, {
        'min_values': min_values,
        'max_values': max_values
    }


def ranges_handler(df: pd.DataFrame, metadata: Dict) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    Detect and split range patterns in columns into min/max columns.
    
    Args:
        df: DataFrame to process
        metadata: Profiling metadata dictionary
    
    Returns:
        Tuple of (updated_df, updated_metadata, list_of_split_columns)
    """
    print(f"   [RANGES] Detecting range patterns in columns...")
    
    split_columns_info = []
    
    for col in list(df.columns):
        # Check if column contains range patterns
        has_range, range_data = detect_range_pattern(df[col])
        
        if has_range and range_data:
            min_col_name = f"{col}_min"
            max_col_name = f"{col}_max"
            
            print(f"     [RANGE DETECTED] {col} -> {min_col_name}, {max_col_name}")
            
            # Create min and max columns
            df[min_col_name] = range_data['min_values']
            df[max_col_name] = range_data['max_values']
            
            # Store info for metadata update
            split_columns_info.append({
                'original': col,
                'min_col': min_col_name,
                'max_col': max_col_name,
                'min_values': range_data['min_values'],
                'max_values': range_data['max_values']
            })
            
            # Remove original column
            df.drop(columns=[col], inplace=True)
    
    # Update metadata for split columns
    if split_columns_info:
        for split_info in split_columns_info:
            # Find and remove original column from metadata
            col_summaries = metadata.get('column_wise_summary', [])
            
            for idx, summary in enumerate(col_summaries):
                if summary['column_name'] == split_info['original']:
                    col_summaries.pop(idx)
                    break
            
            # Add min/max columns to metadata
            min_series = pd.Series(split_info['min_values'])
            max_series = pd.Series(split_info['max_values'])
            
            # Add min column
            col_summaries.append({
                'column_name': split_info['min_col'],
                'inferred_dtype': 'varchar',
                'null_count': int(min_series.isna().sum()),
                'null_percentage': round((min_series.isna().sum() / len(min_series) * 100), 2) if len(min_series) > 0 else 0,
                'unique_count': int(min_series.nunique(dropna=True)),
                'sample_values': [str(v) for v in min_series.dropna().head(5).tolist()],
                'semantic_type': 'unknown',
                'semantic_confidence': 0.0,
                'semantic_notes': f'Extracted from range column: {split_info["original"]}'
            })
            
            # Add max column
            col_summaries.append({
                'column_name': split_info['max_col'],
                'inferred_dtype': 'varchar',
                'null_count': int(max_series.isna().sum()),
                'null_percentage': round((max_series.isna().sum() / len(max_series) * 100), 2) if len(max_series) > 0 else 0,
                'unique_count': int(max_series.nunique(dropna=True)),
                'sample_values': [str(v) for v in max_series.dropna().head(5).tolist()],
                'semantic_type': 'unknown',
                'semantic_confidence': 0.0,
                'semantic_notes': f'Extracted from range column: {split_info["original"]}'
            })
            
            metadata['column_wise_summary'] = col_summaries
        
        # Update column count
        metadata['number_of_columns'] = len(df.columns)
        
        print(f"     [RANGES] Split {len(split_columns_info)} range column(s)")
    
    split_column_names = [info['original'] for info in split_columns_info]
    return df, metadata, split_column_names


# ==============================
# SEMANTIC DETECTION & NORMALIZATION
# ==============================

def detect_and_normalize_column(column: pd.Series, patterns: Dict) -> Tuple[pd.Series, str, float]:
    """
    Detect semantic type and normalize a column using patterns.
    
    Args:
        column: Pandas Series to process
        patterns: Dictionary of pattern instances
    
    Returns:
        Tuple of (normalized_series, semantic_type, confidence)
    """
    # Detect semantic type
    non_null = column.dropna()
    if len(non_null) == 0:
        return column, 'unknown', 0.0
    
    # Sample for detection
    sample_size = min(100, len(non_null))
    sample_series = non_null.head(sample_size)
    
    # Test against all patterns
    pattern_scores = {}
    for semantic_type, pattern_instance in patterns.items():
        try:
            confidence = pattern_instance.detect(sample_series)
            pattern_scores[semantic_type] = confidence
        except Exception as e:
            pattern_scores[semantic_type] = 0.0
    
    # Find best match
    if not pattern_scores:
        return column, 'unknown', 0.0
    
    sorted_scores = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
    best_semantic, best_confidence = sorted_scores[0]
    
    # If confidence >= 50%, normalize using pattern
    if best_confidence >= 0.5:
        try:
            pattern_instance = patterns[best_semantic]
            normalized = pattern_instance.normalize(column)
            return normalized, best_semantic, best_confidence
        except Exception as e:
            print(f"        [ERROR] Failed to normalize {column.name} as {best_semantic}: {e}")
            return column, best_semantic, best_confidence
    else:
        return column, 'unknown', best_confidence


# ==============================
# MAIN PROCESSING FUNCTION
# ==============================

def process_varchar_columns(userid: str, sessionid: str, filename: str):
    """
    Main function to process varchar columns for a dataset.
    
    Process:
    1. Load cleaned dataset
    2. Load profiling metadata
    3. Identify varchar columns
    4. For each varchar column:
       - Check if ID-like → SKIP
       - Check if long text → SKIP
       - Check if extractable → EXTRACT components
    5. Handle ranges
    6. Re-detect and normalize new columns
    7. Update dataset and metadata
    
    Args:
        userid: User ID
        sessionid: Session ID
        filename: Dataset filename (without _cleaned.csv suffix)
    """
    print(f"\n{'='*80}")
    print(f"COLUMN TYPE RESOLVER - Processing: {userid}/{sessionid}/{filename}")
    print(f"{'='*80}\n")
    
    # Load cleaned dataset
    dataset_storage_path = f"output/{userid}/{sessionid}/{filename}_cleaned.csv"
    print(f"[1/7] Loading cleaned dataset: {dataset_storage_path}")
    
    try:
        content = download_file(dataset_storage_path)
        df = pd.read_csv(BytesIO(content))
        print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"      ERROR: Failed to load dataset: {e}")
        return
    
    # Load profiling metadata
    metadata_storage_path = f"meta_data/{userid}/{sessionid}/{filename}_profiling.json"
    print(f"\n[2/7] Loading profiling metadata: {metadata_storage_path}")
    
    try:
        metadata = download_json(metadata_storage_path)
        print(f"      Loaded metadata with {len(metadata.get('column_wise_summary', []))} column summaries")
    except Exception as e:
        print(f"      ERROR: Failed to load metadata: {e}")
        return
    
    # Identify varchar columns
    print(f"\n[3/7] Identifying varchar columns...")
    varchar_columns = []
    
    for col_summary in metadata.get('column_wise_summary', []):
        col_name = col_summary['column_name']
        semantic_type = col_summary.get('semantic_type', '').lower()
        
        # ONLY process columns with semantic_type = "varchar" or "unknown"
        if semantic_type in ['varchar', 'unknown'] and col_name in df.columns:
            varchar_columns.append(col_name)
    
    print(f"      Found {len(varchar_columns)} varchar columns: {varchar_columns}")
    
    if not varchar_columns:
        print(f"      No varchar columns to process.")
        return
    
    # Process each varchar column
    print(f"\n[4/7] Processing varchar columns...")
    
    columns_to_drop = []
    extracted_columns = {}
    
    for col_name in varchar_columns:
        print(f"\n   Processing: {col_name}")
        column = df[col_name]
        
        # Check 1: ID-like?
        if is_id_like(column):
            print(f"      → KEEP (ID-like - no extraction needed)")
            continue
        
        # Check 2: Long text?
        if is_long_text(column):
            print(f"      → KEEP (Long unstructured text - no extraction needed)")
            continue
        
        # Check 3: Extractable?
        if is_extractable_numeric_text(column):
            print(f"      → EXTRACT (Numeric + short text)")
            
            # Extract components for all rows
            extracted_data = []
            for val in column:
                components = extract_structured_components(val)
                extracted_data.append(components)
            
            # Determine which columns to create based on what was actually extracted
            # Check first few rows to see what fields are present
            sample_components = [c for c in extracted_data[:10] if c]
            all_keys = set()
            for comp in sample_components:
                all_keys.update(comp.keys())
            
            # Track columns created for this column
            col_component_count = 0
            
            # Create columns for all extracted keys
            for key in all_keys:
                new_col_name = f"{col_name}_{key}"
                
                # Extract values for this key
                values = []
                for comp in extracted_data:
                    values.append(comp.get(key, np.nan))
                
                df[new_col_name] = values
                extracted_columns[new_col_name] = pd.Series(values)
                col_component_count += 1
            
            print(f"         Extracted {col_component_count} components")
            
            # Drop original column
            columns_to_drop.append(col_name)
        else:
            print(f"      → KEEP (No action needed)")
    
    # Drop skipped/extracted columns
    if columns_to_drop:
        print(f"\n   Dropping {len(columns_to_drop)} columns: {columns_to_drop}")
        df.drop(columns=columns_to_drop, inplace=True)
        
        # Update metadata
        col_summaries = metadata.get('column_wise_summary', [])
        metadata['column_wise_summary'] = [
            s for s in col_summaries if s['column_name'] not in columns_to_drop
        ]
    
    # Add extracted columns to metadata
    if extracted_columns:
        print(f"\n   Adding {len(extracted_columns)} extracted columns to metadata...")
        col_summaries = metadata.get('column_wise_summary', [])
        
        for col_name, col_data in extracted_columns.items():
            # Infer dtype
            if col_data.dtype == 'bool':
                dtype = 'boolean'
            elif pd.api.types.is_integer_dtype(col_data):
                dtype = 'integer'
            elif pd.api.types.is_float_dtype(col_data):
                dtype = 'float'
            else:
                dtype = 'varchar'
            
            col_summaries.append({
                'column_name': col_name,
                'inferred_dtype': dtype,
                'null_count': int(col_data.isna().sum()),
                'null_percentage': round((col_data.isna().sum() / len(col_data) * 100), 2) if len(col_data) > 0 else 0,
                'unique_count': int(col_data.nunique(dropna=True)),
                'sample_values': [str(v) for v in col_data.dropna().head(5).tolist()],
                'semantic_type': 'unknown',
                'semantic_confidence': 0.0,
                'semantic_notes': 'Extracted from varchar column'
            })
        
        metadata['column_wise_summary'] = col_summaries
    
    # Update column count
    metadata['number_of_columns'] = len(df.columns)
    
    # Re-detect and normalize new columns
    print(f"\n[5/7] Re-detecting and normalizing new columns...")
    
    # Load patterns
    patterns = PATTERNS
    print(f"      Loaded {len(patterns)} patterns")
    
    # Get all new column names (only extracted columns)
    new_column_names = list(extracted_columns.keys())
    
    # Filter to columns that still exist in DataFrame
    new_column_names = [col for col in new_column_names if col in df.columns]
    
    print(f"      Processing {len(new_column_names)} new columns...")
    
    # Track column renames
    column_renames = {}
    
    for col_name in new_column_names:
        
        # Check if this column should be processed based on metadata semantic_type
        # ONLY process columns with semantic_type == "varchar" or "unknown"
        should_process = False
        col_summaries = metadata.get('column_wise_summary', [])
        for summary in col_summaries:
            if summary['column_name'] == col_name:
                semantic_type_in_meta = summary.get('semantic_type', 'unknown')
                # Only process varchar and unknown types
                if semantic_type_in_meta in ['varchar', 'unknown']:
                    should_process = True
                break
        
        if not should_process:
            print(f"         {col_name}: SKIPPED (already has semantic type)")
            continue
        
        # Detect and normalize
        normalized_series, semantic_type, confidence = detect_and_normalize_column(
            df[col_name], patterns
        )
        
        # Update DataFrame with normalized values
        df[col_name] = normalized_series
        
        # Determine new column name based on semantic type (like column_handler.py)
        new_col_name = col_name
        
        # Get pattern instance for additional info
        if semantic_type in patterns:
            pattern_instance = patterns[semantic_type]
            
            # Currency: Rename column to include currency code
            if semantic_type == 'currency':
                if hasattr(pattern_instance, 'detected_currency') and pattern_instance.detected_currency:
                    # Single currency detected
                    currency_code = pattern_instance.detected_currency
                    # Check if column already ends with value_min or value_max
                    if col_name.endswith('_value_min'):
                        base_name = col_name[:-10]  # Remove '_value_min'
                        new_col_name = f"{base_name}_value_min_({currency_code})"
                    elif col_name.endswith('_value_max'):
                        base_name = col_name[:-10]  # Remove '_value_max'
                        new_col_name = f"{base_name}_value_max_({currency_code})"
                    elif col_name.endswith('_value'):
                        base_name = col_name[:-6]  # Remove '_value'
                        new_col_name = f"{base_name}_value_({currency_code})"
                    else:
                        new_col_name = f"{col_name}_({currency_code})"
            
            # Measurement units: rename with unit suffix
            elif hasattr(pattern_instance, 'detected_unit') and pattern_instance.detected_unit:
                unit = pattern_instance.detected_unit
                # Check if column already ends with value_min or value_max
                if col_name.endswith('_value_min'):
                    base_name = col_name[:-10]
                    new_col_name = f"{base_name}_value_min_({unit})"
                elif col_name.endswith('_value_max'):
                    base_name = col_name[:-10]
                    new_col_name = f"{base_name}_value_max_({unit})"
                elif col_name.endswith('_value'):
                    base_name = col_name[:-6]
                    new_col_name = f"{base_name}_value_({unit})"
                else:
                    new_col_name = f"{col_name}_({unit})"
        
        # Rename column if needed
        if new_col_name != col_name:
            df.rename(columns={col_name: new_col_name}, inplace=True)
            column_renames[col_name] = new_col_name
            print(f"         {col_name} → {new_col_name}: {semantic_type} (confidence: {confidence:.2f})")
        else:
            print(f"         {col_name}: {semantic_type} (confidence: {confidence:.2f})")
        
        # Update metadata with normalized values
        col_summaries = metadata.get('column_wise_summary', [])
        for summary in col_summaries:
            # Update using the new column name if it was renamed
            current_col_name = column_renames.get(col_name, col_name)
            
            if summary['column_name'] == col_name:
                # Update column name in metadata if it was renamed
                summary['column_name'] = current_col_name
                summary['semantic_type'] = semantic_type
                summary['semantic_confidence'] = round(confidence, 2)
                summary['inferred_dtype'] = str(df[current_col_name].dtype)
                
                # Update sample_values with normalized data
                summary['sample_values'] = [str(v) for v in df[current_col_name].dropna().head(5).tolist()]
                
                # Update statistics with normalized data
                summary['null_count'] = int(df[current_col_name].isna().sum())
                summary['null_percentage'] = round((df[current_col_name].isna().sum() / len(df[current_col_name]) * 100), 2) if len(df[current_col_name]) > 0 else 0
                summary['unique_count'] = int(df[current_col_name].nunique(dropna=True))
                break
    
    # Save updated dataset
    print(f"\n[6/7] Saving updated dataset...")
    
    try:
        # Save CSV to Supabase
        content = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
        upload_file(dataset_storage_path, content, "text/csv")
        print(f"      ✓ Saved dataset: {dataset_storage_path}")
        
        # Save metadata to Supabase
        upload_json(metadata_storage_path, metadata)
        print(f"      ✓ Saved metadata: {metadata_storage_path}")
        
    except Exception as e:
        print(f"      ERROR: Failed to save: {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"  Dropped columns: {len(columns_to_drop)}")
    print(f"  Extracted columns: {len(extracted_columns)}")
    print(f"  Final columns: {len(df.columns)}")
    print(f"{'='*80}\n")


# ==============================
# CLI INTERFACE
# ==============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Column Type Resolver")
    parser.add_argument("--userid", type=str, help="User ID", required=False)
    parser.add_argument("--sessionid", type=str, help="Session ID", required=False)
    parser.add_argument("--filename", type=str, help="Dataset filename (without _cleaned.csv)", required=False)
    
    args = parser.parse_args()
    
    # Get userid (from args or prompt)
    userid = args.userid
    if not userid:
        userid = input("Enter User ID: ").strip()
        if not userid:
            print("Error: User ID is required")
            exit(1)
    
    # Get sessionid (from args or prompt)
    sessionid = args.sessionid
    if not sessionid:
        sessionid = input("Enter Session ID: ").strip()
        if not sessionid:
            print("Error: Session ID is required")
            exit(1)
    
    if args.filename:
        # Process single dataset
        process_varchar_columns(userid, sessionid, args.filename)
    else:
        # Process all datasets for this user/session
        folder_prefix = f"output/{userid}/{sessionid}"
        
        try:
            all_files = list_files(folder_prefix)
            cleaned_files = [f for f in all_files if f.endswith('_cleaned.csv')]
            
            if not cleaned_files:
                print(f"No datasets found for user {userid}, session {sessionid}")
                exit(1)
            
            print(f"\nProcessing all datasets for user {userid}, session {sessionid}:")
            print(f"Found {len(cleaned_files)} dataset(s)\n")
            
            for cleaned_file in cleaned_files:
                filename = cleaned_file.replace('_cleaned.csv', '')
                process_varchar_columns(userid, sessionid, filename)
                print()  # Blank line between datasets
                
        except Exception as e:
            print(f"Error listing files: {e}")
            exit(1)


# ==============================
# PIPELINE API — matches other preprocessing modules
# ==============================

def process_user_datasets(userid: str, sessionid: str):
    """
    Process all cleaned datasets for a user session.
    Loops over every *_cleaned.csv in output/{userid}/{sessionid}/ and
    runs process_varchar_columns on each one.

    Args:
        userid: User ID
        sessionid: Session ID
    """
    print("=" * 70)
    print("COLUMN TYPE RESOLVER - AutoML Preprocessing System")
    print("=" * 70)
    print(f"User ID:    {userid}")
    print(f"Session ID: {sessionid}")
    print()

    folder_prefix = f"output/{userid}/{sessionid}"
    print(f" Searching for datasets in: {folder_prefix}")

    try:
        all_files = list_files(folder_prefix)
    except Exception as e:
        print(f"  Error listing files: {e}")
        return

    cleaned_files = [f for f in all_files if f.endswith('_cleaned.csv')]

    if not cleaned_files:
        print(f"  No cleaned datasets found for user: {userid}, session: {sessionid}")
        return

    print(f" Found {len(cleaned_files)} dataset(s)")

    success_count = 0
    error_count = 0

    for cleaned_file in cleaned_files:
        filename = cleaned_file.replace('_cleaned.csv', '')
        try:
            process_varchar_columns(userid, sessionid, filename)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"\n  Error processing {cleaned_file}: {e}")

    print("\n" + "=" * 70)
    print(" PROCESSING SUMMARY")
    print("=" * 70)
    print(f" Successfully processed: {success_count} dataset(s)")
    if error_count > 0:
        print(f" Errors encountered:     {error_count} dataset(s)")
    print()
    print(" Column type resolution completed!")
    print("=" * 70)
