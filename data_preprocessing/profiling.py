"""
Data Profiling Module for AutoML Preprocessing System

This module performs comprehensive profiling of datasets without any modifications.
It extracts metadata and statistics for each dataset found in the user's input directory.

Author: AutoML Preprocessing System
Date: 2025-12-23
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, List, Any, Tuple

try:
    from .supabase_storage import download_file, upload_json, list_files
except ImportError:
    from supabase_storage import download_file, upload_json, list_files


def infer_dtype(series: pd.Series) -> str:
    """
    Infer the data type of a pandas Series.
    
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


def get_sample_values(series: pd.Series, n: int = 5) -> List[Any]:
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


def check_mixed_types(series: pd.Series) -> bool:
    """
    Check if a column contains mixed data types.
    
    Args:
        series: Pandas Series to check
    
    Returns:
        True if mixed types are detected, False otherwise
    """
    if not pd.api.types.is_object_dtype(series.dtype):
        return False
    
    # Sample non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    
    # Check type consistency in a sample
    sample = non_null.head(100)
    types_found = set()
    
    for val in sample:
        if isinstance(val, bool):
            types_found.add('bool')
        elif isinstance(val, (int, np.integer)):
            types_found.add('int')
        elif isinstance(val, (float, np.floating)):
            types_found.add('float')
        elif isinstance(val, str):
            types_found.add('str')
        else:
            types_found.add('other')
    
    return len(types_found) > 1


def profile_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Profile a single column and extract detailed statistics.
    
    Args:
        series: Pandas Series to profile
        column_name: Name of the column
    
    Returns:
        Dictionary containing column-level profiling information
    """
    total_count = len(series)
    null_count = int(series.isna().sum())
    null_percentage = round((null_count / total_count * 100), 2) if total_count > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    
    profile = {
        "column_name": column_name,
        "inferred_dtype": infer_dtype(series),
        "null_count": null_count,
        "null_percentage": null_percentage,
        "unique_count": unique_count,
        "sample_values": get_sample_values(series, n=5)
    }
    
    return profile


def detect_potential_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Detect columns that might be ID columns based on uniqueness ratio.
    
    Args:
        df: Pandas DataFrame to analyze
        threshold: Uniqueness ratio threshold (default: 0.95)
    
    Returns:
        List of column names that are potential ID columns
    """
    potential_ids = []
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        if non_null_count == 0:
            continue
        
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / non_null_count
        
        if unique_ratio > threshold:
            potential_ids.append(col)
    
    return potential_ids


def profile_dataset(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Perform comprehensive profiling of an entire dataset.
    
    Args:
        df: Pandas DataFrame to profile
        filename: Name of the dataset file
    
    Returns:
        Dictionary containing complete profiling information
    """
    print(f"  📊 Profiling dataset: {filename}")
    
    # Dataset-level statistics
    num_rows, num_cols = df.shape
    
    # Column-wise profiling
    column_summaries = []
    has_mixed_types_flag = False
    
    for col in df.columns:
        col_profile = profile_column(df[col], col)
        column_summaries.append(col_profile)
        
        # Check for mixed types
        if check_mixed_types(df[col]):
            has_mixed_types_flag = True
    
    # Dataset-level flags
    has_missing = df.isna().any().any()
    has_duplicates = df.duplicated().any()
    potential_id_cols = detect_potential_id_columns(df)
    
    # Compile complete profile
    profile = {
        "file_name": filename,
        "number_of_rows": int(num_rows),
        "number_of_columns": int(num_cols),
        "column_wise_summary": column_summaries,
        "dataset_level_flags": {
            "has_missing_values": bool(has_missing),
            "has_duplicates": bool(has_duplicates),
            "has_mixed_types": has_mixed_types_flag,
            "has_potential_id_columns": len(potential_id_cols) > 0,
            "potential_id_columns": potential_id_cols
        }
    }
    
    print(f"  ✅ Profiling completed for {filename}")
    print(f"     Rows: {num_rows}, Columns: {num_cols}")
    print(f"     Missing: {has_missing}, Duplicates: {has_duplicates}")
    
    return profile


def load_dataset_from_storage(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from Supabase storage.
    Robust CSV loading with multiple fallback strategies.

    Args:
        filepath: Supabase storage path, e.g. "input/user1/session1/data.csv"

    Returns:
        Pandas DataFrame containing the dataset
    """
    file_content = download_file(filepath)
    if False:  # placeholder to keep the block structure intact
        pass
    with BytesIO(file_content) as f:
        file_content = f.read()
    
    # Determine file type and load accordingly
    if filepath.lower().endswith('.csv'):
        # Try multiple strategies to load CSV robustly
        strategies = [
            # Strategy 1: Standard CSV with all quoting
            {'quoting': 3, 'encoding': 'utf-8', 'index_col': False},
            # Strategy 2: Ignore errors with error_bad_lines (older pandas)
            {'encoding': 'utf-8', 'quotechar': '"', 'error_bad_lines': False, 'index_col': False},
            # Strategy 3: Try with different encoding
            {'encoding': 'latin-1', 'error_bad_lines': False, 'index_col': False},
            # Strategy 4: Engine python (slower but more robust)
            {'engine': 'python', 'encoding': 'utf-8', 'error_bad_lines': False, 'index_col': False},
            # Strategy 5: Last resort - skip initial space
            {'engine': 'python', 'encoding': 'utf-8', 'skipinitialspace': True, 'index_col': False}
        ]
        
        last_error = None
        for i, params in enumerate(strategies):
            try:
                df = pd.read_csv(BytesIO(file_content), **params)
                # Success - return the dataframe
                return df
            except (pd.errors.ParserError, TypeError, Exception) as e:
                last_error = e
                continue
        
        # If all strategies failed, raise the last error
        if last_error:
            raise last_error
            
    elif filepath.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(BytesIO(file_content))
    elif filepath.lower().endswith('.json'):
        df = pd.read_json(BytesIO(file_content))
    elif filepath.lower().endswith('.parquet'):
        df = pd.read_parquet(BytesIO(file_content))
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return df


def save_profile_to_storage(profile: Dict[str, Any], userid: str, sessionid: str, filename: str):
    """
    Save profiling results to Supabase storage as JSON.

    Args:
        profile:   Profiling dictionary to save
        userid:    User ID
        sessionid: Session ID
        filename:  Original dataset filename
    """
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_profiling.json"
    path = f"meta_data/{userid}/{sessionid}/{output_filename}"
    upload_json(path, profile)
    print(f"  💾 Saved profiling to: {path}")


def process_user_datasets(userid: str, sessionid: str):
    """
    Main processing function: profile all datasets for a given user/session.

    Args:
        userid:    User identifier
        sessionid: Session identifier
    """
    print("=" * 70)
    print("🚀 DATA PROFILING - AutoML Preprocessing System")
    print("=" * 70)
    print(f"👤 User ID:    {userid}")
    print(f"🔑 Session ID: {sessionid}")
    print()

    folder = f"input/{userid}/{sessionid}"
    print(f"📂 Searching for datasets in: {folder}")

    try:
        files = list_files(folder)
    except Exception as e:
        print(f"❌ Failed to list files in Supabase: {e}")
        sys.exit(1)

    if not files:
        print(f"⚠️  No datasets found in: {folder}")
        return

    print(f"✅ Found {len(files)} dataset(s)")
    print()

    success_count = 0
    error_count = 0

    for filename in files:
        try:
            print(f"{'=' * 70}")
            print(f"Processing: {filename}")
            print(f"{'-' * 70}")

            file_path = f"input/{userid}/{sessionid}/{filename}"
            print(f"  📥 Loading dataset...")
            df = load_dataset_from_storage(file_path)

            profile = profile_dataset(df, filename)
            save_profile_to_storage(profile, userid, sessionid, filename)

            success_count += 1
            print()

        except Exception as e:
            error_count += 1
            print(f"  ❌ Error processing {filename}: {str(e)}")
            print()
            continue

    print("=" * 70)
    print("📊 PROFILING SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully profiled: {success_count} dataset(s)")
    if error_count > 0:
        print(f"❌ Errors encountered: {error_count} dataset(s)")
    print()
    print("🎉 Profiling completed!")
    print("=" * 70)


def main():
    """Main entry point for the profiling script."""
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
        print("❌ Error: User ID cannot be empty")
        sys.exit(1)
    if not sessionid:
        print("❌ Error: Session ID cannot be empty")
        sys.exit(1)

    print()
    process_user_datasets(userid, sessionid)


if __name__ == "__main__":
    main()
