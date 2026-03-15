"""
Row Handler Module for AutoML Preprocessing System

This module performs row-level operations on cleaned datasets:
- Remove duplicate rows
- Update profiling metadata with new row counts

Author: AutoML Preprocessing System
Date: 2025-12-28
"""

import os
import sys
import json
import pandas as pd
from io import BytesIO, StringIO
from typing import Dict, Any, List
from datetime import datetime
try:
    from .supabase_storage import download_file, upload_file, download_json, upload_json, list_files
except ImportError:
    from supabase_storage import download_file, upload_file, download_json, upload_json, list_files


# ==============================
# METADATA OPERATIONS
# ==============================

def load_profiling_metadata(userid: str, sessionid: str, filename: str) -> Dict[str, Any]:
    """
    Load existing profiling metadata for a dataset from Supabase storage.
    
    Args:
        userid: User ID
        sessionid: Session ID
        filename: Dataset filename
    
    Returns:
        Profiling metadata dictionary or None if not found
    """
    base_name = os.path.splitext(filename)[0]
    # Remove _cleaned suffix if present
    if base_name.endswith('_cleaned'):
        base_name = base_name[:-8]
    
    storage_path = f"meta_data/{userid}/{sessionid}/{base_name}_profiling.json"
    
    try:
        metadata = download_json(storage_path)
        return metadata
    except Exception as e:
        print(f"    Warning: Failed to load profiling metadata: {e}")
        return None


def save_profiling_metadata(metadata: Dict[str, Any], userid: str, sessionid: str, filename: str):
    """
    Save updated profiling metadata back to Supabase storage.
    
    Args:
        metadata: Updated profiling metadata
        userid: User ID
        sessionid: Session ID
        filename: Original dataset filename
    """
    base_name = os.path.splitext(filename)[0]
    # Remove _cleaned suffix if present
    if base_name.endswith('_cleaned'):
        base_name = base_name[:-8]
    
    storage_path = f"meta_data/{userid}/{sessionid}/{base_name}_profiling.json"
    upload_json(storage_path, metadata)
    print(f"   Updated profiling metadata: {storage_path}")


# ==============================
# DATASET OPERATIONS
# ==============================

def load_dataset_from_storage(filepath: str) -> pd.DataFrame:
    """
    Load a dataset from Supabase storage.
    
    Args:
        filepath: Supabase storage path, e.g. "output/user1/session1/data.csv"
    
    Returns:
        Pandas DataFrame
    """
    content = download_file(filepath)
    file_buffer = BytesIO(content)
    
    if filepath.lower().endswith('.csv'):
        # Try different encodings for CSV files
        try:
            df = pd.read_csv(file_buffer, encoding='utf-8')
        except UnicodeDecodeError:
            file_buffer.seek(0)
            try:
                df = pd.read_csv(file_buffer, encoding='latin-1')
            except:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding='cp1252')
    elif filepath.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_buffer)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Preserve integer types by converting float columns with NaN to nullable Int64
    # This prevents integer columns from being converted to float when they have missing values
    for col in df.columns:
        if df[col].dtype == 'float64':
            # Check if this column contains only integer-like values (ignoring NaN)
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                # Check if all non-null values are whole numbers
                if (non_null_values % 1 == 0).all():
                    # Convert to nullable integer type (Int64 can hold NaN)
                    df[col] = df[col].astype('Int64')
    
    return df


def save_dataset_to_storage(df: pd.DataFrame, userid: str, sessionid: str, filename: str):
    """
    Save cleaned dataset back to Supabase storage output directory.
    
    Args:
        df: DataFrame to save
        userid: User ID
        sessionid: Session ID
        filename: Original filename (will preserve _cleaned suffix if present)
    """
    storage_path = f"output/{userid}/{sessionid}/{filename}"
    content = df.to_csv(index=False, encoding='utf-8').encode('utf-8')
    upload_file(storage_path, content, "text/csv")
    print(f"   Saved cleaned dataset: {storage_path}")


# ==============================
# ROW-LEVEL PROCESSING
# ==============================

def drop_rows_with_missing_values(df: pd.DataFrame, metadata: Dict[str, Any], threshold: float = 0.70) -> pd.DataFrame:
    """
    Drop rows where missing values >= threshold percentage.
    
    Args:
        df: Pandas DataFrame
        metadata: Profiling metadata dictionary
        threshold: Minimum percentage of missing values to drop row (default: 0.70 = 70%)
    
    Returns:
        DataFrame with high-missing-value rows removed
    """
    original_rows = len(df)
    
    print(f"   Checking for rows with >= {threshold*100:.0f}% missing values...")
    print(f"     Original row count: {original_rows}")
    
    # Calculate missing percentage per row
    total_cols = len(df.columns)
    missing_per_row = df.isna().sum(axis=1)
    missing_percentage_per_row = missing_per_row / total_cols
    
    # Identify rows to drop
    rows_to_drop = missing_percentage_per_row >= threshold
    rows_dropped_count = rows_to_drop.sum()
    
    # Drop rows
    df_cleaned = df[~rows_to_drop].copy()
    
    new_rows = len(df_cleaned)
    
    if rows_dropped_count > 0:
        print(f"     [MISSING VALUES] {rows_dropped_count} row(s) dropped (>= {threshold*100:.0f}% missing)")
        print(f"     New row count: {new_rows}")
        
        # Update metadata
        if metadata:
            if 'row_processing_summary' not in metadata:
                metadata['row_processing_summary'] = {}
            
            metadata['row_processing_summary']['dropped_missing_rows'] = int(rows_dropped_count)
    else:
        print(f"     No rows with >= {threshold*100:.0f}% missing values")
        
        # Update metadata
        if metadata:
            if 'row_processing_summary' not in metadata:
                metadata['row_processing_summary'] = {}
            
            metadata['row_processing_summary']['dropped_missing_rows'] = 0
    
    return df_cleaned


def remove_duplicate_rows(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame and update metadata.
    
    Args:
        df: Pandas DataFrame
        metadata: Profiling metadata dictionary
    
    Returns:
        DataFrame with duplicates removed
    """
    original_rows = len(df)
    
    print(f"   Checking for duplicate rows...")
    print(f"     Original row count: {original_rows}")
    
    # Remove duplicate rows (keep first occurrence)
    df_cleaned = df.drop_duplicates(keep='first')
    
    new_rows = len(df_cleaned)
    duplicates_removed = original_rows - new_rows
    
    if duplicates_removed > 0:
        print(f"     [DUPLICATES REMOVED] {duplicates_removed} duplicate row(s) removed")
        print(f"     New row count: {new_rows}")
        
        # Update metadata
        if metadata:
            metadata['number_of_rows'] = new_rows
            
            # Update dataset-level flags
            if 'dataset_level_flags' not in metadata:
                metadata['dataset_level_flags'] = {}
            metadata['dataset_level_flags']['has_duplicates'] = False
            
            # Add row processing summary
            if 'row_processing_summary' not in metadata:
                metadata['row_processing_summary'] = {}
            
            metadata['row_processing_summary'].update({
                'original_row_count': original_rows,
                'duplicate_rows_removed': duplicates_removed,
                'final_row_count': new_rows,
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Update null percentages for all columns (based on new row count)
            if 'column_wise_summary' in metadata:
                for col_summary in metadata['column_wise_summary']:
                    col_name = col_summary['column_name']
                    if col_name in df_cleaned.columns:
                        null_count = df_cleaned[col_name].isna().sum()
                        null_percentage = (null_count / new_rows * 100) if new_rows > 0 else 0
                        col_summary['null_count'] = int(null_count)
                        col_summary['null_percentage'] = round(null_percentage, 2)
    else:
        print(f"     No duplicate rows found")
        
        # Update metadata to indicate no duplicates
        if metadata:
            if 'dataset_level_flags' not in metadata:
                metadata['dataset_level_flags'] = {}
            metadata['dataset_level_flags']['has_duplicates'] = False
            
            # Add row processing summary
            if 'row_processing_summary' not in metadata:
                metadata['row_processing_summary'] = {}
            
            metadata['row_processing_summary'].update({
                'original_row_count': original_rows,
                'duplicate_rows_removed': 0,
                'final_row_count': new_rows,
                'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return df_cleaned


def process_dataset(filepath: str, userid: str, sessionid: str, filename: str) -> bool:
    """
    Process a single dataset: remove duplicate rows and update metadata.
    
    Args:
        filepath: Supabase storage path to the dataset file
        userid: User ID
        sessionid: Session ID
        filename: Dataset filename
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 70}")
    print(f"Processing: {filename}")
    print(f"{'-' * 70}")
    
    try:
        # Load dataset
        print(f"   Loading dataset from: {filepath}")
        df = load_dataset_from_storage(filepath)
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Load profiling metadata
        metadata = load_profiling_metadata(userid, sessionid, filename)
        
        # Step 1: Drop rows with >= 70% missing values
        df_cleaned = drop_rows_with_missing_values(df, metadata, threshold=0.70)
        
        # Step 2: Remove duplicate rows
        df_cleaned = remove_duplicate_rows(df_cleaned, metadata)
        
        # Update final row count in metadata
        if metadata:
            metadata['number_of_rows'] = len(df_cleaned)
            
            # Update null percentages for all columns (based on final row count)
            if 'column_wise_summary' in metadata:
                for col_summary in metadata['column_wise_summary']:
                    col_name = col_summary['column_name']
                    if col_name in df_cleaned.columns:
                        null_count = df_cleaned[col_name].isna().sum()
                        null_percentage = (null_count / len(df_cleaned) * 100) if len(df_cleaned) > 0 else 0
                        col_summary['null_count'] = int(null_count)
                        col_summary['null_percentage'] = round(null_percentage, 2)
            
            # Update final row count in row processing summary
            if 'row_processing_summary' in metadata:
                metadata['row_processing_summary']['final_row_count'] = len(df_cleaned)
        
        # Save cleaned dataset back to storage
        save_dataset_to_storage(df_cleaned, userid, sessionid, filename)
        
        # Save updated metadata
        if metadata:
            save_profiling_metadata(metadata, userid, sessionid, filename)
        
        print(f"   ✓ Processing complete")
        return True
        
    except Exception as e:
        print(f"   ✗ Error processing {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==============================
# MAIN PROCESSING
# ==============================

def process_user_datasets(userid: str, sessionid: str):
    """
    Main processing function: process all datasets in output folder for a user session.
    
    Args:
        userid: User identifier
        sessionid: Session identifier
    """
    print("=" * 70)
    print("ROW HANDLER - AutoML Preprocessing System")
    print("=" * 70)
    print(f"User ID: {userid}")
    print(f"Session ID: {sessionid}")
    print()
    
    try:
        # List all files in output directory for this session
        folder_prefix = f"output/{userid}/{sessionid}"
        print(f" Searching for datasets in: {folder_prefix}")
        
        try:
            all_files = list_files(folder_prefix)
        except Exception as e:
            print(f"  Error listing files: {e}")
            print(f"  No datasets found for user: {userid}, session: {sessionid}")
            return
        
        # Filter for supported file formats
        supported_formats = ['.csv', '.xlsx', '.xls']
        dataset_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in supported_formats)]
        
        if not dataset_files:
            print(f"  No dataset files found for user: {userid}, session: {sessionid}")
            return
        
        print(f" Found {len(dataset_files)} dataset(s)")
        
        # Process each dataset
        success_count = 0
        error_count = 0
        
        for filename in dataset_files:
            file_path = f"output/{userid}/{sessionid}/{filename}"
            
            if process_dataset(file_path, userid, sessionid, filename):
                success_count += 1
            else:
                error_count += 1
        
        # Final summary
        print("\n" + "=" * 70)
        print(" PROCESSING SUMMARY")
        print("=" * 70)
        print(f" Successfully processed: {success_count} dataset(s)")
        if error_count > 0:
            print(f" Errors encountered: {error_count} dataset(s)")
        print()
        print(" Row handling completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f" Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main entry point for the row handler script.
    """
    print()
    
    # Get user ID and session ID from command line or prompt
    if len(sys.argv) > 1:
        userid = sys.argv[1]
    else:
        userid = input("Enter User ID: ").strip()
    
    if not userid:
        print(" Error: User ID cannot be empty")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        sessionid = sys.argv[2]
    else:
        sessionid = input("Enter Session ID: ").strip()
    
    if not sessionid:
        print(" Error: Session ID cannot be empty")
        sys.exit(1)
    
    print()
    
    # Process datasets
    process_user_datasets(userid, sessionid)


if __name__ == "__main__":
    main()
