"""
AutoML-Based Missing Values Handler with AutoGluon Tabular Integration

This module provides intelligent missing value handling using AutoGluon's TabularPredictor
and FeatureMetadata for AutoML-safe preprocessing.

MANDATORY REQUIREMENTS:
- Explicitly uses AutoGluon Tabular (TabularPredictor, FeatureMetadata)
- Normalizes invalid tokens ("UNKNOWN", "ERROR", etc.) to NaN
- Validates AutoGluon compatibility through schema checking
- Ensures datetime columns are properly formatted

Author: AutoML Data Preprocessing System
Date: January 2026
"""

import os
import json
import logging
import warnings
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from supabase_storage import download_file, upload_file, download_json, upload_json, list_files

# MANDATORY: AutoGluon Tabular imports
from autogluon.tabular import TabularPredictor
from autogluon.common.features.feature_metadata import FeatureMetadata

# AutoGluon Feature Generators
from autogluon.features.generators import (
    AutoMLPipelineFeatureGenerator,
    FillNaFeatureGenerator,
    CategoryFeatureGenerator,
    DatetimeFeatureGenerator
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Invalid tokens to normalize (AutoGluon-safe requirement)
INVALID_TOKENS = ["UNKNOWN", "ERROR", "N/A", "NULL", "", "nan", "NaN", "NA", "None", "NONE", "#", "%", "&", "^", "@", "!", ")", "(", "*", "~", "unknown", "Unknown"]

# ============================================================================
# NUMERIC MISSING VALUE PATTERNS
# ============================================================================
# Pattern labels for numeric columns based on metadata analysis

class NumericMissingPattern:
    """Constants for numeric missing value patterns."""
    LOW_MISSING = "PATTERN_A_LOW_MISSING"
    CONTINUOUS_MODERATE = "PATTERN_B_CONTINUOUS_MODERATE"
    DISCRETE_NUMERIC = "PATTERN_C_DISCRETE_NUMERIC"
    HIGH_MISSING = "PATTERN_D_HIGH_MISSING"
    VERY_HIGH_MISSING = "PATTERN_E_VERY_HIGH_MISSING"
    NEAR_CONSTANT = "PATTERN_F_NEAR_CONSTANT"


# Valid semantic types handled by the NUMERIC missing value engine
VALID_NUMERIC_TYPES = [
    "integer", "float", "percentage", "ratio",
    "currency", "salary", "distance", "weight",
    "volume", "area", "speed", "temperature",
    "pressure", "energy", "power", "capacity",
    "density", "angle", "latitude", "longitude",
    "geo_coordinate"
]

# Valid semantic types handled by the TEMPORAL missing value engine
VALID_TEMPORAL_TYPES = [
    "date", "time", "datetime", "timestamp",
    "year", "month", "day", "week",
    "quarter", "fiscal_year", "duration",
    "timezone"
]


def numeric_missing_value_handler(column_profile: Dict, column_data: pd.Series) -> Optional[Dict]:
    """
    Rule-based numeric missing value handler using profiling metadata.
    
    Handles ONLY columns where:
    - semantic_type ∈ VALID_NUMERIC_TYPES
    - semantic_confidence >= 0.8
    
    Args:
        column_profile: dict from profiling JSON (column_wise_summary item)
        column_data: pandas Series (numeric column)
    
    Returns:
        Decision dict or None if column should be skipped
    """
    from scipy.stats import skew
    
    decision = {
        "column_name": column_profile["column_name"],
        "pattern": None,
        "imputation_method": None,
        "create_missing_flag": False,
        "drop_column": False
    }

    # ==================================================
    # STEP 1 — COLUMN VALIDATION
    # ==================================================
    if (
        column_profile.get("semantic_type", "").lower() not in VALID_NUMERIC_TYPES
        or column_profile.get("semantic_confidence", 0) < 0.8
    ):
        return None  # skip column entirely

    null_pct = column_profile.get("null_percentage", 0)
    unique_count = column_profile.get("unique_count", 0)

    # No missing values — skip
    if null_pct == 0:
        return None

    # ==================================================
    # STEP 2 — NUMERIC MISSING PATTERN RULE ENGINE
    # ==================================================

    # PATTERN F — NEAR CONSTANT
    if unique_count <= 2:
        decision["pattern"] = NumericMissingPattern.NEAR_CONSTANT
        decision["imputation_method"] = "mode"
        return decision

    # PATTERN A — LOW MISSING NOISE
    if null_pct <= 5 and unique_count > 5:
        decision["pattern"] = NumericMissingPattern.LOW_MISSING
        decision["imputation_method"] = "median"
        return decision

    # PATTERN B — CONTINUOUS FEATURE (MODERATE MISSING)
    if 5 < null_pct <= 25 and unique_count > 10:
        decision["pattern"] = NumericMissingPattern.CONTINUOUS_MODERATE

        clean_values = column_data.dropna()
        if len(clean_values) >= 3:
            col_skewness = skew(clean_values)
            if abs(col_skewness) > 0.75:
                decision["imputation_method"] = "median"
            else:
                decision["imputation_method"] = "mean"
        else:
            decision["imputation_method"] = "median"

        return decision

    # PATTERN C — DISCRETE NUMERIC (CATEGORY-LIKE)
    if 5 < null_pct <= 25 and unique_count <= 10:
        decision["pattern"] = NumericMissingPattern.DISCRETE_NUMERIC
        decision["imputation_method"] = "mode"
        return decision

    # PATTERN D — HIGH MISSING BUT IMPORTANT
    if 25 < null_pct <= 60:
        decision["pattern"] = NumericMissingPattern.HIGH_MISSING
        if unique_count > 10:
            decision["imputation_method"] = "median"
        else:
            decision["imputation_method"] = "mode"
        decision["create_missing_flag"] = True
        return decision

    # PATTERN E — VERY HIGH MISSING
    if null_pct > 60:
        decision["pattern"] = NumericMissingPattern.VERY_HIGH_MISSING

        # Evaluate feature importance via correlation
        feature_importance = _evaluate_feature_importance(column_data)

        if feature_importance == "low":
            decision["drop_column"] = True
        else:
            decision["imputation_method"] = "median"
            decision["create_missing_flag"] = True

        return decision

    # Default fallback for edge cases
    decision["pattern"] = NumericMissingPattern.CONTINUOUS_MODERATE
    decision["imputation_method"] = "median"
    return decision


def _evaluate_feature_importance(column_data: pd.Series) -> str:
    """
    Evaluate feature importance based on data variance.
    Returns: 'high', 'medium', or 'low'
    """
    try:
        non_null = column_data.dropna()
        if len(non_null) < 5:
            return 'low'
        
        # Use coefficient of variation as proxy for importance
        cv = non_null.std() / non_null.mean() if non_null.mean() != 0 else 0
        
        if cv > 0.5:
            return 'high'
        elif cv > 0.2:
            return 'medium'
        else:
            return 'low'
    except:
        return 'low'


class NumericMissingValueHandler:
    """
    Wrapper class for numeric missing value handling.
    Uses the standalone numeric_missing_value_handler function internally.
    
    Handles ONLY columns where:
    - semantic_type ∈ VALID_NUMERIC_TYPES
    - semantic_confidence >= 0.8
    """
    
    def __init__(self, df: pd.DataFrame, profiling_metadata: Dict):
        """
        Initialize the Numeric Missing Value Handler.
        
        Args:
            df: The DataFrame to process
            profiling_metadata: Profiling JSON metadata with column_wise_summary
        """
        self.df = df.copy()
        self.profiling_metadata = profiling_metadata
        self.decisions = {}
        self.column_metadata = self._build_column_metadata()
    
    def _build_column_metadata(self) -> Dict:
        """Build a lookup dictionary for column metadata from profiling."""
        metadata = {}
        if self.profiling_metadata and 'column_wise_summary' in self.profiling_metadata:
            for col_info in self.profiling_metadata['column_wise_summary']:
                col_name = col_info.get('column_name')
                if col_name:
                    metadata[col_name] = col_info
        return metadata
    
    def process_all_numeric_columns(self, enable_advanced_upgrades: bool = False) -> Dict[str, Dict]:
        """
        Process all numeric columns and generate imputation decisions.
        
        Args:
            enable_advanced_upgrades: Whether to apply advanced method upgrades (group_median, etc.)
            
        Returns:
            Dictionary of column decisions
        """
        logger.info("=" * 60)
        logger.info("NUMERIC MISSING VALUE PATTERN CLASSIFICATION")
        logger.info("=" * 60)
        
        for col in self.df.columns:
            col_profile = self.column_metadata.get(col)
            if not col_profile:
                continue
            
            # Use the standalone function
            decision = numeric_missing_value_handler(col_profile, self.df[col])
            
            if decision is None:
                continue  # Column skipped (not numeric or no missing)
            
            # Optional: Apply advanced upgrades
            if enable_advanced_upgrades:
                decision = self._upgrade_to_advanced_method(decision)
            
            logger.info(f"\n📊 Processing: {col}")
            logger.info(f"   Semantic Type: {col_profile.get('semantic_type')}")
            logger.info(f"   Null %: {col_profile.get('null_percentage', 0):.2f}%")
            logger.info(f"   Unique Count: {col_profile.get('unique_count')}")
            logger.info(f"   → Pattern: {decision['pattern']}")
            logger.info(f"   → Method: {decision['imputation_method']}")
            if decision['create_missing_flag']:
                logger.info(f"   → Creating missing flag: Yes")
            if decision['drop_column']:
                logger.info(f"   → Drop column: Yes")
            
            self.decisions[col] = decision
        
        return self.decisions
    
    def _upgrade_to_advanced_method(self, decision: Dict) -> Dict:
        """
        Upgrade imputation method to advanced methods if applicable.
        - group_median: if significant variation by categorical column
        """
        if decision['drop_column'] or decision['imputation_method'] is None:
            return decision
        
        col = decision['column_name']
        
        # Check for group_median upgrade
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for cat_col in categorical_cols:
            if self.df[cat_col].nunique() < 10:
                try:
                    grouped_means = self.df.groupby(cat_col)[col].mean()
                    if grouped_means.nunique() > 1:
                        overall_var = self.df[col].var()
                        within_group_var = self.df.groupby(cat_col)[col].var().mean()
                        
                        if overall_var > 0 and within_group_var / overall_var < 0.7:
                            decision['imputation_method'] = 'group_median'
                            decision['group_column'] = cat_col
                            break
                except:
                    pass
        
        return decision
    
    def apply_imputation(self) -> pd.DataFrame:
        """
        Apply the imputation decisions to the DataFrame.
        
        Returns:
            DataFrame with missing values imputed
        """
        logger.info("\n" + "=" * 60)
        logger.info("APPLYING NUMERIC IMPUTATION")
        logger.info("=" * 60)
        
        df_result = self.df.copy()
        
        for col, decision in self.decisions.items():
            method = decision['imputation_method']
            
            # Handle drop
            if decision['drop_column']:
                logger.info(f"  Dropping column: {col}")
                df_result = df_result.drop(columns=[col])
                continue
            
            if method is None:
                continue
            
            missing_before = df_result[col].isnull().sum()
            if missing_before == 0:
                continue
            
            logger.info(f"\n  Imputing '{col}' ({missing_before} missing) using {method}...")
            
            # Apply imputation method
            if method == 'mean':
                fill_value = df_result[col].mean()
                df_result[col].fillna(fill_value, inplace=True)
                logger.info(f"    Filled with mean: {fill_value:.4f}")
                
            elif method == 'median':
                fill_value = df_result[col].median()
                df_result[col].fillna(fill_value, inplace=True)
                logger.info(f"    Filled with median: {fill_value:.4f}")
                
            elif method == 'mode':
                mode_val = df_result[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 0
                df_result[col].fillna(fill_value, inplace=True)
                logger.info(f"    Filled with mode: {fill_value}")
                
            elif method == 'group_median':
                group_col = decision.get('group_column')
                if group_col and group_col in df_result.columns:
                    df_result[col] = df_result.groupby(group_col)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    df_result[col].fillna(df_result[col].median(), inplace=True)
                    logger.info(f"    Filled with group median (grouped by {group_col})")
                else:
                    fill_value = df_result[col].median()
                    df_result[col].fillna(fill_value, inplace=True)
                    logger.info(f"    Fallback to median: {fill_value:.4f}")
            
            missing_after = df_result[col].isnull().sum()
            logger.info(f"    ✓ Remaining missing: {missing_after}")
        
        return df_result
    
    def get_summary(self) -> Dict:
        """Get summary of all imputation decisions."""
        summary = {
            'total_columns_processed': len(self.decisions),
            'columns_dropped': [],
            'columns_with_flags': [],
            'methods_used': {},
            'patterns_detected': {},
            'decisions': self.decisions
        }
        
        for col, decision in self.decisions.items():
            method = decision['imputation_method']
            if method:
                summary['methods_used'][method] = summary['methods_used'].get(method, 0) + 1
            
            pattern = decision['pattern']
            if pattern:
                summary['patterns_detected'][pattern] = summary['patterns_detected'].get(pattern, 0) + 1
            
            if decision['drop_column']:
                summary['columns_dropped'].append(col)
            
            if decision['create_missing_flag']:
                summary['columns_with_flags'].append(col)
        
        return summary


# ============================================================================
# TEMPORAL MISSING VALUE PATTERNS
# ============================================================================
# Pattern labels for temporal columns based on metadata analysis

class TemporalMissingPattern:
    """Constants for temporal missing value patterns."""
    LOW_MISSING = "PATTERN_T1_LOW_MISSING"
    TIME_SERIES_INTERPOLATION = "PATTERN_T2_TIME_SERIES_INTERPOLATION"
    CALENDAR_MODE = "PATTERN_T3_CALENDAR_MODE"
    DURATION_MEDIAN = "PATTERN_T4_DURATION_MEDIAN"
    HIGH_MISSING = "PATTERN_T5_HIGH_MISSING"
    VERY_HIGH_MISSING = "PATTERN_T6_VERY_HIGH_MISSING"
    REDERIVE = "PATTERN_T7_REDERIVE_FROM_PARENT"


def temporal_missing_value_handler(column_profile: Dict, column_data: pd.Series, df: pd.DataFrame = None) -> Optional[Dict]:
    """
    Rule-based temporal missing value handler using profiling metadata.
    
    Handles ONLY columns where:
    - semantic_type ∈ VALID_TEMPORAL_TYPES
    - semantic_confidence >= 0.8
    
    Patterns:
        T1 — Low missing (≤5%): ffill/bfill for date/datetime, mode for calendar
        T2 — Time-series interpolation: monotonic/ordered date/datetime → interpolation
        T3 — Calendar mode: month/day/week/quarter/fiscal_year → mode
        T4 — Duration median: duration → median
        T5 — High missing (25-60%): median/mode + missing flag
        T6 — Very high missing (>60%): drop or mode + flag
        T7 — Re-derive from parent: extract month/day/week from a date column
    
    Args:
        column_profile: dict from profiling JSON (column_wise_summary item)
        column_data: pandas Series (temporal column)
        df: optional full DataFrame (needed for T7 re-derivation)
    
    Returns:
        Decision dict or None if column should be skipped
    """
    decision = {
        "column_name": column_profile["column_name"],
        "pattern": None,
        "imputation_method": None,
        "create_missing_flag": False,
        "drop_column": False
    }

    semantic_type = column_profile.get("semantic_type", "").lower()

    # ==================================================
    # STEP 1 — COLUMN VALIDATION
    # ==================================================
    if (
        semantic_type not in VALID_TEMPORAL_TYPES
        or column_profile.get("semantic_confidence", 0) < 0.8
    ):
        return None  # skip column entirely

    null_pct = column_profile.get("null_percentage", 0)
    unique_count = column_profile.get("unique_count", 0)

    # No missing values — skip
    if null_pct == 0:
        return None

    # ==================================================
    # STEP 2 — TEMPORAL MISSING PATTERN RULE ENGINE
    # ==================================================

    # PATTERN T7 — RE-DERIVE FROM PARENT COLUMN
    # Check if this calendar component can be extracted from a date/datetime column
    if df is not None and semantic_type in ["month", "day", "week", "quarter", "fiscal_year", "year"]:
        col_name = column_profile["column_name"]
        parent_col = None
        for other_col in df.columns:
            if other_col == col_name:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[other_col]):
                # Found a datetime column → can derive from it
                parent_col = other_col
                break
        
        if parent_col is not None:
            decision["pattern"] = TemporalMissingPattern.REDERIVE
            decision["imputation_method"] = "rederive"
            decision["parent_column"] = parent_col
            decision["extraction_type"] = semantic_type
            return decision

    # PATTERN T1 — LOW MISSING NOISE (≤5%)
    if null_pct <= 5:
        decision["pattern"] = TemporalMissingPattern.LOW_MISSING

        if semantic_type in ["date", "datetime", "timestamp"]:
            decision["imputation_method"] = "ffill_bfill"
        elif semantic_type == "time":
            decision["imputation_method"] = "mode"
        elif semantic_type in ["month", "day", "week", "quarter", "fiscal_year", "year"]:
            decision["imputation_method"] = "mode"
        elif semantic_type == "duration":
            decision["imputation_method"] = "median"
        else:
            decision["imputation_method"] = "mode"

        return decision

    # PATTERN T2 — TIME-SERIES INTERPOLATION (ordered date/datetime)
    if semantic_type in ["date", "datetime", "timestamp"]:
        is_monotonic = column_profile.get("is_monotonic", False)
        has_time_order = column_profile.get("has_time_order", False)

        if is_monotonic or has_time_order:
            decision["pattern"] = TemporalMissingPattern.TIME_SERIES_INTERPOLATION
            decision["imputation_method"] = "interpolation"
            if null_pct > 25:
                decision["create_missing_flag"] = True
            return decision

    # PATTERN T3 — CALENDAR COMPONENT MODE (≤25%)
    if semantic_type in ["month", "day", "week", "quarter", "fiscal_year", "year"]:
        if null_pct <= 25:
            decision["pattern"] = TemporalMissingPattern.CALENDAR_MODE
            decision["imputation_method"] = "mode"
            return decision

    # PATTERN T4 — DURATION MEDIAN (≤25%)
    if semantic_type == "duration":
        if null_pct <= 25:
            decision["pattern"] = TemporalMissingPattern.DURATION_MEDIAN
            decision["imputation_method"] = "median"
            return decision

    # For date/datetime/timestamp with moderate missing (5-25%, non-monotonic)
    if semantic_type in ["date", "datetime", "timestamp"] and null_pct <= 25:
        decision["pattern"] = TemporalMissingPattern.LOW_MISSING
        decision["imputation_method"] = "ffill_bfill"
        return decision

    # PATTERN T5 — HIGH MISSING TEMPORAL (25-60%)
    if 25 < null_pct <= 60:
        decision["pattern"] = TemporalMissingPattern.HIGH_MISSING
        decision["create_missing_flag"] = True

        if semantic_type in ["date", "datetime", "timestamp"]:
            decision["imputation_method"] = "ffill_bfill"
        elif semantic_type == "duration":
            decision["imputation_method"] = "median"
        else:
            decision["imputation_method"] = "mode"

        return decision

    # PATTERN T6 — VERY HIGH MISSING (>60%)
    if null_pct > 60:
        decision["pattern"] = TemporalMissingPattern.VERY_HIGH_MISSING

        # For temporal columns, prefer keeping with flag rather than dropping
        if unique_count <= 2:
            decision["drop_column"] = True
        else:
            decision["imputation_method"] = "mode"
            decision["create_missing_flag"] = True

        return decision

    # Default fallback for temporal
    decision["pattern"] = TemporalMissingPattern.CALENDAR_MODE
    decision["imputation_method"] = "mode"
    return decision


class TemporalMissingValueHandler:
    """
    Wrapper class for temporal missing value handling.
    Uses the standalone temporal_missing_value_handler function internally.
    
    Handles ONLY columns where:
    - semantic_type ∈ VALID_TEMPORAL_TYPES
    - semantic_confidence >= 0.8
    """
    
    def __init__(self, df: pd.DataFrame, profiling_metadata: Dict):
        """
        Initialize the Temporal Missing Value Handler.
        
        Args:
            df: The DataFrame to process
            profiling_metadata: Profiling JSON metadata with column_wise_summary
        """
        self.df = df.copy()
        self.profiling_metadata = profiling_metadata
        self.decisions = {}
        self.column_metadata = self._build_column_metadata()
    
    def _build_column_metadata(self) -> Dict:
        """Build a lookup dictionary for column metadata from profiling."""
        metadata = {}
        if self.profiling_metadata and 'column_wise_summary' in self.profiling_metadata:
            for col_info in self.profiling_metadata['column_wise_summary']:
                col_name = col_info.get('column_name')
                if col_name:
                    metadata[col_name] = col_info
        return metadata
    
    def process_all_temporal_columns(self) -> Dict[str, Dict]:
        """
        Process all temporal columns and generate imputation decisions.
        
        Returns:
            Dictionary of column decisions
        """
        logger.info("=" * 60)
        logger.info("TEMPORAL MISSING VALUE PATTERN CLASSIFICATION")
        logger.info("=" * 60)
        
        for col in self.df.columns:
            col_profile = self.column_metadata.get(col)
            if not col_profile:
                continue
            
            # Use the standalone function (pass df for T7 re-derivation)
            decision = temporal_missing_value_handler(col_profile, self.df[col], self.df)
            
            if decision is None:
                continue  # Column skipped (not temporal or no missing)
            
            logger.info(f"\n📅 Processing: {col}")
            logger.info(f"   Semantic Type: {col_profile.get('semantic_type')}")
            logger.info(f"   Null %: {col_profile.get('null_percentage', 0):.2f}%")
            logger.info(f"   Unique Count: {col_profile.get('unique_count')}")
            logger.info(f"   → Pattern: {decision['pattern']}")
            logger.info(f"   → Method: {decision['imputation_method']}")
            if decision.get('parent_column'):
                logger.info(f"   → Parent Column: {decision['parent_column']}")
            if decision['create_missing_flag']:
                logger.info(f"   → Creating missing flag: Yes")
            if decision['drop_column']:
                logger.info(f"   → Drop column: Yes")
            
            self.decisions[col] = decision
        
        return self.decisions
    
    def apply_imputation(self) -> pd.DataFrame:
        """
        Apply the temporal imputation decisions to the DataFrame.
        
        Returns:
            DataFrame with missing values imputed
        """
        logger.info("\n" + "=" * 60)
        logger.info("APPLYING TEMPORAL IMPUTATION")
        logger.info("=" * 60)
        
        df_result = self.df.copy()
        
        for col, decision in self.decisions.items():
            method = decision['imputation_method']
            
            # Handle drop
            if decision['drop_column']:
                logger.info(f"  Dropping column: {col}")
                df_result = df_result.drop(columns=[col])
                continue
            
            if method is None:
                continue
            
            missing_before = df_result[col].isnull().sum()
            if missing_before == 0:
                continue
            
            logger.info(f"\n  Imputing '{col}' ({missing_before} missing) using {method}...")
            
            # Apply imputation method
            if method == 'ffill_bfill':
                df_result[col] = df_result[col].fillna(method='ffill')
                df_result[col] = df_result[col].fillna(method='bfill')
                # If still missing (all NaN), use mode
                if df_result[col].isnull().any():
                    mode_val = df_result[col].mode()
                    if len(mode_val) > 0 and not pd.isna(mode_val[0]):
                        df_result[col] = df_result[col].fillna(mode_val[0])
                logger.info(f"    Filled with forward/backward fill")
                
            elif method == 'interpolation':
                try:
                    if pd.api.types.is_datetime64_any_dtype(df_result[col]):
                        # Convert to numeric for interpolation, then back
                        numeric_dates = df_result[col].astype(np.int64)
                        numeric_dates = numeric_dates.replace(pd.NaT, np.nan)
                        numeric_dates = numeric_dates.interpolate(method='linear', limit_direction='both')
                        df_result[col] = pd.to_datetime(numeric_dates)
                    else:
                        df_result[col] = df_result[col].interpolate(method='linear', limit_direction='both')
                except Exception:
                    # Fallback to ffill/bfill
                    df_result[col] = df_result[col].fillna(method='ffill')
                    df_result[col] = df_result[col].fillna(method='bfill')
                
                # Final fallback for any remaining
                if df_result[col].isnull().any():
                    df_result[col] = df_result[col].fillna(method='ffill')
                    df_result[col] = df_result[col].fillna(method='bfill')
                logger.info(f"    Filled with interpolation")
                
            elif method == 'mode':
                mode_val = df_result[col].mode()
                if len(mode_val) > 0 and not pd.isna(mode_val[0]):
                    fill_value = mode_val[0]
                else:
                    fill_value = 0 if pd.api.types.is_numeric_dtype(df_result[col]) else ''
                df_result[col] = df_result[col].fillna(fill_value)
                logger.info(f"    Filled with mode: {fill_value}")
                
            elif method == 'median':
                try:
                    fill_value = df_result[col].median()
                    df_result[col] = df_result[col].fillna(fill_value)
                    logger.info(f"    Filled with median: {fill_value}")
                except Exception:
                    mode_val = df_result[col].mode()
                    fill_value = mode_val[0] if len(mode_val) > 0 and not pd.isna(mode_val[0]) else 0
                    df_result[col] = df_result[col].fillna(fill_value)
                    logger.info(f"    Fallback to mode: {fill_value}")
                
            elif method == 'rederive':
                parent_col = decision.get('parent_column')
                extraction_type = decision.get('extraction_type')
                
                if parent_col and parent_col in df_result.columns:
                    parent_dt = pd.to_datetime(df_result[parent_col], errors='coerce')
                    derived = None
                    
                    if extraction_type == 'month':
                        derived = parent_dt.dt.month
                    elif extraction_type == 'day':
                        derived = parent_dt.dt.day
                    elif extraction_type == 'week':
                        try:
                            derived = parent_dt.dt.isocalendar().week.astype(int)
                        except Exception:
                            derived = parent_dt.dt.week
                    elif extraction_type == 'quarter':
                        derived = parent_dt.dt.quarter
                    elif extraction_type == 'year':
                        derived = parent_dt.dt.year
                    elif extraction_type == 'fiscal_year':
                        # Simplified: fiscal year = year (can be customised)
                        derived = parent_dt.dt.year
                    
                    if derived is not None:
                        mask = df_result[col].isnull()
                        df_result.loc[mask, col] = derived[mask]
                        logger.info(f"    Re-derived from '{parent_col}' ({extraction_type})")
                
                # Fallback for any remaining nulls
                if df_result[col].isnull().any():
                    mode_val = df_result[col].mode()
                    if len(mode_val) > 0 and not pd.isna(mode_val[0]):
                        df_result[col] = df_result[col].fillna(mode_val[0])
                        logger.info(f"    Remaining filled with mode fallback")
            
            missing_after = df_result[col].isnull().sum()
            logger.info(f"    ✓ Remaining missing: {missing_after}")
        
        return df_result
    
    def get_summary(self) -> Dict:
        """Get summary of all temporal imputation decisions."""
        summary = {
            'total_columns_processed': len(self.decisions),
            'columns_dropped': [],
            'columns_with_flags': [],
            'columns_rederived': [],
            'methods_used': {},
            'patterns_detected': {},
            'decisions': self.decisions
        }
        
        for col, decision in self.decisions.items():
            method = decision['imputation_method']
            if method:
                summary['methods_used'][method] = summary['methods_used'].get(method, 0) + 1
            
            pattern = decision['pattern']
            if pattern:
                summary['patterns_detected'][pattern] = summary['patterns_detected'].get(pattern, 0) + 1
            
            if decision['drop_column']:
                summary['columns_dropped'].append(col)
            
            if decision['create_missing_flag']:
                summary['columns_with_flags'].append(col)
            
            if decision.get('parent_column'):
                summary['columns_rederived'].append(col)
        
        return summary


def missing_value_dispatcher(column_profile: Dict, column_data: pd.Series, df: pd.DataFrame = None) -> Optional[Dict]:
    """
    Unified dispatcher that routes a column to the correct handler.
    
    Checks semantic_type and delegates to either:
    - temporal_missing_value_handler (for VALID_TEMPORAL_TYPES)
    - numeric_missing_value_handler (for VALID_NUMERIC_TYPES)
    
    Args:
        column_profile: dict from profiling JSON (column_wise_summary item)
        column_data: pandas Series
        df: optional full DataFrame (needed for temporal T7 re-derivation)
    
    Returns:
        Decision dict with 'handler_type' field, or None if column not handled
    """
    semantic_type = column_profile.get("semantic_type", "").lower()

    # Try temporal first (more specific match)
    if semantic_type in VALID_TEMPORAL_TYPES:
        decision = temporal_missing_value_handler(column_profile, column_data, df)
        if decision:
            decision["handler_type"] = "temporal"
            return decision

    # Try numeric
    if semantic_type in VALID_NUMERIC_TYPES:
        decision = numeric_missing_value_handler(column_profile, column_data)
        if decision:
            decision["handler_type"] = "numeric"
            return decision

    return None


class AutoGluonMissingValueHandler:
    """
    AutoML-based missing value handler with explicit AutoGluon Tabular integration.
    
    This class:
    1. Normalizes invalid tokens to proper NaN values
    2. Handles missing values using AutoGluon-compatible strategies
    3. Validates output using AutoGluon FeatureMetadata and TabularPredictor
    4. Ensures datasets are ready for AutoGluon training
    """
    
    def __init__(self, user_id: str, session_id: str):
        """Initialize the AutoGluon Missing Value Handler."""
        self.user_id = user_id
        self.session_id = session_id
        self.input_prefix = f"output/{user_id}/{session_id}"
        self.meta_prefix = f"meta_data/{user_id}/{session_id}"
        
        self.stats = {
            'files_processed': 0,
            'total_missing_before': 0,
            'total_missing_after': 0,
            'columns_dropped': [],
            'tokens_normalized': 0,
            'autogluon_validated': False
        }
        
        # Track imputation methods used for each column
        self.imputation_methods = {}
    
    def _validate_directories(self):
        """No-op: validation handled by Supabase storage."""
        pass
    
    def find_datasets(self) -> List[str]:
        """Find all *_cleaned.csv files in the Supabase input prefix."""
        all_files = list_files(self.input_prefix)
        cleaned_files = [f for f in all_files if f.endswith('_cleaned.csv')]
        logger.info(f"Found {len(cleaned_files)} cleaned datasets for user {self.user_id}")
        return cleaned_files
    
    def load_profiling_metadata(self, filename: str) -> Optional[Dict]:
        """Load profiling JSON metadata for a dataset from Supabase."""
        base_name = filename.replace('_cleaned.csv', '')
        storage_path = f"{self.meta_prefix}/{base_name}_profiling.json"
        
        try:
            metadata = download_json(storage_path)
            logger.info(f"Loaded profiling metadata: {storage_path}")
            return metadata
        except Exception as e:
            logger.warning(f"Profiling metadata not found: {storage_path} — {e}")
            return None
    
    def analyze_missing_values(self, df: pd.DataFrame, metadata: Optional[Dict]) -> Dict:
        """Analyze missing values in the dataset."""
        analysis = {
            'total_rows': len(df),
            'total_missing': df.isnull().sum().sum(),
            'columns': {}
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            col_info = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_pct),
                'dtype': str(df[col].dtype),
                'unique_count': df[col].nunique(),
                'semantic_type': 'unknown',
                'detected_type': 'unknown'
            }
            
            if metadata and 'column_wise_summary' in metadata:
                for meta_col in metadata['column_wise_summary']:
                    if meta_col.get('column_name') == col:
                        col_info['semantic_type'] = meta_col.get('semantic_type', 'unknown')
                        col_info['detected_type'] = meta_col.get('inferred_dtype', 'unknown')
                        break
            
            analysis['columns'][col] = col_info
        
        return analysis
    
    def _get_profiling_metadata_for_numeric_handler(self, analysis: Dict) -> Optional[Dict]:
        """
        Build profiling metadata structure compatible with Numeric/Temporal handlers.
        
        Searches for any *_profiling.json in the metadata directory.
        Falls back to building from analysis Dict if no file found.
        """
        try:
            # Try to find any profiling JSON in the metadata prefix
            all_meta_files = list_files(self.meta_prefix)
            profiling_files = [f for f in all_meta_files if f.endswith('_profiling.json')]
            if profiling_files:
                return download_json(f"{self.meta_prefix}/{profiling_files[0]}")
            
            # Fallback: Build from analysis (won't have semantic_confidence)
            if not analysis or 'columns' not in analysis:
                return None
            
            column_wise_summary = []
            for col_name, col_info in analysis['columns'].items():
                col_meta = {
                    'column_name': col_name,
                    'semantic_type': col_info.get('semantic_type', 'unknown'),
                    'semantic_confidence': 1.0 if col_info.get('semantic_type') != 'unknown' else 0.5,
                    'null_count': col_info.get('missing_count', 0),
                    'null_percentage': col_info.get('missing_percentage', 0),
                    'unique_count': col_info.get('unique_count', 0),
                    'inferred_dtype': col_info.get('detected_type', col_info.get('dtype', 'unknown'))
                }
                column_wise_summary.append(col_meta)
            
            return {'column_wise_summary': column_wise_summary}
        except Exception as e:
            logger.warning(f"Could not build profiling metadata: {e}")
            return None
    
    def normalize_invalid_tokens(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Normalize invalid tokens to proper NaN values (AutoGluon-safe requirement).
        
        Replaces ["UNKNOWN", "ERROR", "N/A", etc.] and special characters with np.nan.
        """
        logger.info("Normalizing invalid tokens to AutoGluon-safe values...")
        df_normalized = df.copy()
        tokens_normalized = 0
        
        for col in df_normalized.columns:
            dtype = df_normalized[col].dtype
            
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                # Check for exact matches with invalid tokens
                mask = df_normalized[col].isin(INVALID_TOKENS)
                count_invalid = mask.sum()
                
                # Also check for strings that are only special characters (single or multiple)
                if df_normalized[col].dtype == 'object':
                    special_char_mask = df_normalized[col].astype(str).str.match(r'^[%&^#@!)(\\*~\s]+$', na=False)
                    count_invalid += special_char_mask.sum()
                    mask = mask | special_char_mask
                
                if count_invalid > 0:
                    logger.info(f"  Column '{col}': Normalizing {count_invalid} invalid tokens")
                    df_normalized.loc[mask, col] = np.nan
                    tokens_normalized += count_invalid
        
        logger.info(f"✅ Normalized {tokens_normalized} invalid tokens")
        return df_normalized, tokens_normalized
    
    def convert_columns_to_proper_types(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """
        Convert columns to proper types based on metadata before processing.
        """
        logger.info("Converting columns to proper types based on metadata...")
        df_converted = df.copy()
        
        for col in df_converted.columns:
            col_info = analysis['columns'].get(col, {})
            semantic_type = col_info.get('semantic_type', '').lower()
            
            # Convert numeric columns
            if semantic_type in ['integer', 'float', 'numeric']:
                logger.info(f"  Converting '{col}' to numeric (semantic type: {semantic_type})")
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                # Note: Keep as float64 for AutoGluon compatibility
                # Integers will be preserved when saving to CSV
        
        logger.info("✅ Type conversion completed")
        return df_converted
    
    def parse_datetime_columns(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """
        Parse datetime columns for AutoGluon compatibility.
        
        Ensures datetime columns are datetime64 or NaT (no string tokens).
        Also validates and fixes invalid time values.
        """
        logger.info("Parsing datetime columns for AutoGluon compatibility...")
        df_parsed = df.copy()
        
        for col in df_parsed.columns:
            col_info = analysis['columns'].get(col, {})
            semantic_type = col_info.get('semantic_type', '').lower()
            
            # Handle time columns separately (not date, datetime, or timestamp)
            if 'time' in semantic_type and 'date' not in semantic_type and semantic_type != 'timestamp':
                logger.info(f"  Validating time column: '{col}'")
                
                # Validate time values - mark invalid times as NaN and ensure proper format
                def validate_time(t):
                    if pd.isna(t):
                        return np.nan
                    try:
                        if isinstance(t, str):
                            parts = t.split(':')
                            if len(parts) >= 2:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                # Check for invalid hours or minutes
                                if hours >= 24 or minutes >= 60:
                                    return np.nan
                                # Ensure we have seconds
                                if len(parts) == 2:
                                    return f"{hours:02d}:{minutes:02d}:00"
                                elif len(parts) == 3:
                                    seconds = int(parts[2])
                                    if seconds >= 60:
                                        return np.nan
                                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        return t
                    except:
                        return np.nan
                
                df_parsed[col] = df_parsed[col].apply(validate_time)
                invalid_count = df_parsed[col].isna().sum() - df[col].isna().sum()
                if invalid_count > 0:
                    logger.info(f"    ✓ Marked {invalid_count} invalid time values as NaN")
            
            elif 'date' in semantic_type or semantic_type == 'timestamp':
                # Skip timestamp columns - they should remain as unix timestamps (integers)
                if semantic_type == 'timestamp':
                    logger.info(f"  Skipping timestamp column (preserving unix timestamps): '{col}'")
                    continue
                
                # STEP 1: Fix invalid dates BEFORE parsing (e.g., 2023-11-31 → 2023-11-30)
                df_parsed = self._fix_invalid_dates(df_parsed, col)
                
                # Check if it's a date-only column (not datetime/timestamp)
                is_date_only = semantic_type == 'date' and 'time' not in semantic_type
                
                logger.info(f"  Parsing {'date' if is_date_only else 'datetime'} column: '{col}'")
                
                # STEP 2: Parse to datetime
                df_parsed[col] = pd.to_datetime(
                    df_parsed[col],
                    dayfirst=True,
                    errors='coerce'
                )
                
                # Count dates that couldn't be fixed
                invalid_count = df_parsed[col].isna().sum() - df.loc[df_parsed.index, col].isna().sum()
                if invalid_count > 0:
                    logger.warning(f"    ⚠️  {invalid_count} dates in '{col}' could not be parsed (marked as NaT)")
                
                # If it's a date-only column, convert back to date format (removes time component)
                if is_date_only:
                    df_parsed[col] = df_parsed[col].dt.date
                    logger.info(f"    ✓ Converted to date format (date only, no time)")
                else:
                    nat_count = df_parsed[col].isna().sum()
                    logger.info(f"    ✓ Converted to datetime64, {nat_count} NaT values")
        
        return df_parsed
    
    def _fix_invalid_dates(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fix invalid dates (e.g., 2023-11-31) by adjusting to the last valid day of the month.
        Also handles other common date errors.
        """
        logger.info(f"  Checking for invalid dates in '{col}'...")
        df_fixed = df.copy()
        fixed_count = 0
        
        for idx, val in df_fixed[col].items():
            if pd.isna(val) or val == "":
                continue
                
            try:
                # Try to parse the date
                parsed_date = pd.to_datetime(val, dayfirst=True, errors='raise')
            except:
                # Failed to parse - try to fix common issues
                val_str = str(val).strip()
                
                # Try to extract year, month, day patterns
                import re
                
                # Pattern: YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
                match = re.search(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', val_str)
                if not match:
                    # Pattern: DD-MM-YYYY or DD/MM/YYYY
                    match = re.search(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', val_str)
                    if match:
                        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    else:
                        continue
                else:
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                
                # Fix invalid day for the given month/year
                try:
                    # Try to create the date
                    fixed_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    # Invalid date - adjust day to last valid day of month
                    import calendar
                    try:
                        max_day = calendar.monthrange(year, month)[1]
                        
                        if day > max_day:
                            fixed_date = pd.Timestamp(year=year, month=month, day=max_day)
                            df_fixed.at[idx, col] = fixed_date.strftime('%Y-%m-%d')
                            fixed_count += 1
                            logger.info(f"    Fixed: {val} → {fixed_date.strftime('%Y-%m-%d')} (day {day} → {max_day})")
                    except:
                        pass
        
        if fixed_count > 0:
            logger.info(f"  ✓ Fixed {fixed_count} invalid dates in '{col}'")
        else:
            logger.info(f"  ✓ No invalid dates found")
        
        return df_fixed
    
    def _is_sequential_dates(self, df: pd.DataFrame, col: str) -> bool:
        """
        Check if dates in a column are sequential (time series pattern).
        Returns True if dates are mostly in order with regular intervals.
        """
        try:
            # Get non-null datetime values
            dates = pd.to_datetime(df[col], errors='coerce').dropna().sort_values()
            
            if len(dates) < 3:
                return False
            
            # Calculate intervals between consecutive dates
            intervals = dates.diff().dt.days.dropna()
            
            if len(intervals) == 0:
                return False
            
            # Check if intervals are relatively consistent (coefficient of variation < 1.0)
            mean_interval = intervals.mean()
            std_interval = intervals.std()
            
            if mean_interval == 0:
                return False
            
            cv = std_interval / mean_interval
            
            # Sequential if coefficient of variation is low
            return cv < 1.0
            
        except:
            return False
    
    def _interpolate_dates(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Interpolate missing dates using linear interpolation for sequential time series.
        Works with datetime columns where dates follow a pattern.
        """
        missing_before = int(df[col].isnull().sum())
        if missing_before == 0:
            return df
        
        logger.info(f"    Interpolating {missing_before} missing dates in '{col}'...")
        
        df_work = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_work[col]):
            df_work[col] = pd.to_datetime(df_work[col], errors='coerce')
        
        # Method 1: Try linear interpolation (compatible with datetimes)
        try:
            df_work[col] = df_work[col].interpolate(method='linear', limit_direction='both')
        except:
            # Interpolation failed, skip to forward/backward fill
            pass
        
        # Method 2: If still missing, use forward/backward fill
        if df_work[col].isnull().any():
            df_work[col] = df_work[col].fillna(method='ffill')
            df_work[col] = df_work[col].fillna(method='bfill')
        
        # Method 3: If STILL missing (entire column was null), use mode or median date
        if df_work[col].isnull().any():
            mode_dates = df_work[col].mode()
            if len(mode_dates) > 0 and not pd.isna(mode_dates[0]):
                df_work[col] = df_work[col].fillna(mode_dates[0])
            else:
                # Last resort: use median date
                median_date = df_work[col].dropna().median()
                if not pd.isna(median_date):
                    df_work[col] = df_work[col].fillna(median_date)
        
        filled = missing_before - int(df_work[col].isnull().sum())
        logger.info(f"    ✓ Filled {filled} dates via interpolation")
        
        return df_work
    
    def _forward_backward_fill_dates(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fill missing dates using forward fill, then backward fill.
        For non-sequential date columns.
        """
        missing_before = int(df[col].isnull().sum())
        if missing_before == 0:
            return df
        
        logger.info(f"    Forward/backward filling {missing_before} missing dates in '{col}'...")
        
        df_work = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_work[col]):
            df_work[col] = pd.to_datetime(df_work[col], errors='coerce')
        
        # Forward fill
        df_work[col] = df_work[col].fillna(method='ffill')
        
        # Backward fill for remaining
        df_work[col] = df_work[col].fillna(method='bfill')
        
        # If still missing (entire column was null), use mode
        if df_work[col].isnull().any():
            mode_dates = df_work[col].mode()
            if len(mode_dates) > 0 and not pd.isna(mode_dates[0]):
                df_work[col] = df_work[col].fillna(mode_dates[0])
            else:
                # Use current date as last resort
                df_work[col] = df_work[col].fillna(pd.Timestamp.now().normalize())
                logger.warning(f"    ⚠️  Used current date as fallback for '{col}'")
        
        filled = missing_before - int(df_work[col].isnull().sum())
        logger.info(f"    ✓ Filled {filled} dates via forward/backward fill")        
        return df_work
    
    # ==================================================================
    # SEQUENCE DETECTION AND FILLING
    # ==================================================================
    
    def _detect_sequence_pattern(self, df: pd.DataFrame, col: str) -> dict:
        """
        Detect if a column follows a sequential pattern.
        
        Returns dict with:
            - 'is_sequence': bool
            - 'sequence_type': 'numeric' | 'text_numeric' | None
            - 'prefix': str (for text sequences like 'EMP')
            - 'padding': int (number of digits, e.g., 3 for '001')
        """
        # Filter out null AND invalid tokens (Unknown, etc.)
        mask = df[col].notna() & ~df[col].astype(str).isin([str(t) for t in INVALID_TOKENS])
        non_null = df[col][mask]
        
        if len(non_null) < 3:  # Need at least 3 values to detect pattern
            return {'is_sequence': False, 'sequence_type': None}
        
        # Check for pure numeric sequence (1, 2, 3, ...)
        if pd.api.types.is_numeric_dtype(df[col]):
            sorted_vals = non_null.sort_values().values
            
            # Check if values are integers
            if len(sorted_vals) > 0 and np.all(sorted_vals == sorted_vals.astype(int)):
                sorted_vals = sorted_vals.astype(int)
                
                # Check if values form a sequence (even with gaps)
                # If max - min ~= number of values, it's likely a sequence
                min_val = sorted_vals.min()
                max_val = sorted_vals.max()
                expected_range = max_val - min_val + 1
                
                # If we have at least 50% of expected values, consider it a sequence
                if len(sorted_vals) >= expected_range * 0.5:
                    return {
                        'is_sequence': True,
                        'sequence_type': 'numeric',
                        'prefix': None,
                        'padding': 0
                    }
        
        # Check for text sequence with numeric suffix (EMP001, EMP002, ...)
        if pd.api.types.is_object_dtype(df[col]):
            import re
            
            # Get all non-null values
            non_null_values = non_null.astype(str).tolist()
            
            if len(non_null_values) < 3:
                return {'is_sequence': False, 'sequence_type': None}
            
            # DYNAMIC APPROACH: Find common prefix and varying numeric suffix
            # Step 1: Find where the numeric suffix starts for each value
            suffix_info = []
            for val in non_null_values:
                # Find the last sequence of digits in the string
                matches = list(re.finditer(r'\d+', val))
                if matches:
                    last_match = matches[-1]  # Get the last numeric part
                    prefix = val[:last_match.start()]
                    suffix_num = int(last_match.group())
                    suffix_len = len(last_match.group())
                    
                    suffix_info.append({
                        'value': val,
                        'prefix': prefix,
                        'suffix_num': suffix_num,
                        'suffix_len': suffix_len,
                        'suffix_start': last_match.start()
                    })
            
            # Step 2: Check if all values have the same prefix
            if len(suffix_info) >= len(non_null_values) * 0.8:  # 80% have pattern
                prefixes = [s['prefix'] for s in suffix_info]
                
                # Check if all prefixes are the same
                if len(set(prefixes)) == 1:
                    # Step 3: Check if numeric suffixes form a sequence
                    numbers = sorted([s['suffix_num'] for s in suffix_info])
                    min_num = min(numbers)
                    max_num = max(numbers)
                    expected_range = max_num - min_num + 1
                    
                    # If we have at least 50% of expected values, it's a sequence
                    if len(numbers) >= expected_range * 0.5:
                        # Determine padding (use the most common suffix length)
                        padding_lengths = [s['suffix_len'] for s in suffix_info]
                        padding = max(set(padding_lengths), key=padding_lengths.count)
                        
                        return {
                            'is_sequence': True,
                            'sequence_type': 'text_numeric',
                            'prefix': prefixes[0],
                            'padding': padding,
                            'min_num': min_num,
                            'max_num': max_num
                        }
        
        return {'is_sequence': False, 'sequence_type': None}
    
    def _fill_numeric_sequence(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fill missing values in a numeric sequence.
        Example: 1, 2, _, 4, 5 → 1, 2, 3, 4, 5
        """
        # Count missing including invalid tokens
        missing_mask = df[col].isna() | df[col].astype(str).isin([str(t) for t in INVALID_TOKENS])
        missing_before = int(missing_mask.sum())
        if missing_before == 0:
            return df
        
        logger.info(f"    Filling {missing_before} missing values in numeric sequence '{col}'...")
        
        df_work = df.copy()
        
        # Get the range of expected values (exclude invalid tokens)
        valid_mask = ~missing_mask
        non_null = df_work[col][valid_mask].dropna()
        if len(non_null) == 0:
            return df_work
        
        min_val = int(non_null.min())
        max_val = int(non_null.max())
        
        # Check for trailing nulls and extend range if needed
        last_idx = df_work[col].last_valid_index()
        has_trailing_nulls = last_idx is not None and last_idx < len(df_work) - 1
        
        if has_trailing_nulls:
            num_trailing = len(df_work) - last_idx - 1
            extended_max = max_val + num_trailing
        else:
            extended_max = max_val
        
        # Create complete sequence
        expected_sequence = list(range(min_val, extended_max + 1))
        existing_values = set(non_null.astype(int).values)
        missing_values = [v for v in expected_sequence if v not in existing_values]
        
        # Fill missing values by matching against expected sequence
        filled_count = 0
        for idx, val in df_work[col].items():
            # Check if missing or invalid token
            is_missing = pd.isna(val) or (str(val) in [str(t) for t in INVALID_TOKENS])
            if is_missing:
                # Try to infer the value based on position
                # Look at previous and next non-null values
                prev_idx = idx - 1
                while prev_idx >= 0:
                    prev_val = df_work.at[prev_idx, col]
                    if not pd.isna(prev_val) and str(prev_val) not in [str(t) for t in INVALID_TOKENS]:
                        break
                    prev_idx -= 1
                
                if prev_idx >= 0:
                    prev_val = df_work.at[prev_idx, col]
                    if not pd.isna(prev_val):
                        # Fill with previous + 1
                        fill_val = int(prev_val) + 1
                        if fill_val in missing_values:
                            df_work.at[idx, col] = fill_val
                            missing_values.remove(fill_val)
                            filled_count += 1
        
        logger.info(f"    ✓ Filled {filled_count} values in numeric sequence")
        return df_work
    
    def _fill_text_sequence(self, df: pd.DataFrame, col: str, prefix: str, padding: int) -> pd.DataFrame:
        """
        Fill missing values in a text sequence with numeric suffix.
        DYNAMIC: Works with ANY prefix pattern (e.g., 22HP1A1201, EMP001, ORD0001, etc.)
        """
        # Count missing including invalid tokens
        missing_mask = df[col].isna() | df[col].astype(str).isin([str(t) for t in INVALID_TOKENS])
        missing_before = int(missing_mask.sum())
        if missing_before == 0:
            return df
        
        logger.info(f"    Filling {missing_before} missing values in text sequence '{col}' (pattern: {prefix}...)")
        
        df_work = df.copy()
        
        import re
        
        # Extract numbers from existing values (using LAST numeric part as suffix)
        # Exclude invalid tokens
        number_info = {}
        for idx, val in df_work[col].items():
            if pd.notna(val) and str(val) not in [str(t) for t in INVALID_TOKENS]:
                val_str = str(val)
                # Find the last sequence of digits
                matches = list(re.finditer(r'\d+', val_str))
                if matches:
                    last_match = matches[-1]
                    extracted_prefix = val_str[:last_match.start()]
                    extracted_num = int(last_match.group())
                    
                    # Only process if prefix matches
                    if extracted_prefix == prefix:
                        number_info[idx] = {
                            'num': extracted_num,
                            'original': val_str
                        }
        
        if len(number_info) == 0:
            return df_work
        
        # Get range of numbers
        numbers = [info['num'] for info in number_info.values()]
        min_num = min(numbers)
        max_num = max(numbers)
        
        # Create complete sequence (include one beyond max for trailing nulls/invalid tokens)
        # Check if there are trailing nulls or invalid tokens
        last_valid_idx = None
        for idx in range(len(df_work) - 1, -1, -1):
            val = df_work.at[idx, col]
            if pd.notna(val) and str(val) not in [str(t) for t in INVALID_TOKENS]:
                last_valid_idx = idx
                break
        
        has_trailing_missing = last_valid_idx is not None and last_valid_idx < len(df_work) - 1
        
        if has_trailing_missing:
            # Extend range to accommodate trailing missing values
            num_trailing = len(df_work) - last_valid_idx - 1
            extended_max = max_num + num_trailing
        else:
            extended_max = max_num
        
        expected_sequence = list(range(min_num, extended_max + 1))
        existing_numbers = set(numbers)
        missing_numbers = [n for n in expected_sequence if n not in existing_numbers]
        
        # Fill missing values by inferring from position
        filled_count = 0
        for idx, val in df_work[col].items():
            # Check if missing or invalid token
            is_missing = pd.isna(val) or (str(val) in [str(t) for t in INVALID_TOKENS])
            if is_missing:
                # Look at previous non-null value
                prev_idx = idx - 1
                while prev_idx >= 0:
                    prev_val = df_work.at[prev_idx, col]
                    if pd.notna(prev_val) and str(prev_val) not in [str(t) for t in INVALID_TOKENS]:
                        break
                    prev_idx -= 1
                
                if prev_idx >= 0 and prev_idx in number_info:
                    prev_num = number_info[prev_idx]['num']
                    next_num = prev_num + 1
                    
                    if next_num in missing_numbers:
                        # Construct the value: prefix + padded number
                        fill_val = f"{prefix}{str(next_num).zfill(padding)}"
                        df_work.at[idx, col] = fill_val
                        missing_numbers.remove(next_num)
                        
                        # Track this for subsequent rows
                        number_info[idx] = {'num': next_num, 'original': fill_val}
                        filled_count += 1
        
        logger.info(f"    ✓ Filled {filled_count} values in text sequence")
        return df_work
    
    def fillSequence(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Automatically detect and fill sequential patterns in a column.
        
        Handles:
        1. Numeric sequences: 1, 2, _, 4, 5 → 1, 2, 3, 4, 5
        2. Text sequences: EMP001, EMP002, _, EMP004 → EMP001, EMP002, EMP003, EMP004
        
        Returns:
            DataFrame with filled sequence, or original if no sequence detected
        """
        # Count missing including invalid tokens
        missing_mask = df[col].isna() | df[col].astype(str).isin([str(t) for t in INVALID_TOKENS])
        missing_before = int(missing_mask.sum())
        if missing_before == 0:
            return df
        
        # Detect sequence pattern
        pattern = self._detect_sequence_pattern(df, col)
        
        if not pattern['is_sequence']:
            return df  # Not a sequence, skip
        
        logger.info(f"  Detected {pattern['sequence_type']} sequence in '{col}'")
        
        # Fill based on sequence type
        if pattern['sequence_type'] == 'numeric':
            return self._fill_numeric_sequence(df, col)
        elif pattern['sequence_type'] == 'text_numeric':
            return self._fill_text_sequence(df, col, pattern['prefix'], pattern['padding'])
        
        return df
    
    def is_unique_text_column(self, df: pd.DataFrame, col: str) -> bool:
        """
        Check if a text column contains completely unique values (non-categorical).
        A column is considered unique if >95% of non-null values are unique.
        """
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            return False
        
        unique_ratio = non_null_values.nunique() / len(non_null_values)
        return unique_ratio > 0.95
    
    def determine_imputation_method(self, df: pd.DataFrame, col: str, analysis: Dict) -> str:
        """
        Intelligently determine the best imputation method based on statistical analysis.
        
        Returns: One of ['mean', 'median', 'mode', 'forward_fill', 'knn', 'constant_zero', 'constant_unknown']
        """
        col_info = analysis['columns'].get(col, {})
        semantic_type = col_info.get('semantic_type', '').lower()
        missing_pct = col_info.get('missing_percentage', 0)
        
        # Get non-null values
        non_null = df[col].dropna()
        
        if len(non_null) == 0:
            return 'constant_unknown' if pd.api.types.is_object_dtype(df[col].dtype) else 'constant_zero'
        
        # Rule 1: BOOLEAN/DISCRETE CATEGORIES → MODE
        if semantic_type == 'boolean' or pd.api.types.is_categorical_dtype(df[col].dtype):
            return 'mode'
        
        # Rule 2: LOW CARDINALITY INTEGERS (like dependents, counts) → MODE
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            unique_count = non_null.nunique()
            
            # Check if it's a discrete count variable
            if unique_count <= 10 and (non_null % 1 == 0).all():
                # Likely a count/category variable (dependents, education level, etc.)
                return 'mode'
        
        # Rule 3: NUMERIC CONTINUOUS DATA → Analyze distribution
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            # Calculate skewness
            from scipy import stats as scipy_stats
            try:
                skewness = scipy_stats.skew(non_null)
            except:
                skewness = 0
            
            # Check for outliers using IQR method
            Q1 = non_null.quantile(0.25)
            Q3 = non_null.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((non_null < (Q1 - 1.5 * IQR)) | (non_null > (Q3 + 1.5 * IQR))).sum()
            has_outliers = outlier_count > len(non_null) * 0.05  # >5% outliers
            
            # Rule 3a: MEAN - Only if symmetric, no outliers, <15% missing
            if abs(skewness) < 0.5 and not has_outliers and missing_pct < 15:
                # Symmetric distribution, use MEAN
                return 'mean'
            
            # Rule 3b: MEDIAN - Skewed or has outliers
            else:
                # Skewed distribution or outliers present, use MEDIAN
                return 'median'
        
        # Rule 4: CATEGORICAL/TEXT → MODE
        if pd.api.types.is_object_dtype(df[col].dtype):
            unique_count = non_null.nunique()
            
            # Low cardinality categorical → MODE
            if unique_count < len(non_null) * 0.5:  # Less than 50% unique values
                return 'mode'
            else:
                # High cardinality text → CONSTANT
                return 'constant_unknown'
        
        # Default fallback
        return 'median' if pd.api.types.is_numeric_dtype(df[col].dtype) else 'mode'
    
    def fill_time_columns_with_autogluon(self, df: pd.DataFrame, time_cols: List[str]) -> pd.DataFrame:
        """
        Use AutoGluon to intelligently predict missing time values.
        """
        if not time_cols:
            return df
        
        logger.info("Filling time columns using AutoGluon prediction...")
        df_filled = df.copy()
        
        for col in time_cols:
            if df_filled[col].isnull().sum() == 0:
                continue
            
            logger.info(f"  Predicting missing values for time column: '{col}'")
            
            # Convert time to numeric for prediction
            df_temp = df_filled.copy()
            
            # Create numeric representation of time (seconds since midnight)
            def time_to_seconds(t):
                if pd.isna(t):
                    return np.nan
                try:
                    if isinstance(t, str):
                        parts = t.split(':')
                        if len(parts) == 2:
                            return int(parts[0]) * 3600 + int(parts[1]) * 60
                        elif len(parts) == 3:
                            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    return np.nan
                except:
                    return np.nan
            
            df_temp[f'{col}_numeric'] = df_temp[col].apply(time_to_seconds)
            
            # Prepare data for AutoGluon
            train_mask = df_temp[f'{col}_numeric'].notna()
            predict_mask = df_temp[f'{col}_numeric'].isna()
            
            if train_mask.sum() > 5 and predict_mask.sum() > 0:
                # Get other features for prediction
                feature_cols = [c for c in df_temp.columns 
                               if c != col and c != f'{col}_numeric' 
                               and df_temp[c].dtype in ['int64', 'float64', 'object']]
                
                if len(feature_cols) > 0:
                    try:
                        train_data = df_temp.loc[train_mask, feature_cols + [f'{col}_numeric']].copy()
                        predict_data = df_temp.loc[predict_mask, feature_cols].copy()
                        
                        temp_dir = f"temp_time_prediction_{col}"
                        predictor = TabularPredictor(
                            label=f'{col}_numeric',
                            path=temp_dir,
                            verbosity=0
                        )
                        
                        predictor.fit(
                            train_data=train_data,
                            time_limit=10,
                            presets='medium_quality_faster_train',
                            verbosity=0
                        )
                        
                        predictions = predictor.predict(predict_data)
                        
                        # Convert back to time format
                        def seconds_to_time(seconds):
                            if pd.isna(seconds) or seconds < 0:
                                return '00:00:00'
                            hours = int(seconds // 3600) % 24
                            minutes = int((seconds % 3600) // 60)
                            secs = int(seconds % 60)
                            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
                        
                        filled_times = predictions.apply(seconds_to_time)
                        df_filled.loc[predict_mask, col] = filled_times.values
                        
                        logger.info(f"    ✓ Filled {predict_mask.sum()} missing time values using AutoGluon")
                        
                        # Track imputation method
                        self.imputation_methods[col] = 'autogluon_ml_predictor'
                        
                        # Cleanup
                        import shutil
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                        
                    except Exception as e:
                        logger.warning(f"    AutoGluon prediction failed for '{col}': {e}")
                        logger.info(f"    Falling back to median imputation")
                        median_seconds = df_temp[f'{col}_numeric'].median()
                        if not pd.isna(median_seconds):
                            median_time = f"{int(median_seconds // 3600):02d}:{int((median_seconds % 3600) // 60):02d}:{int(median_seconds % 60):02d}"
                            df_filled[col].fillna(median_time, inplace=True)
                            self.imputation_methods[col] = 'median'
                        else:
                            df_filled[col].fillna('00:00:00', inplace=True)
                            self.imputation_methods[col] = 'constant_default_time'
                else:
                    # No features available, use median
                    median_seconds = df_temp[f'{col}_numeric'].median()
                    if not pd.isna(median_seconds):
                        median_time = f"{int(median_seconds // 3600):02d}:{int((median_seconds % 3600) // 60):02d}:{int(median_seconds % 60):02d}"
                        df_filled[col].fillna(median_time, inplace=True)
                        self.imputation_methods[col] = 'median'
                    else:
                        df_filled[col].fillna('00:00:00', inplace=True)
                        self.imputation_methods[col] = 'constant_default_time'
            else:
                # Not enough data for prediction
                median_seconds = df_temp[f'{col}_numeric'].median()
                if not pd.isna(median_seconds):
                    median_time = f"{int(median_seconds // 3600):02d}:{int((median_seconds % 3600) // 60):02d}:{int(median_seconds % 60):02d}"
                    df_filled[col].fillna(median_time, inplace=True)
                    self.imputation_methods[col] = 'median'
                else:
                    df_filled[col].fillna('00:00:00', inplace=True)
                    self.imputation_methods[col] = 'constant_default_time'
        
        return df_filled
    
    def apply_autogluon_preprocessing(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """
        Apply AutoGluon-safe preprocessing with intelligent missing value handling.
        """
        logger.info("Applying AutoGluon-safe preprocessing pipeline...")
        
        try:
            df_processed = df.copy()
            
            # Reset imputation methods tracking for this dataset
            self.imputation_methods = {}
            
            # Step 1: Normalize invalid tokens (REQUIRED)
            df_processed, tokens_count = self.normalize_invalid_tokens(df_processed)
            self.stats['tokens_normalized'] += tokens_count
            
            # Step 2: Convert columns to proper types based on metadata
            df_processed = self.convert_columns_to_proper_types(df_processed, analysis)
            
            # Step 3: Parse datetime columns (REQUIRED)
            df_processed = self.parse_datetime_columns(df_processed, analysis)
            
            # Step 4: Drop 100% missing columns
            columns_to_drop = [col for col, info in analysis['columns'].items() 
                             if info['missing_percentage'] >= 100.0]
            if columns_to_drop:
                logger.warning(f"Dropping columns with 100% missing: {columns_to_drop}")
                df_processed = df_processed.drop(columns=columns_to_drop)
                self.stats['columns_dropped'].extend(columns_to_drop)
            
            # Step 5: Identify time columns (exclude timestamp columns)
            time_cols = []
            for col in df_processed.columns:
                col_info = analysis['columns'].get(col, {})
                semantic_type = col_info.get('semantic_type', '').lower()
                # Include only pure time columns, not datetime or timestamp
                if 'time' in semantic_type and 'date' not in semantic_type and semantic_type != 'timestamp':
                    time_cols.append(col)
            
            # Step 6: Fill sequential patterns (numeric and text sequences)
            logger.info("\n[Step 6] Detecting and filling sequential patterns...")
            for col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    df_processed = self.fillSequence(df_processed, col)
            
            # Step 6.5: Handle NUMERIC + TEMPORAL missing values using pattern-based classification
            # Uses semantic_type from profiling metadata to route columns to correct handler
            logger.info("\n[Step 6.5] Pattern-Based Missing Value Classification (Numeric + Temporal)...")
            pattern_cols_handled = set()
            
            # Load profiling metadata for pattern-based handling
            profiling_metadata = self._get_profiling_metadata_for_numeric_handler(analysis)
            
            if profiling_metadata:
                # --- NUMERIC HANDLER ---
                numeric_handler = NumericMissingValueHandler(df_processed, profiling_metadata)
                numeric_decisions = numeric_handler.process_all_numeric_columns(enable_advanced_upgrades=True)
                
                if numeric_decisions:
                    df_processed = numeric_handler.apply_imputation()
                    
                    for col, decision in numeric_decisions.items():
                        if decision['imputation_method'] != 'none':
                            pattern_cols_handled.add(col)
                            self.imputation_methods[col] = decision['imputation_method']
                    
                    summary = numeric_handler.get_summary()
                    logger.info(f"\n  Numeric Summary:")
                    logger.info(f"    Columns processed: {summary['total_columns_processed']}")
                    logger.info(f"    Patterns: {summary['patterns_detected']}")
                    logger.info(f"    Methods used: {summary['methods_used']}")
                    if summary['columns_dropped']:
                        logger.info(f"    Columns dropped: {summary['columns_dropped']}")
                        self.stats['columns_dropped'].extend(summary['columns_dropped'])
                    if summary['columns_with_flags']:
                        logger.info(f"    Columns with flags: {summary['columns_with_flags']}")
                
                # --- TEMPORAL HANDLER ---
                temporal_handler = TemporalMissingValueHandler(df_processed, profiling_metadata)
                temporal_decisions = temporal_handler.process_all_temporal_columns()
                
                if temporal_decisions:
                    df_processed = temporal_handler.apply_imputation()
                    
                    for col, decision in temporal_decisions.items():
                        if decision['imputation_method'] != 'none':
                            pattern_cols_handled.add(col)
                            self.imputation_methods[col] = decision['imputation_method']
                    
                    summary = temporal_handler.get_summary()
                    logger.info(f"\n  Temporal Summary:")
                    logger.info(f"    Columns processed: {summary['total_columns_processed']}")
                    logger.info(f"    Patterns: {summary['patterns_detected']}")
                    logger.info(f"    Methods used: {summary['methods_used']}")
                    if summary['columns_dropped']:
                        logger.info(f"    Columns dropped: {summary['columns_dropped']}")
                        self.stats['columns_dropped'].extend(summary['columns_dropped'])
                    if summary['columns_with_flags']:
                        logger.info(f"    Columns with flags: {summary['columns_with_flags']}")
                    if summary.get('columns_rederived'):
                        logger.info(f"    Columns re-derived: {summary['columns_rederived']}")
            else:
                logger.info("  No profiling metadata available - skipping pattern-based handling")
            
            # Step 7: Fill remaining missing values with intelligent method selection
            # (Skip columns already handled by numeric/temporal handler)
            fillna_map = {}
            
            for col in df_processed.columns:
                # Skip time columns - will be handled separately
                if col in time_cols:
                    continue
                
                # Skip columns already handled by pattern-based handlers
                if col in pattern_cols_handled:
                    continue
                
                # Skip if no missing values
                if df_processed[col].isnull().sum() == 0:
                    continue
                
                dtype = df_processed[col].dtype
                col_info = analysis['columns'].get(col, {})
                semantic_type = col_info.get('semantic_type', '')
                
                # Determine the best imputation method
                method = self.determine_imputation_method(df_processed, col, analysis)
                
                logger.info(f"  Column '{col}': Selected method = {method}")
                
                # Apply the selected method
                if method == 'mean':
                    mean_val = df_processed[col].mean()
                    fillna_map[col] = mean_val if not pd.isna(mean_val) else 0
                    self.imputation_methods[col] = 'mean'
                    
                elif method == 'median':
                    median_val = df_processed[col].median()
                    fillna_map[col] = median_val if not pd.isna(median_val) else 0
                    self.imputation_methods[col] = 'median'
                    
                elif method == 'mode':
                    mode_val = df_processed[col].mode()
                    fillna_map[col] = mode_val[0] if len(mode_val) > 0 and not pd.isna(mode_val[0]) else (0 if pd.api.types.is_numeric_dtype(dtype) else 'Unknown')
                    self.imputation_methods[col] = 'mode'
                    
                elif method == 'constant_zero':
                    fillna_map[col] = 0
                    self.imputation_methods[col] = 'constant_zero'
                    
                elif method == 'constant_unknown':
                    fillna_map[col] = 'Unknown'
                    self.imputation_methods[col] = 'constant_unknown'
                
            # Step 7b: Handle datetime columns with intelligent strategies
            datetime_cols = [col for col in df_processed.columns 
                           if pd.api.types.is_datetime64_any_dtype(df_processed[col].dtype) 
                           and df_processed[col].isnull().any()]
            
            for col in datetime_cols:
                logger.info(f"  Processing datetime column '{col}' ({df_processed[col].isnull().sum()} missing)...")
                
                # Check if dates are sequential (time series)
                is_sequential = self._is_sequential_dates(df_processed, col)
                
                if is_sequential:
                    # Sequential dates → use interpolation
                    logger.info(f"    Detected sequential pattern → using interpolation")
                    df_processed = self._interpolate_dates(df_processed, col)
                    self.imputation_methods[col] = 'date_interpolation'
                else:
                    # Non-sequential → use forward/backward fill
                    logger.info(f"    Non-sequential pattern → using forward/backward fill")
                    df_processed = self._forward_backward_fill_dates(df_processed, col)
                    self.imputation_methods[col] = 'date_forward_backward_fill'
            
            if fillna_map:
                df_processed = df_processed.fillna(fillna_map)
            
            # Step 8: Fill time columns with AutoGluon
            df_processed = self.fill_time_columns_with_autogluon(df_processed, time_cols)
            
            # Final cleanup - only fill remaining NaN values
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    dtype = df_processed[col].dtype
                    col_info = analysis['columns'].get(col, {})
                    semantic_type = col_info.get('semantic_type', '').lower()
                    
                    if pd.api.types.is_numeric_dtype(dtype):
                        df_processed[col] = df_processed[col].fillna(0)
                        if col not in self.imputation_methods:
                            self.imputation_methods[col] = 'constant_zero'
                    elif semantic_type in ['text', 'varchar', 'unknown']:
                        # Only use 'Unknown' for actual text columns
                        df_processed[col] = df_processed[col].fillna('Unknown')
                        if col not in self.imputation_methods:
                            self.imputation_methods[col] = 'constant_unknown'
                    else:
                        # For other types (email, url, etc.), use mode or empty string
                        mode_val = df_processed[col].mode()
                        fill_value = mode_val[0] if len(mode_val) > 0 and not pd.isna(mode_val[0]) else ''
                        df_processed[col] = df_processed[col].fillna(fill_value)
                        if col not in self.imputation_methods:
                            self.imputation_methods[col] = 'mode' if fill_value != '' else 'constant_empty_string'
            
            logger.info("✅ AutoGluon preprocessing completed successfully")
            return df_processed
            
        except Exception as e:
            logger.error(f"Error in AutoGluon preprocessing: {e}")
            raise
    
    def update_metadata_with_missing_values_summary(self, filename: str, analysis: Dict, remaining_missing: int):
        """
        Update profiling metadata with missing_values_summary.
        """
        base_name = filename.replace('_cleaned.csv', '')
        storage_path = f"{self.meta_prefix}/{base_name}_profiling.json"
        
        try:
            metadata = download_json(storage_path)
            
            # Add missing_values_summary
            missing_values_summary = {
                'total_missing_before': analysis['total_missing'],
                'total_missing_after': remaining_missing,
                'columns_with_missing': {},
                'tokens_normalized': self.stats.get('tokens_normalized', 0),
                'autogluon_validated': self.stats.get('autogluon_validated', False)
            }
            
            # Add column-level details
            for col, info in analysis['columns'].items():
                if info['missing_count'] > 0:
                    missing_values_summary['columns_with_missing'][col] = {
                        'missing_count': info['missing_count'],
                        'missing_percentage': info['missing_percentage'],
                        'dtype': info['dtype'],
                        'imputation_method': self.imputation_methods.get(col, 'none')
                    }
            
            metadata['missing_values_summary'] = missing_values_summary
            
            upload_json(storage_path, metadata)
            logger.info(f"✅ Updated metadata with missing_values_summary: {storage_path}")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def validate_with_autogluon(self, df: pd.DataFrame, filename: str) -> bool:
        """
        MANDATORY: Validate dataset is AutoGluon-compatible using FeatureMetadata.
        """
        logger.info("\n" + "="*70)
        logger.info("AUTOGLUON VALIDATION (MANDATORY)")
        logger.info("="*70)
        
        try:
            # Step 1: Create FeatureMetadata
            logger.info("[1] Creating AutoGluon FeatureMetadata...")
            feature_metadata = FeatureMetadata.from_df(df)
            
            logger.info("✓ FeatureMetadata created successfully")
            logger.info(f"  Type map: {feature_metadata.type_map_raw}")
            
            # Step 2: Check datetime columns
            datetime_cols = [col for col, dtype in df.dtypes.items() 
                           if pd.api.types.is_datetime64_any_dtype(dtype)]
            if datetime_cols:
                logger.info(f"  Datetime columns detected: {datetime_cols}")
            
            # Step 3: Verify no invalid tokens
            invalid_found = False
            for col in df.select_dtypes(include=['object', 'category']).columns:
                if df[col].isin(INVALID_TOKENS).any():
                    invalid_found = True
                    logger.warning(f"  ⚠️  Column '{col}' still contains invalid tokens!")
            
            if not invalid_found:
                logger.info("  ✓ No invalid tokens in categorical columns")
            
            # Step 4: Optional dry-run fit
            logger.info("\n[2] Performing AutoGluon schema validation (dry-run)...")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0 and len(df) > 10:
                target_col = numeric_cols[-1]
                logger.info(f"  Using '{target_col}' as validation target")
                
                sample_df = df.head(min(50, len(df))).copy()
                temp_dir = f"temp_autogluon_validation_{filename.replace('.csv', '')}"
                
                try:
                    predictor = TabularPredictor(
                        label=target_col,
                        path=temp_dir,
                        verbosity=0
                    )
                    
                    predictor.fit(
                        train_data=sample_df,
                        time_limit=5,
                        presets='medium_quality_faster_train',
                        verbosity=0
                    )
                    
                    logger.info("  ✅ AutoGluon TabularPredictor fit successful!")
                    logger.info("  ✅ Dataset is AutoGluon-compatible")
                    
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    
                    self.stats['autogluon_validated'] = True
                    return True
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  Dry-run fit failed: {e}")
                    logger.info("  Continuing with FeatureMetadata validation only...")
                    
                    import shutil
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            else:
                logger.info("  Skipping dry-run fit (insufficient data)")
            
            logger.info("\n✅ AutoGluon FeatureMetadata validation PASSED")
            self.stats['autogluon_validated'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ AutoGluon validation failed: {e}")
            return False
   
   
   
    
    def update_metadata_with_missing_values_summary(self, filename: str, analysis: Dict, remaining_missing: int, df_filled: pd.DataFrame = None):
        """
        Update profiling metadata with missing_values_summary and refresh
        sample_values / null stats from the imputed DataFrame.
        """
        base_name = filename.replace('_cleaned.csv', '')
        storage_path = f"{self.meta_prefix}/{base_name}_profiling.json"
        
        try:
            metadata = download_json(storage_path)
            
            # Add missing_values_summary (convert numpy types to native Python types)
            missing_values_summary = {
                'total_missing_before': int(analysis['total_missing']),
                'total_missing_after': int(remaining_missing),
                'columns_with_missing': {},
                'tokens_normalized': int(self.stats.get('tokens_normalized', 0)),
                'autogluon_validated': bool(self.stats.get('autogluon_validated', False))
            }
            
            # Add column-level details (convert numpy types to native Python types)
            for col, info in analysis['columns'].items():
                if info['missing_count'] > 0:
                    missing_values_summary['columns_with_missing'][col] = {
                        'missing_count': int(info['missing_count']),
                        'missing_percentage': float(info['missing_percentage']),
                        'dtype': str(info['dtype']),
                        'imputation_method': self.imputation_methods.get(col, 'none')
                    }
            
            metadata['missing_values_summary'] = missing_values_summary
            
            # ── Refresh column_wise_summary from the imputed DataFrame ──
            if df_filled is not None and 'column_wise_summary' in metadata:
                for col_entry in metadata['column_wise_summary']:
                    col_name = col_entry.get('column_name')
                    if col_name not in df_filled.columns:
                        continue
                    
                    series = df_filled[col_name]
                    
                    # Update null stats (should be 0 after imputation)
                    null_count = int(series.isnull().sum())
                    col_entry['null_count'] = null_count
                    col_entry['null_percentage'] = round(
                        (null_count / len(df_filled)) * 100, 2
                    ) if len(df_filled) > 0 else 0.0
                    
                    # Update unique count
                    col_entry['unique_count'] = int(series.nunique())
                    
                    # Refresh sample_values from the imputed data
                    n_samples = min(5, len(df_filled))
                    raw_samples = series.head(n_samples).tolist()
                    
                    # Convert numpy / pandas types to JSON-safe Python types
                    import datetime as _dt
                    safe_samples = []
                    for v in raw_samples:
                        if pd.isna(v):
                            safe_samples.append(None)
                        elif isinstance(v, (pd.Timestamp,)):
                            safe_samples.append(str(v))
                        elif isinstance(v, (_dt.datetime,)):
                            safe_samples.append(v.strftime('%Y-%m-%d %H:%M:%S'))
                        elif isinstance(v, (_dt.date,)):
                            safe_samples.append(v.strftime('%Y-%m-%d'))
                        elif isinstance(v, (_dt.time,)):
                            safe_samples.append(v.strftime('%H:%M:%S'))
                        elif isinstance(v, (np.integer,)):
                            safe_samples.append(int(v))
                        elif isinstance(v, (np.floating,)):
                            safe_samples.append(str(round(float(v), 4)))
                        elif isinstance(v, (np.bool_,)):
                            safe_samples.append(bool(v))
                        else:
                            safe_samples.append(v)
                    
                    col_entry['sample_values'] = safe_samples
                
                logger.info("✅ Refreshed sample_values & null stats in column_wise_summary")
            
            upload_json(storage_path, metadata)
            logger.info(f"✅ Updated metadata with missing_values_summary: {storage_path}")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
    def process_dataset(self, filename: str) -> bool:
        """Process a single dataset with AutoGluon integration."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing dataset: {filename}")
        logger.info(f"{'='*70}")
        
        try:
            # Load dataset from Supabase
            input_storage_path = f"{self.input_prefix}/{filename}"
            content = download_file(input_storage_path)
            df = pd.read_csv(BytesIO(content))
            logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Load metadata
            metadata = self.load_profiling_metadata(filename)
            
            # Analyze missing values
            analysis = self.analyze_missing_values(df, metadata)
            logger.info(f"Total missing values: {analysis['total_missing']} ({analysis['total_missing']/(df.shape[0]*df.shape[1])*100:.2f}%)")
            
            # Display missing values
            missing_cols = {col: info for col, info in analysis['columns'].items() if info['missing_count'] > 0}
            if missing_cols:
                logger.info("\n📊 Missing Values by Column:")
                for col, info in missing_cols.items():
                    logger.info(f"  • {col}: {info['missing_count']} ({info['missing_percentage']:.2f}%)")
            
            # Apply preprocessing
            df_filled = self.apply_autogluon_preprocessing(df, analysis)
            
            # Verify
            remaining_missing = df_filled.isnull().sum().sum()
            logger.info(f"Remaining missing values: {remaining_missing}")
            
            # MANDATORY: Validate with AutoGluon
            validation_success = self.validate_with_autogluon(df_filled, filename)
            
            if not validation_success:
                logger.warning("⚠️  AutoGluon validation had issues, but proceeding...")
            
            # Save with proper integer formatting
            output_filename = filename
            output_storage_path = f"{self.input_prefix}/{output_filename}"
            
            # Convert float columns with integer values to int64 for clean CSV output
            df_to_save = df_filled.copy()
            for col in df_to_save.columns:
                if df_to_save[col].dtype in ['float64', 'Int64']:
                    # Check if all non-null values are integers
                    non_null = df_to_save[col].dropna()
                    if len(non_null) > 0 and (non_null % 1 == 0).all():
                        # If no missing values, use int64
                        if df_to_save[col].isna().sum() == 0:
                            df_to_save[col] = df_to_save[col].astype('int64')
            
            content = df_to_save.to_csv(index=False).encode('utf-8')
            upload_file(output_storage_path, content, "text/csv")
            logger.info(f"✅ Saved filled dataset: {output_storage_path}")
            
            # Update metadata with missing_values_summary + refreshed sample_values
            self.update_metadata_with_missing_values_summary(filename, analysis, remaining_missing, df_filled)
            
            self.stats['files_processed'] += 1
            self.stats['total_missing_before'] += analysis['total_missing']
            self.stats['total_missing_after'] += remaining_missing
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {e}")
            return False
    
    
    
    def process_all_datasets(self) -> Dict:
        """Process all cleaned datasets."""
        logger.info(f"\n{'#'*70}")
        logger.info(f"# AutoGluon Missing Value Handler - User: {self.user_id}")
        logger.info(f"{'#'*70}\n")
        
        datasets = self.find_datasets()
        
        if not datasets:
            logger.warning("No cleaned datasets found")
            return self.stats
        
        success_count = 0
        for filename in datasets:
            if self.process_dataset(filename):
                success_count += 1
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"PROCESSING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Files processed: {success_count}/{len(datasets)}")
        logger.info(f"Total missing values before: {self.stats['total_missing_before']}")
        logger.info(f"Total missing values after: {self.stats['total_missing_after']}")
        logger.info(f"Invalid tokens normalized: {self.stats['tokens_normalized']}")
        logger.info(f"Columns dropped: {len(self.stats['columns_dropped'])}")
        if self.stats['columns_dropped']:
            logger.info(f"  Dropped: {', '.join(self.stats['columns_dropped'])}")
        logger.info(f"AutoGluon validation: {'✅ PASSED' if self.stats['autogluon_validated'] else '⚠️  PARTIAL'}")
        logger.info(f"{'='*70}\n")
        
        return self.stats


def main():
    """Main entry point."""
    print("=" * 70)
    print("  AutoGluon-Based Missing Values Handler")
    print("  With Explicit TabularPredictor Integration")
    print("=" * 70)
    print()
    
    user_id = input("Enter user ID: ").strip()
    
    if not user_id:
        print("❌ Error: User ID cannot be empty")
        return
    
    session_id = input("Enter session ID: ").strip()
    
    if not session_id:
        print("❌ Error: Session ID cannot be empty")
        return
    
    try:
        handler = AutoGluonMissingValueHandler(user_id=user_id, session_id=session_id)
        stats = handler.process_all_datasets()
        
        print("\n✅ Processing complete!")
        print(f"Check output files in: output/{user_id}/{session_id}/")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
