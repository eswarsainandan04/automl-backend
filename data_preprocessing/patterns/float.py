import re
import pandas as pd
import numpy as np
from .base import BasePattern


class FloatPattern(BasePattern):
    semantic_type = "float"
    
    regex_patterns = [
        r'^[+-]?(\d+\.\d+)$',  # Float: 123.45, -67.89
        r'^[+-]?\d{1,3}(,\d{3})*\.\d+$',  # Float with commas: 1,234.56
        r'^[+-]?(\.\d+)$',  # Float starting with decimal: .5, .123
        r'^[+-]?(\d+\.?\d*)([eE][+-]?\d+)$',  # Scientific notation: 1.5e10
    ]
    
    def detect(self, values) -> float:
        """
        Detect if column contains float/decimal values.
        Returns confidence score between 0 and 1.
        """
        if len(values) == 0:
            return 0.0
        
        # Check if column is named "float" or "decimal" (case-insensitive)
        column_name = ""
        if hasattr(values, 'name') and values.name:
            column_name = str(values.name).lower()
        
        matched = 0
        total = 0
        has_decimal = False
        
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.lower() in ['null', 'nan', 'none', '']:
                continue
                
            total += 1
            
            try:
                # Check if it's already a float type
                if isinstance(value, (float, np.floating)):
                    matched += 1
                    if value != int(value):
                        has_decimal = True
                    continue
                
                # If it's an integer, it can be converted to float
                if isinstance(value, (int, np.integer)):
                    matched += 1
                    continue
                
                # If string, try to parse
                value_str = str(value).strip()
                
                # Remove common separators (commas, spaces)
                value_str = value_str.replace(',', '').replace(' ', '')
                
                # Check if it matches float pattern
                if re.match(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$', value_str):
                    matched += 1
                    # Check if has decimal point
                    if '.' in value_str:
                        has_decimal = True
                    
            except:
                continue
        
        if total == 0:
            return 0.0
        
        # Only consider it a float column if at least one value has decimals
        confidence = matched / total
        
        # Boost confidence if column is named "float" or "decimal"
        if 'float' in column_name or 'decimal' in column_name:
            confidence = min(1.0, confidence * 1.5)  # 50% boost
        
        # If no decimals found, reduce confidence (might be integer column)
        if not has_decimal and 'float' not in column_name:
            confidence *= 0.5
            
        return confidence
    
    def normalize(self, values):
        """
        Normalize all values to float format with 2 decimal places.
        Preserves NaN/None values.
        """
        def clean_float(val):
            if pd.isna(val):
                return np.nan
            
            try:
                # If already numeric, convert to float
                if isinstance(val, (int, float, np.integer, np.floating)):
                    return round(float(val), 2)
                
                # If string, clean and convert
                val_str = str(val).strip()
                
                # Remove commas, spaces
                val_str = val_str.replace(',', '').replace(' ', '')
                
                # Convert to float and round to 2 decimal places
                return round(float(val_str), 2)
                
            except:
                return np.nan
        
        return values.apply(clean_float)
