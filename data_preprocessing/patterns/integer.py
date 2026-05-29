import re
import pandas as pd
import numpy as np
from .base import BasePattern


class IntegerPattern(BasePattern):
    semantic_type = "integer"
    
    regex_patterns = [
        r'^[+-]?\d+$',  # Integer: 123, -456, +789
        r'^[+-]?\d{1,3}(,\d{3})*$',  # Integer with commas: 1,234,567
    ]
    
    def detect(self, values) -> float:
        """
        Detect if column contains integer values.
        Returns confidence score between 0 and 1.
        """
        if len(values) == 0:
            return 0.0
        
        matched = 0
        total = 0
        
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.lower() in ['null', 'nan', 'none', '']:
                continue
                
            total += 1
            
            try:
                # Check if it's already a numeric type
                if isinstance(value, (int, np.integer)):
                    matched += 1
                    continue
                
                # If it's a float, check if it has no decimal part
                if isinstance(value, (float, np.floating)):
                    if value == int(value):
                        matched += 1
                    continue
                
                # If string, try to parse
                value_str = str(value).strip()
                
                # Remove common separators (commas, spaces)
                value_str = value_str.replace(',', '').replace(' ', '')
                
                # Check if it matches integer pattern (optionally with + or -)
                if re.match(r'^[+-]?\d+$', value_str):
                    matched += 1
                    
            except:
                continue
        
        if total == 0:
            return 0.0
            
        return matched / total
    
    def normalize(self, values):
        """
        Normalize all values to integer format.
        Preserves NaN/None values.
        """
        def clean_integer(val):
            if pd.isna(val):
                return np.nan
            
            try:
                # If already integer, return as-is
                if isinstance(val, (int, np.integer)):
                    return int(val)
                
                # If float, convert to int
                if isinstance(val, (float, np.floating)):
                    return int(val)
                
                # If string, clean and convert
                val_str = str(val).strip()
                
                # Remove commas, spaces
                val_str = val_str.replace(',', '').replace(' ', '')
                
                # Convert to integer
                return int(float(val_str))
                
            except:
                return np.nan
        
        return values.apply(clean_integer)
