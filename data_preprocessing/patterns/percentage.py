import re
import pandas as pd
import numpy as np
from .base import BasePattern


class PercentagePattern(BasePattern):
    semantic_type = "percentage"
    detected_suffix = None  # Will store '_percentage' for column renaming
    
    regex_patterns = [
        r'^\d+(?:\.\d+)?%$',  # 25.5%
        r'^\d+(?:\.\d+)?\s*percent$',  # 25.5 percent
        r'^\d+(?:\.\d+)?\s*pct$',  # 25.5 pct
        r'^\d+(?:\.\d+)?\s*\%$',  # With space before %
    ]
    
    def detect(self, values) -> float:
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
                if isinstance(value, float):
                    value_str = str(value)
                else:
                    value_str = str(value).strip()
            except:
                value_str = str(value).strip()
                
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def normalize(self, values):
        # Set suffix for column renaming
        self.detected_suffix = '_percentage'
        
        def clean_percentage(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Remove % symbol, percent, pct text
                val_str = re.sub(r'%|percent|pct', '', val_str, flags=re.IGNORECASE)
                
                # Remove ALL remaining text and symbols - keep only digits, decimal point, minus
                val_str = re.sub(r'[^\d.\-]', '', val_str)
                
                # Return nan if no digits remain
                if not val_str or val_str in ['.', '-', '.-', '-.']:
                    return np.nan
                
                number = float(val_str)
                # Convert percentage to decimal (e.g., 45% -> 0.45)
                decimal_value = number / 100
                return decimal_value
            except:
                return np.nan
        
        return values.apply(clean_percentage)
