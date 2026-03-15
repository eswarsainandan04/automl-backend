import re
import pandas as pd
import numpy as np
from .base import BasePattern


class RatioPattern(BasePattern):
    semantic_type = "ratio"
    
    regex_patterns = [
        r'^\d+:\d+$',  # 16:9, 3:2
        r'^\d+(?:\.\d+)?:\d+(?:\.\d+)?$',  # 1.5:1
        r'^\d+/\d+$',  # 2/3, 1/2 (fraction format)
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
            value_str = str(value).strip()
                
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def normalize(self, values):
        def parse_ratio(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Handle percentage values (just remove % symbol)
                if '%' in val_str:
                    return val_str.replace('%', '').strip()
                
                # Convert "to" format to colon
                val_str = val_str.replace(' to ', ':').replace(' TO ', ':')
                
                # Convert fraction format (2/3) to colon format
                if '/' in val_str:
                    parts = val_str.split('/')
                    if len(parts) == 2:
                        num1 = parts[0].strip()
                        num2 = parts[1].strip()
                        return f"{num1}:{num2}"
                
                # Return as-is if already in ratio format
                return val_str
            except:
                return np.nan
        
        return values.apply(parse_ratio)
