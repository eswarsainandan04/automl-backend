import re
import pandas as pd
import numpy as np
from .base import BasePattern


class WeekPattern(BasePattern):
    semantic_type = "week"
    
    regex_patterns = [
        r'^W(0?[1-9]|[1-4][0-9]|5[0-3])$',
        r'^Week\s*(0?[1-9]|[1-4][0-9]|5[0-3])$',
        r'^Wk\s*(0?[1-9]|[1-4][0-9]|5[0-3])$',
        r'^\d{4}-W(0[1-9]|[1-4][0-9]|5[0-3])$',
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        matched = 0
        total = 0
        for value in values:
            # Skip null/nan values
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.lower() in ['null', 'nan', 'none', '']:
                continue
                
            total += 1
            # Convert to int first to remove decimal point if float
            try:
                if isinstance(value, float):
                    value_str = str(int(value))
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
        def parse_week(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Extract year if present
                year_match = re.match(r'^(\d{4})-?W', val_str)
                if year_match:
                    year = year_match.group(1)
                else:
                    # Use current year as default
                    year = str(pd.Timestamp.now().year)
                
                # Extract week number
                week_match = re.search(r'W?(?:eek|k)?\s*(\d{1,2})', val_str, re.IGNORECASE)
                if week_match:
                    week = int(week_match.group(1))
                else:
                    # Try to parse as plain number
                    week = int(val_str)
                
                if 1 <= week <= 53:
                    return f"{year}-W{week:02d}"
                return np.nan
            except:
                return np.nan
        
        return values.apply(parse_week)
