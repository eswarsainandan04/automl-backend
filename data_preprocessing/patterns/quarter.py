import re
import pandas as pd
import numpy as np
from .base import BasePattern


class QuarterPattern(BasePattern):
    semantic_type = "quarter"
    
    regex_patterns = [
        r'^Q[1-4]$',
        r'^q[1-4]$',
        r'^\d{4}-Q[1-4]$',
        r'^\d{4}Q[1-4]$',
        r'^Q[1-4]\s*\d{4}$',
        r'^Quarter\s*[1-4]$',
        r'^\d{4}-Quarter\s*[1-4]$',
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        matched = 0
        for value in values:
            value_str = str(value).strip()
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        return matched / len(values)
    
    def normalize(self, values):
        def parse_quarter(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Extract year if present
                year_match = re.search(r'(\d{4})', val_str)
                if year_match:
                    year = year_match.group(1)
                else:
                    # Use current year as default
                    year = str(pd.Timestamp.now().year)
                
                # Extract quarter number
                quarter_match = re.search(r'[Qq](?:uarter)?\s*([1-4])', val_str)
                if quarter_match:
                    quarter = quarter_match.group(1)
                else:
                    # Try to parse as plain number
                    quarter = str(int(val_str))
                
                if quarter in ['1', '2', '3', '4']:
                    return f"{year}-Q{quarter}"
                return np.nan
            except:
                return np.nan
        
        return values.apply(parse_quarter)
