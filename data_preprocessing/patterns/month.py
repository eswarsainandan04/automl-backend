import re
import pandas as pd
import numpy as np
from collections import Counter
from .base import BasePattern


class MonthPattern(BasePattern):
    semantic_type = "month"
    
    regex_patterns = [
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*$',
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)$',
    ]
    
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5, 'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    month_names_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        # Column name awareness - boost confidence if column is named 'month'
        column_name = getattr(values, 'name', '').lower() if hasattr(values, 'name') else ''
        confidence_boost = 0.3 if 'month' in column_name else 0.0
        
        matched = 0
        for value in values:
            value_str = str(value).strip()
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        base_confidence = matched / len(values)
        return min(1.0, base_confidence + confidence_boost)
    
    def is_month_name(self, val_str):
        """Check if value is a month name."""
        return bool(re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*$', val_str, re.IGNORECASE) or
                   re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)$', val_str, re.IGNORECASE))
    
    def is_month_number(self, val_str):
        """Check if value is a month number."""
        clean_val = re.sub(r'\d{4}[-/]?', '', val_str)
        clean_val = re.sub(r'[-/]?\d{4}', '', clean_val).strip()
        return bool(re.match(r'^(0?[1-9]|1[0-2])$', clean_val))
    
    def normalize(self, values):
        """Normalize all months to short month names (Jan, Feb, Mar, etc.)."""
        # Always normalize to short month names like "Jan", "Feb", etc.
        def parse_month(val):
            if pd.isna(val):
                return val
            try:
                val_str = str(val).strip().lower()
                
                # Check if it's already a text month name
                for month_name, month_num in self.month_map.items():
                    if val_str.startswith(month_name):
                        return self.month_names_short[month_num - 1]
                
                # Try to convert number to month name
                val_clean = re.sub(r'\d{4}[-/]?', '', val_str)
                val_clean = re.sub(r'[-/]?\d{4}', '', val_clean).strip()
                month = int(val_clean)
                if 1 <= month <= 12:
                    return self.month_names_short[month - 1]
                
                return val
            except:
                return val
        
        return values.apply(parse_month)
