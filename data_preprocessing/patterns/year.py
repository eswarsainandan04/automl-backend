import re
import pandas as pd
import numpy as np
from .base import BasePattern


class YearPattern(BasePattern):
    semantic_type = "year"
    
    regex_patterns = [
            

        

        
        # Year with prefixes/suffixes
        r'^Year\s+\d{4}$',                             # Year 2025
        r'^\d{4}\s+(AD|CE|BC|BCE)$',                   # 2025 AD, 2025 CE
        r'^(AD|CE|BC|BCE)\s+\d{4}$',                   # AD 2025
        r'^Anno\s+\d{4}$',                             # Anno 2025
        r'^\d{4}th$',                                  # 2025th
        r'^\d{4}\s+year$',                             # 2025 year
        r'^\d{4}\s+era$',                              # 2025 era
        
        # Compact formats
        r'^\d{4}CE$',                                  # 2025CE
        r'^Y\d{4}$',                                   # Y2025
        r'^\d[K]\d{2}$',                               # 2K25
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        matched = 0
        for value in values:
            value_str = str(value).strip()
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str):
                    matched += 1
                    break
        
        return matched / len(values)
    
    def normalize(self, values):
        def parse_year(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Remove common prefixes and suffixes
                val_str = re.sub(r'^(FY|Year|Anno|AD|CE|BC|BCE|Y)\s*', '', val_str, flags=re.IGNORECASE)
                val_str = re.sub(r'\s*(AD|CE|BC|BCE|th|year|era|K)$', '', val_str, flags=re.IGNORECASE)
                
                # Handle concatenated year ranges (e.g., 202526 -> 2025)
                if len(val_str) == 6 and val_str.isdigit():
                    val_str = val_str[:4]
                
                # Handle year ranges - take first year
                if '-' in val_str or '/' in val_str:
                    val_str = re.split(r'[-/]', val_str)[0]
                
                # Remove apostrophe
                val_str = val_str.replace("'", "")
                
                # Clean remaining non-digits
                val_str = re.sub(r'\D', '', val_str)
                
                if not val_str:
                    return np.nan
                
                year = int(val_str)
                
                # Convert 2-digit to 4-digit year
                if year < 100:
                    if year < 50:
                        year += 2000
                    else:
                        year += 1900
                
                # Validate reasonable year range (1900-2100)
                if 1900 <= year <= 2100:
                    return year
                else:
                    return np.nan
            except:
                return np.nan
        
        return values.apply(parse_year)
