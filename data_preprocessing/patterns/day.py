import re
import pandas as pd
import numpy as np
from collections import Counter
from .base import BasePattern


class DayPattern(BasePattern):
    semantic_type = "day"
    
    regex_patterns = [
        # Basic day numbers
        
        # Day names (short and full)
        r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*$',                          # Mon, Monday
        r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',   # Full day names
        
        # Ordinal days
        r'^(1st|2nd|3rd|[4-9]th|1[0-9]th|2[0-9]th|3[01]th|21st|22nd|23rd|31st)$',  # 1st, 2nd, 31st
        
        # Descriptive formats with day
        r'^Day\s+\d{1,2}$',                                                # Day 01
        r'^\d{1,2}(st|nd|rd|th)?\s+day$',                                  # 1st day
        r'^D\d{1,2}$',                                                     # D1
        r'^Day\d{1,2}$',                                                   # Day1
        
        # Combined day name + number
        r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*-\d{1,2}$',                 # Mon-01
        r'^\d{1,2}-(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*$',                 # 01-Mon
        r'^\d{1,2}\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*$',               # 1 Monday
        r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+\d{1,2}(st|nd|rd|th)?$', # Monday 1st
        
    ]
    
    # Day name mappings
    day_name_short = {
        'monday': 'Mon', 'tuesday': 'Tue', 'wednesday': 'Wed',
        'thursday': 'Thu', 'friday': 'Fri', 'saturday': 'Sat', 'sunday': 'Sun',
        'mon': 'Mon', 'tue': 'Tue', 'wed': 'Wed',
        'thu': 'Thu', 'fri': 'Fri', 'sat': 'Sat', 'sun': 'Sun'
    }
    
    day_name_full = {
        'mon': 'Monday', 'tue': 'Tuesday', 'wed': 'Wednesday',
        'thu': 'Thursday', 'fri': 'Friday', 'sat': 'Saturday', 'sun': 'Sunday',
        'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
        'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 'sunday': 'Sunday'
    }
    
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
    
    def is_day_name(self, val_str):
        """Check if value is a day name."""
        return bool(re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*$', val_str, re.IGNORECASE) or
                   re.match(r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$', val_str, re.IGNORECASE))
    
    def is_day_number(self, val_str):
        """Check if value is a day number."""
        return bool(re.match(r'^(0?[1-9]|[12][0-9]|3[01])$', val_str) or
                   re.match(r'^(1st|2nd|3rd|[4-9]th|1[0-9]th|2[0-9]th|3[01]th|21st|22nd|23rd|31st)$', val_str, re.IGNORECASE))
    
    def normalize(self, values):
        """Normalize to most common format type (day names or day numbers)."""
        # Detect whether most values are day names or day numbers
        name_count = 0
        number_count = 0
        
        for val in values:
            if pd.notna(val):
                val_str = str(val).strip()
                if self.is_day_name(val_str):
                    name_count += 1
                elif self.is_day_number(val_str):
                    number_count += 1
        
        # Determine most common type
        use_day_names = name_count > number_count
        
        if use_day_names:
            # Normalize to day names (preserve names, skip numbers)
            def parse_day(val):
                if pd.isna(val):
                    return val
                try:
                    val_str = str(val).strip()
                    if self.is_day_name(val_str):
                        # Normalize to full day name
                        return self.day_name_full.get(val_str.lower(), val_str)
                    # Can't convert numbers to day names without date context
                    return val
                except:
                    return val
        else:
            # Normalize to zero-padded day numbers
            def parse_day(val):
                if pd.isna(val):
                    return val
                try:
                    val_str = str(val).strip()
                    
                    if self.is_day_number(val_str):
                        # Remove ordinal suffixes
                        val_str = re.sub(r'(st|nd|rd|th)$', '', val_str, flags=re.IGNORECASE)
                        day = int(val_str)
                        if 1 <= day <= 31:
                            return f"{day:02d}"
                    # Can't convert day names to numbers without date context
                    return val
                except:
                    return val
        
        return values.apply(parse_day)
