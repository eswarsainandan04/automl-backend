import re
import pandas as pd
import numpy as np
from .base import BasePattern


class TimezonePattern(BasePattern):
    semantic_type = "timezone"
    ''' 
    regex_patterns = [
        # UTC/GMT offset formats (support both hyphen - and minus sign −)
        r'^UTC[+−-]\d{1,2}:\d{2}$',  # UTC+05:30, UTC-08:00, UTC−08:00
        r'^GMT[+−-]\d{1,2}:\d{2}$',  # GMT+05:30, GMT-08:00
        r'^[+−-]\d{1,2}:\d{2}$',  # +05:30, -08:00, −08:00
        r'^UTC[+−-]\d{1,2}$',  # UTC+5, UTC-8, UTC+10
        r'^GMT[+−-]\d{1,2}$',  # GMT+5, GMT-8
        
        # Abbreviation with offset or in parentheses (most common in dataset)
        r'^[A-Z]{2,5}\s*UTC[+−-]?\d{1,2}:?\d{0,2}$',  # IST UTC05:30, IST UTC+05:30, AEST UTC10, UTC08:00, UTC03:00
        r'^[A-Z]{2,5}\s*\(UTC[+−-]\d{1,2}:?\d{0,2}\)$',  # IST (UTC+05:30)
        r'^[A-Z]{2,5}\s*\(UTC[+−-]\d{1,2}\)$',  # PST (UTC-08)
        r'^[A-Z]{2,5}\s*UTC[+−-]\d{1,2}:\d{2}$',  # IST UTC+05:30
        r'^[A-Z]{2,5}\s*UTC[+−-]\d{1,2}$',  # AEST UTC+10, PST UTC-08
        r'^[A-Z]{2,5}\s*[+−-]\d{1,2}:\d{2}$',  # EST -05:00
        r'^[A-Z]{2,5}\s*[+−-]\d{1,2}(?:\.\d+)?$',  # IST +05.5, IST 05.5, MSK +03, MSK 03, CET 01
        r'^[A-Z]{2,5}\s+\d{1,2}$',  # NZST 12, SGT 08 (just number without sign)
        r'^UTC\d{1,2}:\d{2}$',  # UTC08:00, UTC03:00 (cleaned format without +)
        
        # Named timezones
        
        # IANA timezone database format
        r'^[A-Z][a-z]+/[A-Z][a-z_]+$',  # Asia/Kolkata, America/New_York
        r'^[A-Z][a-z]+/[A-Z][a-z_]+/[A-Z][a-z_]+$',  # America/Argentina/Buenos_Aires
        
        # Windows timezone names
        r'^[A-Z][A-Za-z\s]+Standard Time$',  # India Standard Time, Pacific Standard Time
        r'^[A-Z][A-Za-z\s]+Daylight Time$',
        
        # Alternative formats
        r'^UTC$',  # UTC
        r'^GMT$',  # GMT
        r'^Z$',  # Zulu time
        
        # Offset with timezone name
        r'^[A-Z]{3,5}\s*[+-]\d{1,2}:?\d{2}$',  # IST+05:30
        r'^\([A-Z]{3,5}\)\s*UTC[+-]\d{1,2}:?\d{2}$',  # (IST) UTC+05:30
        
        # Military time zones
        
        # Full descriptive formats
        r'^UTC\s*[+-]\s*\d{1,2}:\d{2}\s*\([A-Z][a-z]+/[A-Z][a-z_]+\)$',  # UTC + 05:30 (Asia/Kolkata)
    ]
    '''
    regex_patterns = [

    # ===============================
    # Strict UTC/GMT Offset Formats
    # ===============================

     r'^UTC[+−-]\d{1,2}:\d{2}$',        # UTC+05:30
    r'^GMT[+−-]\d{1,2}:\d{2}$',        # GMT-08:00
    r'^UTC[+−-]\d{1,2}$',              # UTC+5
    r'^GMT[+−-]\d{1,2}$',              # GMT-8
    r'^[+−-]\d{1,2}:\d{2}$',           # +05:30
    r'^UTC\d{1,2}:\d{2}$',             # UTC08:00

    # ====================================
    # Abbreviation WITH Explicit UTC Only
    # ====================================

    r'^[A-Z]{2,5}\s+UTC[+−-]\d{1,2}:\d{2}$',   # IST UTC+05:30
    r'^[A-Z]{2,5}\s+\(UTC[+−-]\d{1,2}:\d{2}\)$',  # IST (UTC+05:30)
    r'^[A-Z]{2,5}\s+UTC[+−-]\d{1,2}$',          # PST UTC-08

    # ====================================
    # IANA Timezone Format (Safe)
    # ====================================

    r'^[A-Z][a-z]+/[A-Z][a-z_]+$',                       # Asia/Kolkata
    r'^[A-Z][a-z]+/[A-Z][a-z_]+/[A-Z][a-z_]+$',          # America/Argentina/Buenos_Aires

    # ====================================
    # Windows Timezone Names
    # ====================================

    r'^[A-Z][A-Za-z\s]+Standard Time$',     # India Standard Time
    r'^[A-Z][A-Za-z\s]+Daylight Time$',     # Pacific Daylight Time

    # ====================================
    # Simple Named Zones
    # ====================================

    r'^UTC$',
    r'^GMT$',
    r'^Z$',

    # ====================================
    # Descriptive UTC with IANA
    # ====================================

    r'^UTC\s*[+-]\s*\d{1,2}:\d{2}\s*\([A-Z][a-z]+/[A-Z][a-z_]+\)$',
    ]

    def __init__(self):
        super().__init__()
        self.detected_format = None
    
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
            value_str = str(value).strip()
            
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def _parse_offset(self, value_str):
        """Parse timezone offset to minutes"""
        val_str = value_str.strip()
        
        # Replace minus sign (−) with hyphen (-)
        val_str = val_str.replace('−', '-')
        val_str_upper = val_str.upper()
        
        # Handle parentheses formats: IST (UTC+05:30) or PST (UTC-08)
        paren_match = re.match(r'^[A-Z]{2,5}\s*\(UTC([+−-]\d{1,2}:?\d{0,2})\)$', val_str)
        if paren_match:
            val_str_upper = 'UTC' + paren_match.group(1)
        # Handle space-separated formats: IST UTC+05:30, IST UTC05:30, EST -05:00, AEST UTC10
        elif re.match(r'^[A-Z]{2,5}\s+(UTC)?([+−-]?\d{1,2}:?\d{0,2})$', val_str):
            space_match = re.match(r'^[A-Z]{2,5}\s+(UTC)?([+−-]?\d{1,2}:?\d{0,2})$', val_str)
            offset_str = space_match.group(2).replace('−', '-')
            # If no sign, assume it's UTC+XX (cleaned format)
            if not offset_str.startswith('+') and not offset_str.startswith('-'):
                offset_str = '+' + offset_str
            val_str_upper = 'UTC' + offset_str
        
        # Remove UTC/GMT prefix
        val_str_upper = re.sub(r'^(UTC|GMT)\s*', '', val_str_upper)
        
        # Handle special cases
        if val_str_upper in ['', 'Z']:
            return 0
        
        # Parse offset with colon: +05:30, -08:00
        offset_match = re.match(r'([+−-])(\d{1,2}):(\d{2})', val_str_upper)
        if offset_match:
            sign = 1 if offset_match.group(1) == '+' else -1
            hours = int(offset_match.group(2))
            minutes = int(offset_match.group(3))
            return sign * (hours * 60 + minutes)
        
        # Parse decimal hours: +05.5, +5.75, or just 05.5 (no sign)
        decimal_match = re.match(r'([+−-]?)(\d{1,2})\.(\d+)$', val_str_upper)
        if decimal_match:
            sign = -1 if decimal_match.group(1) == '-' else 1
            hours = int(decimal_match.group(2))
            decimal_part = float('0.' + decimal_match.group(3))
            total_minutes = int((hours + decimal_part) * 60)
            return sign * total_minutes
        
        # Try simple hour offset: +5, -8, or just 5 (no sign)
        simple_match = re.match(r'([+−-]?)(\d{1,2})$', val_str_upper)
        if simple_match:
            sign = -1 if simple_match.group(1) == '-' else 1
            hours = int(simple_match.group(2))
            return sign * hours * 60
        
        # Named timezone abbreviations to offset (common ones)
        timezone_offsets = {
            'UTC': 0, 'GMT': 0, 'Z': 0,
            'EST': -300, 'EDT': -240,  # Eastern
            'CST': -360, 'CDT': -300,  # Central
            'MST': -420, 'MDT': -360, 'MSK': 180,  # Mountain, Moscow
            'PST': -480, 'PDT': -420,  # Pacific
            'IST': 330,  # India
            'JST': 540,  # Japan
            'AEST': 600, 'AEDT': 660,  # Australian Eastern
            'CET': 60, 'CEST': 120,  # Central European
            'BST': 60,  # British Summer
            'NZST': 720, 'NZDT': 780,  # New Zealand
            'BRT': -180, 'BRST': -120,  # Brazil
            'NPT': 345,  # Nepal
            'SGT': 480,  # Singapore
        }
        
        if val_str_upper in timezone_offsets:
            return timezone_offsets[val_str_upper]
        
        return None
    
    def _format_offset(self, minutes, target_format):
        """Format offset minutes to timezone string"""
        if minutes is None or pd.isna(minutes):
            return np.nan
        
        try:
            hours = int(minutes // 60)
            mins = abs(int(minutes % 60))
            sign = '+' if minutes >= 0 else '-'
            abs_hours = abs(hours)
            
            if target_format == 'utc_colon':
                return f"UTC{sign}{abs_hours:02d}:{mins:02d}"
            elif target_format == 'utc_no_colon':
                if mins == 0:
                    return f"UTC{sign}{abs_hours}"
                return f"UTC{sign}{abs_hours:02d}{mins:02d}"
            elif target_format == 'gmt_colon':
                return f"GMT{sign}{abs_hours:02d}:{mins:02d}"
            elif target_format == 'gmt_no_colon':
                if mins == 0:
                    return f"GMT{sign}{abs_hours}"
                return f"GMT{sign}{abs_hours:02d}{mins:02d}"
            elif target_format == 'offset_colon':
                return f"{sign}{abs_hours:02d}:{mins:02d}"
            elif target_format == 'offset_no_colon':
                if mins == 0:
                    return f"{sign}{abs_hours}"
                return f"{sign}{abs_hours:02d}{mins:02d}"
            else:
                # Default to UTC with colon
                return f"UTC{sign}{abs_hours:02d}:{mins:02d}"
        except:
            return np.nan
    
    def normalize(self, values):
        """Keep timezone values in original format (no normalization)"""
        # Return original values without modification
        # The user wants to keep the same format
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values)

