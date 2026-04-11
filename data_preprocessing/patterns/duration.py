import re
import pandas as pd
import numpy as np
from .base import BasePattern


class DurationPattern(BasePattern):
    semantic_type = "duration"
    
    regex_patterns = [
        # Years
        r'^\d+\s*(year|years|yr|yrs)s?$',
        # Months
        r'^\d+\s*(month|months|mon|mons|mo)s?$',
        # Weeks
        r'^\d+\s*(week|weeks|wk|wks)s?$',
        # Days
        r'^\d+\s*(day|days)s?$',
        r'^\d+d$',  # Standalone d like "7d"
        # Hours
        r'^\d+\s*(hour|hours|hr|hrs)s?$',
        r'^\d+h$',  # Standalone h like "3h"
        r'^\d+\.\d+\s*(hour|hours|hr|hrs)s?$',
        # Minutes
        r'^\d+\s*(minute|minutes|min|mins)s?$',
        r'^\d+m$',  # Standalone m like "90m"
        r'^\d+\.\d+\s*(minute|minutes|min|mins)s?$',
        # Seconds
        r'^\d+\s*(second|seconds|sec|secs)s?$',
        r'^\d+s$',  # Standalone s like "30s"
        r'^\d+\.\d+\s*(second|seconds|sec|secs)s?$',
        # Milliseconds
        r'^\d+\s*(millisecond|milliseconds|ms|msec|msecs)$',
        r'^\d+\.\d+\s*(millisecond|milliseconds|ms|msec|msecs)$',
        # Time formats
        r'^\d+:\d+:\d+$',  # HH:MM:SS
        r'^\d+:\d+$',      # HH:MM
        r'^\d+:\d+:\d+\.\d+$',  # HH:MM:SS.mmm
        # ISO 8601 duration format
        r'^P(\d+Y)?(\d+M)?(\d+W)?(\d+D)?(T(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?$',
        # Decimal formats
        r'^\d+\.\d+\s*(hours|days|minutes|weeks|seconds)$',
        # Compound formats (spaces optional)
        r'^\d+h\s*\d+m$',  # 2h 30m
        r'^\d+h\s*\d+m\s*\d+s$',  # 2h 30m 45s
        r'^\d+m\s*\d+s$',  # 10m 30s
        r'^\d+d\s*\d+h$',  # 3d 5h
        r'^\d+w\s*\d+d$',  # 2w 3d
        r'^\d+\s*hrs?\s*\d+\s*mins?$',  # 2 hrs 30 mins
        r'^\d+\s*days?\s*\d+\s*hrs?$',  # 3 days 5 hrs
        r'^\d+\s*mins?\s*\d+\s*secs?$',  # 30 mins 15 secs
        r'^\d+\s*(days?|d)\s+\d+\s*(hours?|hrs?|h)$',  # 2 days 3 hours
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_unit = None  # Will store the most common unit
    
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
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def detect_format(self, value_str):
        """Detect the format/unit of a duration value"""
        val_lower = value_str.lower().strip()
        
        # Check for time format HH:MM:SS or HH:MM - treat as hours
        if re.match(r'^\d+:\d+(?::\d+)?(?:\.\d+)?$', val_lower):
            return 'hours'
        
        # Check for compound formats - classify by primary unit
        if re.match(r'^\d+\s*h\s*\d+\s*m(?:\s*\d+\s*s)?$', val_lower):
            return 'hours'
        if re.match(r'^\d+\s*m\s*\d+\s*s$', val_lower):
            return 'mins'
        if re.match(r'^\d+\s*hrs?\s*\d+\s*mins?(?:\s*\d+\s*secs?)?$', val_lower):
            return 'hours'
        if re.match(r'^\d+\s*mins?\s*\d+\s*secs?$', val_lower):
            return 'mins'
        if re.match(r'^\d+\s*d(?:ays?)?\s*\d+\s*h(?:rs?)?$', val_lower):
            return 'days'
        if re.match(r'^\d+\s*w(?:eeks?)?\s*\d+\s*d(?:ays?)?$', val_lower):
            return 'weeks'
        
        # Check for specific units (order matters - check more specific patterns first)
        if 'millisecond' in val_lower or re.search(r'\bms\b', val_lower) or 'msec' in val_lower:
            return 'ms'
        elif 'year' in val_lower or re.search(r'\byrs?\b', val_lower):
            return 'years'
        elif 'month' in val_lower or re.search(r'\bmon(s|ths?)?\b', val_lower):
            return 'months'
        elif 'week' in val_lower or re.search(r'\bwks?\b', val_lower):
            return 'weeks'
        elif 'day' in val_lower or re.match(r'^\d+d$', val_lower):
            return 'days'
        elif 'hour' in val_lower or re.search(r'\bh(rs?)?\b', val_lower) or re.match(r'^\d+h$', val_lower) or re.match(r'^\d+hr$', val_lower):
            return 'hours'
        elif 'min' in val_lower or re.match(r'^\d+m$', val_lower):
            return 'mins'
        elif 'sec' in val_lower or re.search(r'\bsecs?\b', val_lower) or re.match(r'^\d+s$', val_lower):
            return 'secs'
        else:
            return 'unknown'
    
    def normalize(self, values):
        from collections import Counter
        
        # Detect format for each non-null value
        format_counts = Counter()
        for val in values:
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if val_str.lower() in ['null', 'nan', 'none', '']:
                continue
            fmt = self.detect_format(val_str)
            if fmt != 'unknown':
                format_counts[fmt] += 1
        
        # Find most common format
        if not format_counts:
            return values
        
        most_common_format = format_counts.most_common(1)[0][0]
        
        def parse_to_seconds(val):
            """Convert duration value to seconds"""
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip().lower()
                
                # Handle compound formats
                # 2h 30m or 2h 30m 45s
                compound_match = re.match(r'^(\d+)\s*h(?:rs?)?\s*(\d+)\s*m(?:ins?)?(?:\s*(\d+)\s*s(?:ecs?)?)?$', val_str)
                if compound_match:
                    hours = int(compound_match.group(1))
                    minutes = int(compound_match.group(2))
                    seconds = int(compound_match.group(3) or 0)
                    return hours * 3600 + minutes * 60 + seconds
                
                # 10m 30s
                min_sec_match = re.match(r'^(\d+)\s*m(?:ins?)?\s*(\d+)\s*s(?:ecs?)?$', val_str)
                if min_sec_match:
                    minutes = int(min_sec_match.group(1))
                    seconds = int(min_sec_match.group(2))
                    return minutes * 60 + seconds
                
                # 3d 5h
                day_hour_match = re.match(r'^(\d+)\s*d(?:ays?)?\s*(\d+)\s*h(?:rs?)?$', val_str)
                if day_hour_match:
                    days = int(day_hour_match.group(1))
                    hours = int(day_hour_match.group(2))
                    return days * 24 * 3600 + hours * 3600
                
                # 2w 3d
                week_day_match = re.match(r'^(\d+)\s*w(?:eeks?)?\s*(\d+)\s*d(?:ays?)?$', val_str)
                if week_day_match:
                    weeks = int(week_day_match.group(1))
                    days = int(week_day_match.group(2))
                    return weeks * 7 * 24 * 3600 + days * 24 * 3600
                
                # Handle HH:MM:SS or HH:MM format
                time_match = re.match(r'^(\d+):(\d+)(?::(\d+))?(?:\.(\d+))?$', val_str)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = int(time_match.group(3) or 0)
                    milliseconds = int(time_match.group(4) or 0)
                    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
                
                # Extract numeric value
                num_match = re.search(r'(\d+(?:\.\d+)?)', val_str)
                if not num_match:
                    return np.nan
                
                value_num = float(num_match.group(1))
                
                # Determine unit and convert to seconds (check for more specific patterns first)
                if 'millisecond' in val_str or 'ms' in val_str or 'msec' in val_str:
                    return value_num / 1000
                elif 'year' in val_str or re.search(r'\byrs?\b', val_str):
                    return value_num * 365 * 24 * 3600
                elif 'month' in val_str or 'mon' in val_str:
                    return value_num * 30 * 24 * 3600
                elif 'week' in val_str or re.search(r'\bwks?\b', val_str):
                    return value_num * 7 * 24 * 3600
                elif 'day' in val_str or re.match(r'^\d+d$', val_str):
                    return value_num * 24 * 3600
                elif 'hour' in val_str or re.search(r'\bh(rs?)?\b', val_str) or re.match(r'^\d+h$', val_str) or re.match(r'^\d+hr$', val_str):
                    return value_num * 3600
                elif 'min' in val_str or re.match(r'^\d+m$', val_str):
                    return value_num * 60
                elif 'sec' in val_str or re.search(r'\bsecs?\b', val_str) or re.match(r'^\d+s$', val_str):
                    return value_num
                else:
                    return value_num
            except:
                return np.nan
        
        def format_duration(seconds, target_format):
            """Convert seconds to target format"""
            if pd.isna(seconds):
                return np.nan
            
            try:
                if target_format == 'years':
                    value = seconds / (365 * 24 * 3600)
                    return int(value)
                elif target_format == 'months':
                    value = seconds / (30 * 24 * 3600)
                    return int(value)
                elif target_format == 'weeks':
                    value = seconds / (7 * 24 * 3600)
                    return int(value)
                elif target_format == 'days':
                    value = seconds / (24 * 3600)
                    return int(value)
                elif target_format == 'hours':
                    value = seconds / 3600
                    return int(value)
                elif target_format == 'mins':
                    value = seconds / 60
                    return int(value)
                elif target_format == 'secs':
                    return int(seconds)
                elif target_format == 'ms':
                    value = seconds * 1000
                    return int(value)
                else:
                    return int(seconds)
            except:
                return np.nan
        
        # Convert all values to seconds first, then to target format
        seconds_values = values.apply(parse_to_seconds)
        normalized_values = seconds_values.apply(lambda x: format_duration(x, most_common_format))
        
        # Store detected unit for column renaming
        unit_map = {
            'years': 'years',
            'months': 'months', 
            'weeks': 'weeks',
            'days': 'days',
            'hours': 'hrs',
            'mins': 'mins',
            'secs': 'secs',
            'ms': 'ms'
        }
        self.detected_unit = unit_map.get(most_common_format, 'duration')
        
        return normalized_values
