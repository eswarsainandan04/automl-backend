import re
import pandas as pd
import numpy as np
from .base import BasePattern


class LongitudePattern(BasePattern):
    semantic_type = "longitude"
    
    regex_patterns = [
        # Decimal degrees (DD)
        r'^-?\d{1,3}\.\d+$',  # 78.4867, -122.4194
        r'^-?\d{1,3}\.\d+°$',  # 78.4867°
        r'^-?\d{1,3}\.\d+°?\s*[EW]$',  # 78.4867 E, 78.4867° W
        r'^\d{5,6}\s*[EW]$',  # 00739 W (DMS without separators after cleaning)
        r'^[EW]\d{5,6}$',  # W00739
        
        # Degrees Minutes Seconds (DMS)
        r'^\d{1,3}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"?\s*[EW]$',  # 78°29'12"E
        r'^\d{1,3}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"\s*[EW]$',
        r'^\d{1,3}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"[EW]$',
        r'^\d{1,3}°\d{1,2}\'\d{1,2}(\.\d+)?"[EW]$',
        
        # Degrees Decimal Minutes (DDM)
        r'^\d{1,3}°\s*\d{1,2}\.\d+\'\s*[EW]$',  # 78°29.2'E
        r'^\d{1,3}°\d{1,2}\.\d+\'\s*[EW]$',
        
        # With direction
        r'^[EW]\s*\d{1,3}\.\d+$',  # E 78.4867
        r'^[EW]\d{1,3}\.\d+$',
        r'^\d{1,3}\.\d+\s*[EW]$',  # 78.4867 E
        r'^\d{1,3}\.\d+[EW]$',
        
        # Various formats
        r'^\d{1,3}°\s*\d{1,2}\'\s*[EW]$',  # 78°29'E
        r'^\d{1,3}°\d{1,2}\'[EW]$',
        r'^\d{1,3}\s+\d{1,2}\s+\d{1,2}\s*[EW]$',  # 78 29 12 E
        r'^\d{1,3}-\d{1,2}-\d{1,2}\s*[EW]$',  # 78-29-12E
        
        # Longitude in various notations
        r'^LON:\s*-?\d{1,3}\.\d+$',  # LON: 78.4867
        r'^Long\s*-?\d{1,3}\.\d+$',
        r'^longitude:\s*-?\d{1,3}\.\d+$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = None
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        # Check if column is named "longitude", "long", "lng" (case-insensitive)
        column_name = ""
        if hasattr(values, 'name') and values.name:
            column_name = str(values.name).lower()
        
        is_longitude_column = any(keyword in column_name for keyword in ['longitude', 'long', 'lng', 'lon'])
        
        matched = 0
        total = 0
        has_direction = 0  # Count values with E/W direction
        
        for value in values:
            # Skip null/nan values
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.lower() in ['null', 'nan', 'none', '']:
                continue
                
            total += 1
            value_str = str(value).strip()
            
            # Check if has directional indicators (E/W)
            if re.search(r'[EW]', value_str, re.IGNORECASE):
                has_direction += 1
            
            # Check if it's a valid longitude range (-180 to 180)
            try:
                # Try to extract numeric value
                numeric_val = self._extract_longitude_value(value_str)
                if numeric_val is not None and -180 <= numeric_val <= 180:
                    # Also check if it matches any pattern
                    for pattern in self.regex_patterns:
                        if re.match(pattern, value_str, re.IGNORECASE):
                            matched += 1
                            break
            except:
                pass
        
        if total == 0:
            return 0.0
        
        confidence = matched / total
        
        # ONLY accept as longitude if:
        # 1. Column is named longitude/long/lng OR
        # 2. At least 50% of values have directional indicators (E/W)
        if not is_longitude_column and has_direction < total * 0.5:
            return 0.0  # Reject - likely just regular floats
        
        return confidence
    
    def _extract_longitude_value(self, value_str):
        """Extract numeric longitude value from string"""
        val_str = value_str.strip()
        
        # Remove LON: prefix
        val_str = re.sub(r'^(LON|LONG|LONGITUDE):\s*', '', val_str, flags=re.IGNORECASE)
        
        # Try DMS without separators (cleaned format): 00739 W → 0°07'39" W
        dms_nosep_match = re.match(r'^([EW])?(\d{2,3})(\d{2})(\d{2})\s*([EW])?$', val_str, re.IGNORECASE)
        if dms_nosep_match:
            degrees = float(dms_nosep_match.group(2))
            minutes = float(dms_nosep_match.group(3))
            seconds = float(dms_nosep_match.group(4))
            direction = (dms_nosep_match.group(1) or dms_nosep_match.group(5) or '').upper()
            
            decimal = degrees + minutes / 60 + seconds / 3600
            if direction == 'W' and decimal > 0:
                decimal = -decimal
            return decimal
        
        # Try DMS format with separators
        dms_match = re.match(r'(\d{1,3})°?\s*(\d{1,2})[\'"]\s*(\d{1,2}(?:\.\d+)?)["\']?\s*([EW])?', val_str, re.IGNORECASE)
        if dms_match:
            degrees = float(dms_match.group(1))
            minutes = float(dms_match.group(2))
            seconds = float(dms_match.group(3) or 0)
            direction = dms_match.group(4) or ''
            
            decimal = degrees + minutes / 60 + seconds / 3600
            if direction.upper() == 'W':
                decimal = -decimal
            return decimal
        
        # Try DDM format
        ddm_match = re.match(r'(\d{1,3})°?\s*(\d{1,2}\.\d+)[\'"]\s*([EW])?', val_str, re.IGNORECASE)
        if ddm_match:
            degrees = float(ddm_match.group(1))
            minutes = float(ddm_match.group(2))
            direction = ddm_match.group(3) or ''
            
            decimal = degrees + minutes / 60
            if direction.upper() == 'W':
                decimal = -decimal
            return decimal
        
        # Try decimal with direction (74.0060° W or 74.0060 W)
        dir_match = re.match(r'([EW])?\s*(-?\d{1,3}\.\d+)°?\s*([EW])?', val_str, re.IGNORECASE)
        if dir_match:
            value = float(dir_match.group(2))
            direction = (dir_match.group(1) or dir_match.group(3) or '').upper()
            if direction == 'W' and value > 0:
                value = -value
            return value
        
        # Try simple decimal
        try:
            return float(val_str.replace('°', ''))
        except:
            pass
        
        return None
    
    def normalize(self, values):
        """Normalize longitude values to decimal degrees format"""
        from collections import Counter
        
        # Detect format for each non-null value
        format_counts = Counter()
        for val in values:
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if val_str.lower() in ['null', 'nan', 'none', '']:
                continue
            
            # Determine format type
            if re.search(r'°.*\'.*"', val_str):
                format_counts['dms'] += 1
            elif re.search(r'°.*\'', val_str):
                format_counts['ddm'] += 1
            elif re.search(r'[EW]', val_str, re.IGNORECASE):
                format_counts['dd_with_direction'] += 1
            else:
                format_counts['decimal'] += 1
        
        # Find most common format (default to decimal)
        if not format_counts:
            most_common_format = 'decimal'
        else:
            most_common_format = format_counts.most_common(1)[0][0]
        
        self.detected_format = most_common_format
        
        # Convert all values to decimal degrees first
        decimal_values = values.apply(lambda x: self._extract_longitude_value(str(x)) if pd.notna(x) else np.nan)
        
        # Format based on most common format
        def format_longitude(lon_decimal, target_format):
            if pd.isna(lon_decimal):
                return np.nan
            
            try:
                if target_format == 'decimal':
                    return round(lon_decimal, 6)
                elif target_format == 'dd_with_direction':
                    direction = 'E' if lon_decimal >= 0 else 'W'
                    return f"{abs(lon_decimal):.6f} {direction}"
                elif target_format == 'dms':
                    # Convert to DMS
                    is_negative = lon_decimal < 0
                    lon_abs = abs(lon_decimal)
                    degrees = int(lon_abs)
                    minutes_decimal = (lon_abs - degrees) * 60
                    minutes = int(minutes_decimal)
                    seconds = (minutes_decimal - minutes) * 60
                    direction = 'W' if is_negative else 'E'
                    return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"
                elif target_format == 'ddm':
                    # Convert to DDM
                    is_negative = lon_decimal < 0
                    lon_abs = abs(lon_decimal)
                    degrees = int(lon_abs)
                    minutes = (lon_abs - degrees) * 60
                    direction = 'W' if is_negative else 'E'
                    return f"{degrees}°{minutes:.4f}'{direction}"
                else:
                    return round(lon_decimal, 6)
            except:
                return np.nan
        
        return decimal_values.apply(lambda x: format_longitude(x, most_common_format))
