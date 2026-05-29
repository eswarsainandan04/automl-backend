import re
import pandas as pd
import numpy as np
from .base import BasePattern


class LatitudePattern(BasePattern):
    semantic_type = "latitude"
    
    regex_patterns = [
        # Decimal degrees (DD)
        r'^-?\d{1,2}\.\d+$',  # 17.3850, -23.5505
        r'^-?\d{1,2}\.\d+°$',  # 17.3850°
        r'^-?\d{1,2}\.*\d+°?\s*[NS]$',  # 17.3850 N, 17.3850° S
        r'^\d{6}\s*[NS]$',  # 513026 N (DMS without separators after cleaning)
        r'^[NS]\d{6}$',  # N513026
        
        # Degrees Minutes Seconds (DMS)
        r'^\d{1,2}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"?\s*[NS]$',  # 17°23'06"N
        r'^\d{1,2}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"\s*[NS]$',
        r'^\d{1,2}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"[NS]$',
        r'^\d{1,2}°\d{1,2}\'\d{1,2}(\.\d+)?"[NS]$',
        
        # Degrees Decimal Minutes (DDM)
        r'^\d{1,2}°\s*\d{1,2}\.\d+\'\s*[NS]$',  # 17°23.1'N
        r'^\d{1,2}°\d{1,2}\.\d+\'\s*[NS]$',
        
        # With direction
        r'^[NS]\s*\d{1,2}\.\d+$',  # N 17.3850
        r'^[NS]\d{1,2}\.\d+$',
        r'^\d{1,2}\.\d+\s*[NS]$',  # 17.3850 N
        r'^\d{1,2}\.\d+[NS]$',
        
        # Various formats
        r'^\d{1,2}°\s*\d{1,2}\'\s*[NS]$',  # 17°23'N
        r'^\d{1,2}°\d{1,2}\'[NS]$',
        r'^\d{1,2}\s+\d{1,2}\s+\d{1,2}\s*[NS]$',  # 17 23 06 N
        r'^\d{1,2}-\d{1,2}-\d{1,2}\s*[NS]$',  # 17-23-06N
        
        # Latitude in various notations
        r'^LAT:\s*-?\d{1,2}\.\d+$',  # LAT: 17.3850
        r'^Lat\s*-?\d{1,2}\.\d+$',
        r'^latitude:\s*-?\d{1,2}\.\d+$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = None
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        # Check if column is named "latitude", "lat" (case-insensitive)
        column_name = ""
        if hasattr(values, 'name') and values.name:
            column_name = str(values.name).lower()
        
        is_latitude_column = any(keyword in column_name for keyword in ['latitude', 'lat'])
        
        matched = 0
        total = 0
        has_direction = 0  # Count values with N/S direction
        
        for value in values:
            # Skip null/nan values
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.lower() in ['null', 'nan', 'none', '']:
                continue
                
            total += 1
            value_str = str(value).strip()
            
            # Check if has directional indicators (N/S)
            if re.search(r'[NS]', value_str, re.IGNORECASE):
                has_direction += 1
            
            # Check if it's a valid latitude range (-90 to 90)
            try:
                # Try to extract numeric value
                numeric_val = self._extract_latitude_value(value_str)
                if numeric_val is not None and -90 <= numeric_val <= 90:
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
        
        # ONLY accept as latitude if:
        # 1. Column is named latitude/lat OR
        # 2. At least 50% of values have directional indicators (N/S)
        if not is_latitude_column and has_direction < total * 0.5:
            return 0.0  # Reject - likely just regular floats
        
        return confidence
        return matched / total
    
    def _extract_latitude_value(self, value_str):
        """Extract numeric latitude value from string"""
        val_str = value_str.strip()
        
        # Remove LAT: prefix
        val_str = re.sub(r'^(LAT|LATITUDE):\s*', '', val_str, flags=re.IGNORECASE)
        
        # Try DMS without separators (cleaned format): 513026 N → 51°30'26" N
        dms_nosep_match = re.match(r'^([NS])?(\d{2})(\d{2})(\d{2})\s*([NS])?$', val_str, re.IGNORECASE)
        if dms_nosep_match:
            degrees = float(dms_nosep_match.group(2))
            minutes = float(dms_nosep_match.group(3))
            seconds = float(dms_nosep_match.group(4))
            direction = (dms_nosep_match.group(1) or dms_nosep_match.group(5) or '').upper()
            
            decimal = degrees + minutes / 60 + seconds / 3600
            if direction == 'S' and decimal > 0:
                decimal = -decimal
            return decimal
        
        # Try DMS format with separators
        dms_match = re.match(r'(\d{1,2})°?\s*(\d{1,2})[\'"]\s*(\d{1,2}(?:\.\d+)?)["\']?\s*([NS])?', val_str, re.IGNORECASE)
        if dms_match:
            degrees = float(dms_match.group(1))
            minutes = float(dms_match.group(2))
            seconds = float(dms_match.group(3) or 0)
            direction = dms_match.group(4) or ''
            
            decimal = degrees + minutes / 60 + seconds / 3600
            if direction.upper() == 'S':
                decimal = -decimal
            return decimal
        
        # Try DDM format
        ddm_match = re.match(r'(\d{1,2})°?\s*(\d{1,2}\.\d+)[\'"]\s*([NS])?', val_str, re.IGNORECASE)
        if ddm_match:
            degrees = float(ddm_match.group(1))
            minutes = float(ddm_match.group(2))
            direction = ddm_match.group(3) or ''
            
            decimal = degrees + minutes / 60
            if direction.upper() == 'S':
                decimal = -decimal
            return decimal
        
        # Try decimal with direction (40.7128° N or 40.7128 N)
        dir_match = re.match(r'([NS])?\s*(-?\d{1,2}\.\d+)°?\s*([NS])?', val_str, re.IGNORECASE)
        if dir_match:
            value = float(dir_match.group(2))
            direction = (dir_match.group(1) or dir_match.group(3) or '').upper()
            if direction == 'S' and value > 0:
                value = -value
            return value
        
        # Try simple decimal
        try:
            return float(val_str.replace('°', ''))
        except:
            pass
        
        return None
        
    def _detect_format(self, values) -> str:
        
        # Try with N/S prefix or suffix
        ns_match = re.match(r'([NS])?\s*(-?\d{1,2}\.\d+)\s*([NS])?', val_str)
        if ns_match:
            value = float(ns_match.group(2))
            direction = ns_match.group(1) or ns_match.group(3) or ''
            if direction == 'S' and value > 0:
                value = -value
            return value
        
        return None
    
    def normalize(self, values):
        """Normalize latitude values to decimal degrees format"""
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
            elif re.search(r'[NS]', val_str, re.IGNORECASE):
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
        decimal_values = values.apply(lambda x: self._extract_latitude_value(str(x)) if pd.notna(x) else np.nan)
        
        # Format based on most common format
        def format_latitude(lat_decimal, target_format):
            if pd.isna(lat_decimal):
                return np.nan
            
            try:
                if target_format == 'decimal':
                    return round(lat_decimal, 6)
                elif target_format == 'dd_with_direction':
                    direction = 'N' if lat_decimal >= 0 else 'S'
                    return f"{abs(lat_decimal):.6f} {direction}"
                elif target_format == 'dms':
                    # Convert to DMS
                    is_negative = lat_decimal < 0
                    lat_abs = abs(lat_decimal)
                    degrees = int(lat_abs)
                    minutes_decimal = (lat_abs - degrees) * 60
                    minutes = int(minutes_decimal)
                    seconds = (minutes_decimal - minutes) * 60
                    direction = 'S' if is_negative else 'N'
                    return f"{degrees}°{minutes}'{seconds:.2f}\"{direction}"
                elif target_format == 'ddm':
                    # Convert to DDM
                    is_negative = lat_decimal < 0
                    lat_abs = abs(lat_decimal)
                    degrees = int(lat_abs)
                    minutes = (lat_abs - degrees) * 60
                    direction = 'S' if is_negative else 'N'
                    return f"{degrees}°{minutes:.4f}'{direction}"
                else:
                    return round(lat_decimal, 6)
            except:
                return np.nan
        
        return decimal_values.apply(lambda x: format_latitude(x, most_common_format))
