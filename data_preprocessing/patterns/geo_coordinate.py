import re
import pandas as pd
import numpy as np
from .base import BasePattern


class GeoCoordinatePattern(BasePattern):
    semantic_type = "geo_coordinate"
    
    regex_patterns = [
        # Decimal format: lat, lon
        r'^-?\d{1,2}\.\d+\s*,\s*-?\d{1,3}\.\d+$',  # 17.3850, 78.4867 (also matches cleaned format without parentheses)
        r'^\(-?\d{1,2}\.\d+°?\s*,\s*-?\d{1,3}\.\d+°?\)$',  # (17.3850, 78.4867) or (17.3850°, 78.4867°)
        r'^\[-?\d{1,2}\.\d+°?\s*,\s*-?\d{1,3}\.\d+°?\]$',  # [17.3850, 78.4867]
        r'^-?\d{1,2}\.\d+\s*/\s*-?\d{1,3}\.\d+$',  # 17.3850 / 78.4867
        r'^-?\d{1,2}\.\d+\s*;\s*-?\d{1,3}\.\d+$',  # 17.3850; 78.4867
        r'^-?\d{1,2}\.\d+\s+-?\d{1,3}\.\d+$',  # 17.3850 78.4867, -23.5505 -46.6333
        
        # With degree symbols
        r'^\d{1,2}\.\d+°?\s*[NS]?\s*,\s*\d{1,3}\.\d+°?\s*[EW]?$',  # 17.3850° N, 78.4867° E (cleaned too)
        r'^\d{1,2}\.\d+°?\s*[NS]\s*,\s*\d{1,3}\.\d+°?\s*[EW]$',  # 17.3850 N, 78.4867 E
        
        # With labels
        r'^geo:lat-?\d{1,2}\.\d+;lon-?\d{1,3}\.\d+$',  # geo:lat48.8566;lon2.3522 (cleaned, = removed)
        r'^geo:lat=?-?\d{1,2}\.\d+;lon=?-?\d{1,3}\.\d+$',  # geo:lat=48.8566;lon=2.3522
        r'^lat:\s*-?\d{1,2}\.\d+\s*,?\s*lon:\s*-?\d{1,3}\.\d+$',  # lat: 17.3850, lon: 78.4867
        r'^latitude:\s*-?\d{1,2}\.\d+\s*,?\s*longitude:\s*-?\d{1,3}\.\d+$',
        r'^\d{1,2}\.\d+\s*[NS]\s*,\s*\d{1,3}\.\d+\s*[EW]$',  # 17.3850N, 78.4867E
        
        # DMS format (with or without spaces)
        r'^\d{6}\s*[NS]\s+\d{5,6}\s*[EW]$',  # 513026 N 00739 W (cleaned, no separators)
        r'^\d{1,2}°\d{1,2}[\'\"]\d{1,2}(\.\d+)?[\"\']\s*[NS]\s+\d{1,3}°\d{1,2}[\'\"]\d{1,2}(\.\d+)?[\"\']\s*[EW]$',  # 51°30'26" N 0°07'39" W
        r'^\d{1,2}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"\s*[NS]\s*,\s*\d{1,3}°\s*\d{1,2}\'\s*\d{1,2}(\.\d+)?"\s*[EW]$',
        r'^\d{1,2}°\s*\d{1,2}\'\s*[NS]\s*,\s*\d{1,3}°\s*\d{1,2}\'\s*[EW]$',  # 27°10' N, 78°29' E
        
        # Google Maps style
        r'^@-?\d{1,2}\.\d+,-?\d{1,3}\.\d+$',  # @17.3850,78.4867
        
        # With degree symbol
        r'^\d{1,2}\.\d+°\s*[NS]\s*,\s*\d{1,3}\.\d+°\s*[EW]$',  # 17.3850°N, 78.4867°E
        
        # Geohash or alternative formats
        r'^[NS]\s*\d{1,2}\.\d+\s*,\s*[EW]\s*\d{1,3}\.\d+$',  # N 17.3850, E 78.4867
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = None
        self.latitude_values = None
        self.longitude_values = None
    
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
            
            # Check if it contains coordinate separator
            if any(sep in value_str for sep in [',', '/', ';', 'lat', 'lon']):
                # Try to extract lat/lon values
                coords = self._extract_coordinates(value_str)
                if coords is not None:
                    lat, lon = coords
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        matched += 1
        
        if total == 0:
            return 0.0
        return matched / total
    
    def _extract_coordinates(self, value_str):
        """Extract latitude and longitude from coordinate string"""
        val_str = value_str.strip()
        
        # Remove brackets/parentheses
        val_str = re.sub(r'[\[\]\(\)\{\}@]', '', val_str)
        
        # Try labeled format (lat:, lon:)
        labeled_match = re.search(r'lat(?:itude)?:\s*(-?\d{1,2}\.\d+).*lon(?:gitude)?:\s*(-?\d{1,3}\.\d+)', val_str, re.IGNORECASE)
        if labeled_match:
            try:
                return float(labeled_match.group(1)), float(labeled_match.group(2))
            except:
                pass
        
        # Try simple comma/slash/semicolon separated
        for separator in [',', '/', ';']:
            if separator in val_str:
                parts = val_str.split(separator)
                if len(parts) == 2:
                    try:
                        # Clean and extract numbers
                        lat_str = re.sub(r'[^\d.\-NS]', '', parts[0].strip())
                        lon_str = re.sub(r'[^\d.\-EW]', '', parts[1].strip())
                        
                        # Handle N/S/E/W suffixes
                        lat_multiplier = -1 if 'S' in parts[0].upper() else 1
                        lon_multiplier = -1 if 'W' in parts[1].upper() else 1
                        
                        lat = float(re.sub(r'[NS]', '', lat_str, flags=re.IGNORECASE)) * lat_multiplier
                        lon = float(re.sub(r'[EW]', '', lon_str, flags=re.IGNORECASE)) * lon_multiplier
                        
                        return lat, lon
                    except:
                        pass
        
        # Try space separated
        space_match = re.match(r'(-?\d{1,2}\.\d+)\s+(-?\d{1,3}\.\d+)', val_str)
        if space_match:
            try:
                return float(space_match.group(1)), float(space_match.group(2))
            except:
                pass
        
        return None
    
    def normalize(self, values):
        """Normalize geo coordinates and store lat/lon separately"""
        from collections import Counter
        
        # Extract coordinates first
        coord_values = values.apply(lambda x: self._extract_coordinates(str(x)) if pd.notna(x) else None)
        
        # Store latitude and longitude separately for column splitting
        self.latitude_values = coord_values.apply(lambda x: x[0] if x is not None else np.nan)
        self.longitude_values = coord_values.apply(lambda x: x[1] if x is not None else np.nan)
        
        # Detect format for display purposes (not used for splitting)
        format_counts = Counter()
        for val in values:
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if val_str.lower() in ['null', 'nan', 'none', '']:
                continue
            
            # Determine format type
            if re.search(r'lat.*lon', val_str, re.IGNORECASE):
                format_counts['labeled'] += 1
            elif re.search(r'[\[\(\{]', val_str):
                format_counts['bracketed'] += 1
            elif re.search(r'[NS].*[EW]', val_str, re.IGNORECASE):
                format_counts['directional'] += 1
            elif ',' in val_str:
                format_counts['comma'] += 1
            elif '/' in val_str:
                format_counts['slash'] += 1
            elif ';' in val_str:
                format_counts['semicolon'] += 1
            else:
                format_counts['space'] += 1
        
        # Find most common format (default to comma)
        if not format_counts:
            most_common_format = 'comma'
        else:
            most_common_format = format_counts.most_common(1)[0][0]
        
        self.detected_format = most_common_format
        
        # Return normalized format as single string (for metadata display)
        # The actual column splitting happens in column_handler.py
        def format_coordinates(coords, target_format):
            if coords is None or pd.isna(coords):
                return np.nan
            
            try:
                lat, lon = coords
                
                if target_format == 'comma':
                    return f"{lat:.6f}, {lon:.6f}"
                elif target_format == 'bracketed':
                    return f"({lat:.6f}, {lon:.6f})"
                elif target_format == 'labeled':
                    return f"lat: {lat:.6f}, lon: {lon:.6f}"
                elif target_format == 'directional':
                    lat_dir = 'N' if lat >= 0 else 'S'
                    lon_dir = 'E' if lon >= 0 else 'W'
                    return f"{abs(lat):.6f} {lat_dir}, {abs(lon):.6f} {lon_dir}"
                elif target_format == 'slash':
                    return f"{lat:.6f} / {lon:.6f}"
                elif target_format == 'semicolon':
                    return f"{lat:.6f}; {lon:.6f}"
                elif target_format == 'space':
                    return f"{lat:.6f} {lon:.6f}"
                else:
                    return f"{lat:.6f}, {lon:.6f}"
            except:
                return np.nan
        
        return coord_values.apply(lambda x: format_coordinates(x, most_common_format))
