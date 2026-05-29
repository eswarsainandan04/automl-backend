import re
import pandas as pd
import numpy as np
from .base import BasePattern


class AnglePattern(BasePattern):
    semantic_type = "angle"
    detected_unit = None
    
    regex_patterns = [
        r'^\d+(?:\.\d+)?\s*°$',  # degrees with symbol
        r'^\d+(?:\.\d+)?\s*(?:deg|degree|degrees)$',  # degrees
        r'^\d+(?:\.\d+)?\s*(?:rad|radian|radians)$',  # radians
        r'^\d+(?:\.\d+)?\s*(?:grad|gradian|gradians)$',  # gradians
        r'^\d+(?:\.\d+)?\s*(?:p|pi|π)$',  # pi radians (treat as radians)
        r'^\d+(?:\.\d+)?$',  # plain numbers (assume degrees)
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        matched = 0
        total = 0
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.lower() in ['null', 'nan', 'none', '']:
                continue
                
            total += 1
            value_str = str(value).strip().lower()
            
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def normalize(self, values):
        unit_counts = {}
        parsed_data = []
        
        for val in values:
            if pd.isna(val):
                parsed_data.append((np.nan, None))
                continue
                
            val_str = str(val).strip().lower()
            if val_str in ['null', 'nan', 'none', '']:
                parsed_data.append((np.nan, None))
                continue
            
            match = re.match(r'^(\d+(?:\.\d+)?)\s*(.*)$', val_str)
            if match:
                number = float(match.group(1))
                unit = match.group(2).strip()
                
                # If no unit specified, assume degrees
                if not unit:
                    unit = 'deg'
                # Normalize unit names
                elif unit == '°' or unit in ['deg', 'degree', 'degrees']:
                    unit = 'deg'
                elif unit in ['rad', 'radian', 'radians']:
                    unit = 'rad'
                elif unit in ['p', 'pi', 'π']:  # pi radians
                    unit = 'rad'
                    number = number * np.pi  # Convert from multiples of pi to radians
                elif unit in ['grad', 'gradian', 'gradians']:
                    unit = 'grad'
                else:
                    parsed_data.append((np.nan, None))
                    continue
                
                parsed_data.append((number, unit))
                unit_counts[unit] = unit_counts.get(unit, 0) + 1
            else:
                parsed_data.append((np.nan, None))
        
        if unit_counts:
            most_common_unit = max(unit_counts, key=unit_counts.get)
            self.detected_unit = most_common_unit
        else:
            self.detected_unit = 'deg'
            most_common_unit = 'deg'
        
        def convert_angle(data_tuple, target_unit):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            # Convert to degrees first
            if unit == 'deg':
                degrees = number
            elif unit == 'rad':
                degrees = number * (180 / np.pi)
            elif unit == 'grad':
                degrees = number * 0.9
            else:
                return np.nan
            
            # Convert from degrees to target
            if target_unit == 'deg':
                return degrees
            elif target_unit == 'rad':
                return degrees * (np.pi / 180)
            elif target_unit == 'grad':
                return degrees / 0.9
            else:
                return degrees
        
        result = [convert_angle(data, most_common_unit) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
