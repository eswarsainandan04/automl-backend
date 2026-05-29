import re
import pandas as pd
import numpy as np
from .base import BasePattern


class SpeedPattern(BasePattern):
    semantic_type = "speed"
    detected_unit = None
    
    regex_patterns = [
        # Metric - SPEED ONLY
        r'^\d+(?:\.\d+)?\s*(?:km/h|kmh|kph|kilometers?\s*per\s*hour)$',
        r'^\d+(?:\.\d+)?\s*(?:m/s|mps|meters?\s*per\s*second)$',
        r'^\d+(?:\.\d+)?\s*(?:cm/s|centimeters?\s*per\s*second)$',
        # Imperial - SPEED ONLY
        r'^\d+(?:\.\d+)?\s*(?:mph|mi/h|miles?\s*per\s*hour)$',
        r'^\d+(?:\.\d+)?\s*(?:ft/s|fps|feet\s*per\s*second)$',
        r'^\d+(?:\.\d+)?\s*(?:in/s|inches?\s*per\s*second)$',
        # Nautical - SPEED ONLY
        r'^\d+(?:\.\d+)?\s*(?:knot|knots|kt|kts)$',
        # Aviation
        r'^\d+(?:\.\d+)?\s*(?:mach)$',
    ]
    
    CONVERSIONS = {
        'km/h': 1, 'kmh': 1, 'kph': 1, 'kilometers per hour': 1, 'kilometer per hour': 1,
        'm/s': 3.6, 'mps': 3.6, 'meters per second': 3.6, 'meter per second': 3.6,
        'cm/s': 0.036, 'centimeters per second': 0.036, 'centimeter per second': 0.036,
        'mph': 1.60934, 'mi/h': 1.60934, 'miles per hour': 1.60934, 'mile per hour': 1.60934,
        'ft/s': 1.09728, 'fps': 1.09728, 'feet per second': 1.09728,
        'in/s': 0.09144, 'inches per second': 0.09144, 'inch per second': 0.09144,
        'knot': 1.852, 'knots': 1.852, 'kt': 1.852, 'kts': 1.852,
        'mach': 1234.8,  # Approximate at sea level
    }
    
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
            
            match = re.match(r'^(\d+(?:\.\d+)?)\s*(.+)$', val_str)
            if match:
                number = float(match.group(1))
                unit = match.group(2).strip()
                
                # Normalize unit (remove spaces, lowercase)
                unit = unit.replace(' ', ' ').strip()  # Normalize multiple spaces
                
                if unit in self.CONVERSIONS:
                    parsed_data.append((number, unit))
                    unit_counts[unit] = unit_counts.get(unit, 0) + 1
                else:
                    parsed_data.append((np.nan, None))
            else:
                parsed_data.append((np.nan, None))
        
        if unit_counts:
            most_common_unit = max(unit_counts, key=unit_counts.get)
            self.detected_unit = most_common_unit
        else:
            self.detected_unit = 'km/h'
            most_common_unit = 'km/h'
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            value_in_kmh = number * self.CONVERSIONS[unit]
            value_in_target = value_in_kmh / target_conversion
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 1) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
