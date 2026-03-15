import re
import pandas as pd
import numpy as np
from .base import BasePattern


class PowerPattern(BasePattern):
    semantic_type = "power"
    detected_unit = None
    
    regex_patterns = [
        r'^\d+(?:\.\d+)?\s*(?:W|watt|watts)$',
        r'^\d+(?:\.\d+)?\s*(?:kW|kilowatt|kilowatts)$',
        r'^\d+(?:\.\d+)?\s*(?:MW|megawatt|megawatts)$',
        r'^\d+(?:\.\d+)?\s*(?:GW|gigawatt|gigawatts)$',
        r'^\d+(?:\.\d+)?\s*(?:hp|HP|horsepower)$',
    ]
    
    CONVERSIONS = {
        'w': 1, 'watt': 1, 'watts': 1,
        'kw': 1000, 'kilowatt': 1000, 'kilowatts': 1000,
        'mw': 1000000, 'megawatt': 1000000, 'megawatts': 1000000,
        'gw': 1000000000, 'gigawatt': 1000000000, 'gigawatts': 1000000000,
        'hp': 745.7, 'horsepower': 745.7,
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
            self.detected_unit = 'w'
            most_common_unit = 'w'
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            value_in_watts = number * self.CONVERSIONS[unit]
            value_in_target = value_in_watts / target_conversion
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
