import re
import pandas as pd
import numpy as np
from .base import BasePattern


class TemperaturePattern(BasePattern):
    semantic_type = "temperature"
    detected_unit = None
    
    regex_patterns = [
        # Temperature ONLY - with or without degree symbol, allow negative
        r'^-?\d+(?:\.\d+)?\s*(?:°C|C|celsius|centigrade)$',
        r'^-?\d+(?:\.\d+)?\s*(?:°F|F|fahrenheit)$',
        r'^-?\d+(?:\.\d+)?\s*(?:°K|K|kelvin)$',
        r'^-?\d+(?:\.\d+)?\s*(?:°R|R|rankine)$',
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
            value_str = str(value).strip()
            
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
                
            val_str = str(val).strip()
            if val_str.lower() in ['null', 'nan', 'none', '']:
                parsed_data.append((np.nan, None))
                continue
            
            # Extract number and unit (handle negative temperatures)
            match = re.match(r'^(-?\d+(?:\.\d+)?)\s*°?([CFKRcfkr]|celsius|fahrenheit|kelvin|rankine|centigrade)$', val_str, re.IGNORECASE)
            if match:
                number = float(match.group(1))
                unit = match.group(2).upper()
                
                # Normalize unit names
                if unit.lower() in ['celsius', 'centigrade']:
                    unit = 'C'
                elif unit.lower() in ['fahrenheit']:
                    unit = 'F'
                elif unit.lower() in ['kelvin']:
                    unit = 'K'
                elif unit.lower() in ['rankine']:
                    unit = 'R'
                
                parsed_data.append((number, unit))
                unit_counts[unit] = unit_counts.get(unit, 0) + 1
            else:
                parsed_data.append((np.nan, None))
        
        if unit_counts:
            most_common_unit = max(unit_counts, key=unit_counts.get)
            self.detected_unit = most_common_unit
        else:
            self.detected_unit = 'C'
            most_common_unit = 'C'
        
        def convert_temperature(data_tuple, target_unit):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            # Convert to Celsius first
            if unit == 'C':
                celsius = number
            elif unit == 'F':
                celsius = (number - 32) * 5/9
            elif unit == 'K':
                celsius = number - 273.15
            else:
                return np.nan
            
            # Convert from Celsius to target
            if target_unit == 'C':
                return celsius
            elif target_unit == 'F':
                return celsius * 9/5 + 32
            elif target_unit == 'K':
                return celsius + 273.15
            else:
                return celsius
        
        result = [convert_temperature(data, most_common_unit) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 1) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
