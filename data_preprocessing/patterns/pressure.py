import re
import pandas as pd
import numpy as np
from .base import BasePattern


class PressurePattern(BasePattern):
    semantic_type = "pressure"
    detected_unit = None
    
    regex_patterns = [
        r'^\d+(?:\.\d+)?\s*(?:Pa|pascal|pascals)$',
        r'^\d+(?:\.\d+)?\s*(?:kPa|kilopascal|kilopascals)$',
        r'^\d+(?:\.\d+)?\s*(?:MPa|megapascal|megapascals)$',
        r'^\d+(?:\.\d+)?\s*(?:bar|bars)$',
        r'^\d+(?:\.\d+)?\s*(?:psi|PSI)$',
        r'^\d+(?:\.\d+)?\s*(?:atm|atmosphere|atmospheres)$',
        r'^\d+(?:\.\d+)?\s*(?:mmHg|mm Hg|torr)$',
    ]
    
    CONVERSIONS = {
        'pa': 1, 'pascal': 1, 'pascals': 1,
        'kpa': 1000, 'kilopascal': 1000, 'kilopascals': 1000,
        'mpa': 1000000, 'megapascal': 1000000, 'megapascals': 1000000,
        'bar': 100000, 'bars': 100000,
        'psi': 6894.76, 'psi': 6894.76,
        'atm': 101325, 'atmosphere': 101325, 'atmospheres': 101325,
        'mmhg': 133.322, 'mm hg': 133.322, 'torr': 133.322,
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
            self.detected_unit = 'pa'
            most_common_unit = 'pa'
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            value_in_pa = number * self.CONVERSIONS[unit]
            value_in_target = value_in_pa / target_conversion
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
