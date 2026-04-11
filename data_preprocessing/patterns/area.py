import re
import pandas as pd
import numpy as np
from .base import BasePattern


class AreaPattern(BasePattern):
    semantic_type = "area"
    detected_unit = None
    
    regex_patterns = [
        # Metric - AREA ONLY (not distance or volume)
        r'^\d+(?:\.\d+)?\s*(?:km2|km²|sq\s*km|square\s*kilometer|square\s*kilometers)$',
        r'^\d+(?:\.\d+)?\s*(?:m2|m²|sq\s*m|square\s*meter|square\s*meters|square\s*metre|square\s*metres)$',
        r'^\d+(?:\.\d+)?\s*(?:cm2|cm²|sq\s*cm|square\s*centimeter|square\s*centimeters)$',
        r'^\d+(?:\.\d+)?\s*(?:mm2|mm²|sq\s*mm|square\s*millimeter|square\s*millimeters)$',
        r'^\d+(?:\.\d+)?\s*(?:ha|hectare|hectares)$',
        # Imperial - AREA ONLY
        r'^\d+(?:\.\d+)?\s*(?:sq\s*mi|square\s*mile|square\s*miles)$',
        r'^\d+(?:\.\d+)?\s*(?:acre|acres)$',
        r'^\d+(?:\.\d+)?\s*(?:sq\s*ft|ft2|ft²|square\s*foot|square\s*feet)$',
        r'^\d+(?:\.\d+)?\s*(?:sq\s*in|in2|in²|square\s*inch|square\s*inches)$',
        r'^\d+(?:\.\d+)?\s*(?:sq\s*yd|yd2|yd²|square\s*yard|square\s*yards)$',
    ]
    
    CONVERSIONS = {
        'km2': 1000000, 'km²': 1000000, 'sq km': 1000000, 'sqkm': 1000000, 'square kilometer': 1000000, 'square kilometers': 1000000,
        'm2': 1, 'm²': 1, 'sq m': 1, 'sqm': 1, 'square meter': 1, 'square meters': 1, 'square metre': 1, 'square metres': 1,
        'cm2': 0.0001, 'cm²': 0.0001, 'sq cm': 0.0001, 'sqcm': 0.0001, 'square centimeter': 0.0001, 'square centimeters': 0.0001,
        'mm2': 0.000001, 'mm²': 0.000001, 'sq mm': 0.000001, 'sqmm': 0.000001, 'square millimeter': 0.000001, 'square millimeters': 0.000001,
        'ha': 10000, 'hectare': 10000, 'hectares': 10000,
        'sq mi': 2589988, 'sqmi': 2589988, 'square mile': 2589988, 'square miles': 2589988,
        'acre': 4046.86, 'acres': 4046.86,
        'sq ft': 0.092903, 'sqft': 0.092903, 'ft2': 0.092903, 'ft²': 0.092903, 'square foot': 0.092903, 'square feet': 0.092903,
        'sq in': 0.00064516, 'sqin': 0.00064516, 'in2': 0.00064516, 'in²': 0.00064516, 'square inch': 0.00064516, 'square inches': 0.00064516,
        'sq yd': 0.836127, 'sqyd': 0.836127, 'yd2': 0.836127, 'yd²': 0.836127, 'square yard': 0.836127, 'square yards': 0.836127,
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
            self.detected_unit = 'm2'
            most_common_unit = 'm2'
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            value_in_m2 = number * self.CONVERSIONS[unit]
            value_in_target = value_in_m2 / target_conversion
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
