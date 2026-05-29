import re
import pandas as pd
import numpy as np
from .base import BasePattern


class WeightPattern(BasePattern):
    semantic_type = "weight"
    detected_unit = None
    
    regex_patterns = [
        # Metric system - WEIGHT ONLY (no density units like g/cm3)
        r'^\d+(?:\.\d+)?\s*(?:kg|kilogram|kilograms)$',
        r'^\d+(?:\.\d+)?\s*(?:g|gram|grams)$',
        r'^\d+(?:\.\d+)?\s*(?:mg|milligram|milligrams)$',
        r'^\d+(?:\.\d+)?\s*(?:µg|ug|microgram|micrograms)$',
        r'^\d+(?:\.\d+)?\s*(?:ton|tons|tonne|tonnes|metric\s*ton|metric\s*tons)$',
        # Imperial system - WEIGHT ONLY
        r'^\d+(?:\.\d+)?\s*(?:lb|lbs|pound|pounds)$',
        r'^\d+(?:\.\d+)?\s*(?:oz|ounce|ounces)$',
        r'^\d+(?:\.\d+)?\s*(?:st|stone|stones)$',
        # With explicit space variations
        r'^\d+(?:\.\d+)?\s+(?:kg|g|mg|lb|lbs|oz|ton|tons|tonne|tonnes)$',
        # Note: Excludes any ratio/density units like g/cm3, kg/L
    ]
    
    CONVERSIONS = {
        'kg': 1, 'kilogram': 1, 'kilograms': 1,
        'g': 0.001, 'gram': 0.001, 'grams': 0.001,
        'mg': 0.000001, 'milligram': 0.000001, 'milligrams': 0.000001,
        'µg': 0.000000001, 'ug': 0.000000001, 'microgram': 0.000000001, 'micrograms': 0.000000001,
        'ton': 1000, 'tons': 1000, 'tonne': 1000, 'tonnes': 1000, 'metric ton': 1000, 'metric tons': 1000,
        'lb': 0.453592, 'lbs': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
        'oz': 0.0283495, 'ounce': 0.0283495, 'ounces': 0.0283495,
        'st': 6.35029, 'stone': 6.35029, 'stones': 6.35029,
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
            self.detected_unit = 'kg'
            most_common_unit = 'kg'
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            value_in_kg = number * self.CONVERSIONS[unit]
            value_in_target = value_in_kg / target_conversion
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
