import re
import pandas as pd
import numpy as np
from .base import BasePattern


class VolumePattern(BasePattern):
    semantic_type = "volume"
    detected_unit = None
    
    regex_patterns = [
        # Metric system - VOLUME ONLY (not weight or area)
        r'^\d+(?:\.\d+)?\s*(?:[lL]|liter|liters|litre|litres)$',
        r'^\d+(?:\.\d+)?\s*(?:ml|mL|milliliter|milliliters|millilitre|millilitres)$',
        r'^\d+(?:\.\d+)?\s*(?:cl|cL|centiliter|centiliters|centilitre|centilitres)$',
        r'^\d+(?:\.\d+)?\s*(?:dl|dL|deciliter|deciliters|decilitre|decilitres)$',
        r'^\d+(?:\.\d+)?\s*(?:m3|m³|cubic\s*meter|cubic\s*meters|cubic\s*metre|cubic\s*metres)$',
        r'^\d+(?:\.\d+)?\s*(?:cm3|cm³|cubic\s*centimeter|cubic\s*centimeters)$',
        # Imperial/US system - VOLUME ONLY
        r'^\d+(?:\.\d+)?\s*(?:gal|gallon|gallons)$',
        r'^\d+(?:\.\d+)?\s*(?:qt|quart|quarts)$',
        r'^\d+(?:\.\d+)?\s*(?:pt|pint|pints)$',
        r'^\d+(?:\.\d+)?\s*(?:fl\s*oz|fluid\s*ounce|fluid\s*ounces)$',
        r'^\d+(?:\.\d+)?\s*(?:cup|cups)$',
        r'^\d+(?:\.\d+)?\s*(?:ft3|ft³|cubic\s*foot|cubic\s*feet)$',
        r'^\d+(?:\.\d+)?\s*(?:tbsp|tablespoon|tablespoons)$',
        r'^\d+(?:\.\d+)?\s*(?:tsp|teaspoon|teaspoons)$',
    ]
    
    CONVERSIONS = {
        'l': 1, 'L': 1, 'liter': 1, 'liters': 1, 'litre': 1, 'litres': 1,
        'ml': 0.001, 'mL': 0.001, 'milliliter': 0.001, 'milliliters': 0.001, 'millilitre': 0.001, 'millilitres': 0.001,
        'cl': 0.01, 'cL': 0.01, 'centiliter': 0.01, 'centiliters': 0.01, 'centilitre': 0.01, 'centilitres': 0.01,
        'dl': 0.1, 'dL': 0.1, 'deciliter': 0.1, 'deciliters': 0.1, 'decilitre': 0.1, 'decilitres': 0.1,
        'm3': 1000, 'm³': 1000, 'cubic meter': 1000, 'cubic meters': 1000, 'cubic metre': 1000, 'cubic metres': 1000,
        'cm3': 0.001, 'cm³': 0.001, 'cubic centimeter': 0.001, 'cubic centimeters': 0.001,
        'gal': 3.78541, 'gallon': 3.78541, 'gallons': 3.78541,
        'qt': 0.946353, 'quart': 0.946353, 'quarts': 0.946353,
        'pt': 0.473176, 'pint': 0.473176, 'pints': 0.473176,
        'fl oz': 0.0295735, 'fluid ounce': 0.0295735, 'fluid ounces': 0.0295735,
        'cup': 0.236588, 'cups': 0.236588,
        'ft3': 28.3168, 'ft³': 28.3168, 'cubic foot': 28.3168, 'cubic feet': 28.3168,
        'tbsp': 0.01479, 'tablespoon': 0.01479, 'tablespoons': 0.01479,
        'tsp': 0.00493, 'teaspoon': 0.00493, 'teaspoons': 0.00493,
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
            self.detected_unit = 'l'
            most_common_unit = 'l'
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            value_in_liters = number * self.CONVERSIONS[unit]
            value_in_target = value_in_liters / target_conversion
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
