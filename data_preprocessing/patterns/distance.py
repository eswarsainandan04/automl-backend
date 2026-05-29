import re
import pandas as pd
import numpy as np
from .base import BasePattern


class DistancePattern(BasePattern):
    semantic_type = "distance"
    detected_unit = None  # Will store the most common unit for column renaming
    
    regex_patterns = [
        # Metric system - DISTANCE ONLY
        r'^\d+(?:\.\d+)?\s*(?:km|kilometer|kilometers|kilometre|kilometres)$',
        r'^\d+(?:\.\d+)?\s*(?:m|meter|meters|metre|metres)$',
        r'^\d+(?:\.\d+)?\s*(?:dm|decimeter|decimeters|decimetre|decimetres)$',
        r'^\d+(?:\.\d+)?\s*(?:cm|centimeter|centimeters|centimetre|centimetres)$',
        r'^\d+(?:\.\d+)?\s*(?:mm|millimeter|millimeters|millimetre|millimetres)$',
        r'^\d+(?:\.\d+)?\s*(?:µm|um|micrometer|micrometers|micrometre|micrometres)$',
        # Imperial system - DISTANCE ONLY
        r'^\d+(?:\.\d+)?\s*(?:mi|mile|miles)$',
        r'^\d+(?:\.\d+)?\s*(?:yd|yard|yards)$',
        r'^\d+(?:\.\d+)?\s*(?:ft|foot|feet)$',
        r'^\d+(?:\.\d+)?\s*(?:in|inch|inches)$',
        # Nautical
        r'^\d+(?:\.\d+)?\s*(?:nmi|nm|nautical\s*mile|nautical\s*miles)$',
    ]
    
    # Conversion factors to meters (base unit)
    CONVERSIONS = {
        'km': 1000, 'kilometer': 1000, 'kilometers': 1000,
        'm': 1, 'meter': 1, 'meters': 1, 'metre': 1, 'metres': 1,
        'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01, 'centimetre': 0.01, 'centimetres': 0.01,
        'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001, 'millimetre': 0.001, 'millimetres': 0.001,
        'mi': 1609.34, 'mile': 1609.34, 'miles': 1609.34,
        'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144,
        'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
        'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
        'nmi': 1852, 'nm': 1852, 'nautical mile': 1852, 'nautical miles': 1852,
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
        # First pass: extract all units and find the most common one
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
            
            # Extract number and unit
            match = re.match(r'^(\d+(?:\.\d+)?)\s*(.+)$', val_str)
            if match:
                number = float(match.group(1))
                unit = match.group(2).strip()
                
                # Normalize unit name
                if unit in self.CONVERSIONS:
                    parsed_data.append((number, unit))
                    unit_counts[unit] = unit_counts.get(unit, 0) + 1
                else:
                    parsed_data.append((np.nan, None))
            else:
                parsed_data.append((np.nan, None))
        
        # Determine the most common unit
        if unit_counts:
            most_common_unit = max(unit_counts, key=unit_counts.get)
            self.detected_unit = most_common_unit
        else:
            self.detected_unit = 'm'  # Default to meters
            most_common_unit = 'm'
        
        # Second pass: convert all values to the most common unit
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            # Convert to meters first, then to target unit
            value_in_meters = number * self.CONVERSIONS[unit]
            value_in_target = value_in_meters / target_conversion
            
            # Check if any value has decimals for float formatting
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        
        # Check if any values have decimals
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            # Format all as float with 1 decimal place
            return pd.Series([round(val, 1) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            # Keep as integers
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
