import re
import pandas as pd
import numpy as np
from .base import BasePattern


class CapacityPattern(BasePattern):
    semantic_type = "capacity"
    detected_unit = None
    
    regex_patterns = [
        # Storage capacity (prioritized first)
        r'^\d+(?:\.\d+)?\s*(?:TB|terabyte|terabytes)$',
        r'^\d+(?:\.\d+)?\s*(?:GB|gigabyte|gigabytes)$',
        r'^\d+(?:\.\d+)?\s*(?:MB|megabyte|megabytes)$',
        r'^\d+(?:\.\d+)?\s*(?:KB|kilobyte|kilobytes)$',
        # Storage with suffixes (RAM, storage, disk, etc.)
        r'^\d+(?:\.\d+)?\s*(?:TB|GB|MB|KB)\s+(?:RAM|rom|storage|disk|memory|ssd|hdd)$',
        r'^\d+(?:\.\d+)?\s+(?:TB|GB|MB|KB)(?:\s+(?:RAM|rom|storage|disk|memory|ssd|hdd))?$',
        # Liquid volume (secondary)
        r'^\d+(?:\.\d+)?\s*(?:l|L|liter|liters|litre|litres)$',
        r'^\d+(?:\.\d+)?\s*(?:kl|KL|kiloliter|kiloliters|kilolitre|kilolitres)$',

        r'^\d+(?:\.\d+)?\s*(?:ml|mL|milliliter|milliliters|millilitre|millilitres)$',
        r'^\d+(?:\.\d+)?\s*(?:gal|gallon|gallons)$',
        r'^\d+(?:\.\d+)?\s*(?:qt|quart|quarts)$',
        r'^\d+(?:\.\d+)?\s*(?:pt|pint|pints)$',
        r'^\d+(?:\.\d+)?\s*(?:cup|cups)$',
    ]
    
    CONVERSIONS = {
        # Storage (base: GB)
        'tb': 1024, 'terabyte': 1024, 'terabytes': 1024,
        'gb': 1, 'gigabyte': 1, 'gigabytes': 1,
        'mb': 0.001, 'megabyte': 0.001, 'megabytes': 0.001,
        'kb': 0.000001, 'kilobyte': 0.000001, 'kilobytes': 0.000001,
        # Volume (base: liters) - separate conversion system
        'l': 1, 'liter': 1, 'liters': 1, 'litre': 1, 'litres': 1,
        'kl': 1000,
        'kiloliter': 1000, 'kiloliters': 1000,
        'kilolitre': 1000, 'kilolitres': 1000,
        'ml': 0.001, 'milliliter': 0.001, 'milliliters': 0.001, 'millilitre': 0.001, 'millilitres': 0.001,
        'gal': 3.78541, 'gallon': 3.78541, 'gallons': 3.78541,
        'qt': 0.946353, 'quart': 0.946353, 'quarts': 0.946353,
        'pt': 0.473176, 'pint': 0.473176, 'pints': 0.473176,
        'cup': 0.236588, 'cups': 0.236588,
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
                unit_raw = match.group(2).strip()
                
                # Remove storage suffixes (RAM, storage, disk, etc.) to get the base unit
                unit = re.sub(r'\s*(?:ram|rom|storage|disk|memory|ssd|hdd)$', '', unit_raw, flags=re.IGNORECASE).strip()
                
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
            self.detected_unit = 'gb'  # Default to GB for storage
            most_common_unit = 'gb'
        
        # Determine if this is storage or liquid volume
        is_storage = most_common_unit in ['tb', 'gb', 'mb', 'kb', 'terabyte', 'gigabyte', 'megabyte', 'kilobyte', 'terabytes', 'gigabytes', 'megabytes', 'kilobytes']
        
        target_conversion = self.CONVERSIONS[most_common_unit]
        
        def convert_to_target_unit(data_tuple):
            number, unit = data_tuple
            if pd.isna(number) or unit is None:
                return np.nan
            
            # Check if both source and target are same type (storage or volume)
            unit_is_storage = unit in ['tb', 'gb', 'mb', 'kb', 'terabyte', 'gigabyte', 'megabyte', 'kilobyte', 'terabytes', 'gigabytes', 'megabytes', 'kilobytes']
            
            if is_storage and unit_is_storage:
                # Storage to storage conversion (base: GB)
                value_in_gb = number * self.CONVERSIONS[unit]
                value_in_target = value_in_gb / target_conversion
            elif not is_storage and not unit_is_storage:
                # Volume to volume conversion (base: liters)
                value_in_liters = number * self.CONVERSIONS[unit]
                value_in_target = value_in_liters / target_conversion
            else:
                # Incompatible conversion (storage to volume or vice versa)
                return np.nan
            
            return value_in_target
        
        result = [convert_to_target_unit(data) for data in parsed_data]
        has_decimals = any(not pd.isna(val) and val != int(val) for val in result)
        
        if has_decimals:
            return pd.Series([round(val, 2) if not pd.isna(val) else np.nan for val in result], index=values.index)
        else:
            return pd.Series([int(val) if not pd.isna(val) else np.nan for val in result], index=values.index)
