import pandas as pd
import re
from .base import BasePattern


class VarcharPattern(BasePattern):
    """Pattern for alphanumeric/mixed string+numeric values (varchar/string type)"""
    semantic_type = "varchar"
    
    regex_patterns = [
        r'^[A-Za-z0-9]+$',  # Alphanumeric: ABC123, User001
        r'^[A-Za-z]+\d+$',  # Letters then numbers: Room42, ID123
        r'^\d+[A-Za-z]+$',  # Numbers then letters: 123ABC, 42nd
        r'^[A-Za-z0-9\s]+$',  # Alphanumeric with spaces: User 001, Room 42
        r'^[A-Za-z0-9_-]+$',  # With underscores/hyphens: user_001, ID-123
    ]
    
    def detect(self, values) -> float:
        """Detect if values are mixed string+numeric (varchar)"""
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        v = values.dropna()
        if len(v) == 0:
            return 0.0
        
        varchar_count = 0
        pure_numeric_count = 0
        pure_text_count = 0
        specialized_pattern_count = 0
        
        for val in v:
            val_str = str(val).strip()
            
            # Skip empty
            if not val_str:
                continue
            
            # Check for specialized patterns (timezone, coordinates, etc.) - exclude them
            if re.search(r'\bUTC\b|\bGMT\b|^[A-Z]{2,5}\s+(UTC)?\d+:?\d{0,2}$|^[A-Z]{2,5}\s*[+-]\d', val_str):
                specialized_pattern_count += 1
                continue
            
            # Check if purely numeric
            try:
                float(val_str.replace(',', '').replace('$', '').replace('%', ''))
                pure_numeric_count += 1
                continue
            except:
                pass
            
            # Check if contains both letters and numbers (varchar)
            has_letters = bool(re.search(r'[A-Za-z]', val_str))
            has_numbers = bool(re.search(r'\d', val_str))
            
            if has_letters and has_numbers:
                varchar_count += 1
            elif has_letters and not has_numbers:
                # Pure text (only letters, spaces, punctuation - no digits)
                pure_text_count += 1
            else:
                # Other cases
                pass
        
        total = len(v)
        
        # If most are specialized patterns, return 0
        if specialized_pattern_count / total > 0.5:
            return 0.0
        
        # If more than 60% are mixed alphanumeric, it's varchar
        if varchar_count / total > 0.6:
            return varchar_count / total
        
        # If has some varchar but not dominant, return moderate confidence
        if varchar_count > 0 and varchar_count / total > 0.3:
            return 0.5
        
        return 0.0
    
    def normalize(self, values):
        """Keep varchar values in original format (no normalization)"""
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values)
