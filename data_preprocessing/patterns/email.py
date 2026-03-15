"""
Email Pattern - Detects and normalizes email addresses
Validates and normalizes email formats
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern

class EmailPattern(BasePattern):
    
    regex_patterns = [
        # Standard email pattern
        r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$',
        # Email with dots
        r'^[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*@[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*\.[a-zA-Z]{2,}$',
        # Email with underscores
        r'^[a-zA-Z0-9_]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$',
        # Email with hyphens
        r'^[a-zA-Z0-9\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = 'email'
    
    def detect(self, values):
        """
        Detect if column contains email addresses
        Returns confidence as percentage of valid emails
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        if values.empty:
            return 0.0
        
        # Filter out null values and convert to string
        non_null = values.dropna().astype(str).str.strip()
        
        if len(non_null) == 0:
            return 0.0
        
        # Check how many values match email patterns
        matched = 0
        for val in non_null:
            # Skip missing value indicators
            if val.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                continue
            
            # Check against email regex patterns
            is_email = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val, re.IGNORECASE):
                    is_email = True
                    break
            
            if is_email:
                matched += 1
        
        # Return confidence as percentage
        total = len(non_null)
        return matched / total if total > 0 else 0.0
    
    def normalize(self, values):
        """
        Normalize email addresses to lowercase
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        if values.empty:
            return values
        
        result = []
        for idx, val in values.items():
            # Handle NaN/None
            if pd.isna(val):
                result.append(np.nan)
                continue
            
            # Convert to string and clean
            val_str = str(val).strip()
            
            # Handle missing value indicators
            if val_str.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                result.append(np.nan)
                continue
            
            # Normalize to lowercase
            result.append(val_str.lower())
        
        # Return as object dtype to preserve string type with NaN
        return pd.Series(result, index=values.index, dtype='object')
