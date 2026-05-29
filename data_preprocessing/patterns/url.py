"""
URL Pattern - Detects web URLs and domain names
Detection only - does not normalize
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern

class URLPattern(BasePattern):
    
    regex_patterns = [
        # Full URLs with protocol
        r'^https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$',
        r'^ftp://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$',
        r'^ftps?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$',
        
        # URLs with www
        r'^www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?$',
        
        # Domain names (require common TLDs or subdomain structure to avoid matching filenames)
        r'^[a-zA-Z0-9\-]+\.(com|org|net|edu|gov|mil|int|co|io|ai|dev|app|tech|info|biz|name|pro|museum|aero|asia|cat|coop|jobs|mobi|tel|travel|xxx|[a-z]{2}\.[a-z]{2})(?:/[^\s]*)?$',
        
        # URLs with port
        r'^https?://[a-zA-Z0-9\-\.]+:[0-9]+(?:/[^\s]*)?$',
        
        # Localhost URLs
        r'^https?://localhost(?::[0-9]+)?(?:/[^\s]*)?$',
        r'^https?://127\.0\.0\.1(?::[0-9]+)?(?:/[^\s]*)?$',
        
        # URLs with query parameters
        r'^https?://[^\s]+\?[^\s]+$',
        
        # URLs with fragments
        r'^https?://[^\s]+#[^\s]+$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = 'url'
    
    def detect(self, values):
        """
        Detect if column contains URLs
        Returns confidence as percentage of valid URLs
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
        
        # Check how many values match URL patterns
        matched = 0
        for val in non_null:
            # Skip missing value indicators
            if val.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                continue
            
            # Check against URL regex patterns
            is_url = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val, re.IGNORECASE):
                    is_url = True
                    break
            
            if is_url:
                matched += 1
        
        # Return confidence as percentage
        total = len(non_null)
        return matched / total if total > 0 else 0.0
    
    def normalize(self, values):
        """
        For URLs, return original values unchanged (detection only)
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        # Return as-is without any modifications
        return values
