"""
File Path Pattern - Detects file system paths (Windows, Linux, Mac)
Detection only - does not normalize
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern

class FilePathPattern(BasePattern):
    
    regex_patterns = [
        # Windows absolute paths
        r'^[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$',
        
        # Windows UNC paths
        r'^\\\\[^\\/:*?"<>|\r\n]+\\[^\\/:*?"<>|\r\n]+(?:\\[^\\/:*?"<>|\r\n]+)*$',
        
        # Linux/Mac absolute paths
        r'^/(?:[^/\0]+/)*[^/\0]*$',
        
        # Linux/Mac home directory paths
        r'^~(?:/[^/\0]+)*$',
        
        # Relative paths (Windows)
        r'^\.\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$',
        r'^\.\.\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$',
        
        # Relative paths (Linux/Mac)
        r'^\.(?:/[^/\0]+)+$',
        r'^\.\.(?:/[^/\0]+)+$',
        
        # Windows paths with forward slashes
        r'^[A-Za-z]:/(?:[^/:*?"<>|\r\n]+/)*[^/:*?"<>|\r\n]*$',
        
        # Common Windows system paths
        r'^%[A-Za-z_]+%(?:\\[^\\/:*?"<>|\r\n]+)*$',
        
        # Environment variable paths (Linux/Mac)
        r'^\$[A-Za-z_][A-Za-z0-9_]*(?:/[^/\0]+)*$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = 'file_path'
    
    def detect(self, values):
        """
        Detect if column contains file paths
        Returns confidence as percentage of valid paths
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
        
        # Check how many values match file path patterns
        matched = 0
        for val in non_null:
            # Skip missing value indicators
            if val.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                continue
            
            # Skip very short strings (unlikely to be paths)
            if len(val) < 3:
                continue
            
            # Check against file path regex patterns
            is_path = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val):
                    is_path = True
                    break
            
            if is_path:
                matched += 1
        
        # Return confidence as percentage
        total = len(non_null)
        return matched / total if total > 0 else 0.0
    
    def normalize(self, values):
        """
        For file paths, return original values unchanged (detection only)
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        # Return as-is without any modifications
        return values
