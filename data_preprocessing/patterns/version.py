"""
Version Pattern - Detects software version numbers
Detection only - does not normalize
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern

class VersionPattern(BasePattern):
    
    regex_patterns = [
        # Semantic versioning (major.minor.patch)
        r'^v?\d+\.\d+\.\d+$',
        
        # Semantic versioning with pre-release
        r'^v?\d+\.\d+\.\d+-[a-zA-Z0-9\.\-]+$',
        
        # Semantic versioning with build metadata
        r'^v?\d+\.\d+\.\d+\+[a-zA-Z0-9\.\-]+$',
        
        # Semantic versioning with pre-release and build
        r'^v?\d+\.\d+\.\d+-[a-zA-Z0-9\.\-]+\+[a-zA-Z0-9\.\-]+$',
        
        # Major.minor versioning
        r'^v?\d+\.\d+$',
        
        # Major.minor.patch.build
        r'^v?\d+\.\d+\.\d+\.\d+$',
        
        # With alpha/beta/rc labels
        r'^v?\d+\.\d+\.\d+[-\s]?(alpha|beta|rc|preview|snapshot)[\.\-]?\d*$',
        
        # Date-based versions (YYYY.MM.DD)
        r'^v?\d{4}\.\d{1,2}\.\d{1,2}$',
        
        # Date-based versions (YYYYMMDD)
        r'^v?\d{8}$',
        
        # Ubuntu-style versions
        r'^v?\d+\.\d+(\.\d+)?[a-zA-Z]*$',
        
        # Python-style versions
        r'^v?\d+\.\d+\.\d+[a-z]\d+$',
        
        # With v prefix
        r'^v\d+$',
        
        # Build numbers
        r'^build[\s\-]?\d+$',
        r'^b\d+$',
        
        # Release candidates
        r'^v?\d+\.\d+\.\d+[\s\-]?RC\d+$',
        
        # Milestone versions
        r'^v?\d+\.\d+[\s\-]?M\d+$',
        
        # Snapshot versions
        r'^v?\d+\.\d+\.\d+[\s\-]?SNAPSHOT$',
        
        # X.X.X format with any depth
        r'^v?\d+(?:\.\d+){1,4}$',
        
        # Version with commit hash
        r'^v?\d+\.\d+\.\d+[\-\+][0-9a-f]{7,40}$',
        
        # CalVer (Calendar Versioning) - YYYY.MM
        r'^v?\d{4}\.\d{1,2}$',
        
        # CalVer - YY.MM
        r'^v?\d{2}\.\d{1,2}\.\d+$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = 'version'
    
    def detect(self, values):
        """
        Detect if column contains version numbers
        Returns confidence as percentage of valid version strings
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
        
        # Check how many values match version patterns
        matched = 0
        total_valid = 0  # only count non-empty, non-placeholder values
        for val in non_null:
            # Skip missing value indicators, empty strings, and common placeholders
            if val.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '',
                               'varies with device', 'varies', 'unknown',
                               'not specified', 'not available', 'n/a',
                               'unspecified', 'tbd', 'tba']:
                continue
            
            total_valid += 1
            
            # Check against version regex patterns (case-insensitive for alpha/beta/rc)
            is_version = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val, re.IGNORECASE):
                    is_version = True
                    break
            
            if is_version:
                matched += 1
        
        # Return confidence as percentage of valid (non-empty) values
        return matched / total_valid if total_valid > 0 else 0.0
    
    def normalize(self, values):
        """
        For versions, blank out placeholders like 'Varies with device', 'unknown', etc.
        Returns cleaned version strings, with placeholders set to empty string.
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)

        # List of placeholder/invalid values (case-insensitive)
        placeholders = set([
            '?', 'na', 'n/a', 'nan', 'null', 'none', '',
            'varies with device', 'varies', 'unknown',
            'not specified', 'not available', 'unspecified',
            'tbd', 'tba', 'n.a.', 'n.a', 'n.a', 'n a',
        ])

        def clean_version(val):
            if pd.isna(val):
                return ''
            sval = str(val).strip().lower()
            if sval in placeholders:
                return ''
            return val

        return values.apply(clean_version)
