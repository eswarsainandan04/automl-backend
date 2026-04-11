"""
Network Addresses Pattern - Detects IP addresses, MAC addresses, subnet masks
Detection only - does not normalize
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern

class NetworkAddressesPattern(BasePattern):
    
    regex_patterns = [
        # IPv4 addresses
        r'^(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        
        # IPv4 with CIDR notation
        r'^(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)/[0-9]{1,2}$',
        
        # IPv6 addresses (full)
        r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
        
        # IPv6 addresses (compressed)
        r'^::([0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}$',
        r'^([0-9a-fA-F]{1,4}:){1,6}:([0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}$',
        
        # IPv6 loopback
        r'^::1$',
        
        # IPv6 with zone ID
        r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}%[0-9a-zA-Z]+$',
        
        # MAC addresses (colon separated)
        r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$',
        
        # MAC addresses (hyphen separated)
        r'^([0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}$',
        
        # MAC addresses (dot separated - Cisco style)
        r'^([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}$',
        
        # MAC addresses (no separator)
        r'^[0-9A-Fa-f]{12}$',
        
        # Subnet masks
        r'^255\.255\.255\.(0|128|192|224|240|248|252|254|255)$',
        r'^255\.255\.(0|128|192|224|240|248|252|254|255)\.0$',
        r'^255\.(0|128|192|224|240|248|252|254|255)\.0\.0$',
        
        # Private IP ranges identifiers
        r'^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        r'^172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}$',
        r'^192\.168\.\d{1,3}\.\d{1,3}$',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = 'network_address'
    
    def detect(self, values):
        """
        Detect if column contains network addresses (IP, MAC, etc.)
        Returns confidence as percentage of valid addresses
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
        
        # Check how many values match network address patterns
        matched = 0
        for val in non_null:
            # Skip missing value indicators
            if val.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                continue
            
            # Check against network address regex patterns
            is_network_addr = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val, re.IGNORECASE):
                    is_network_addr = True
                    break
            
            if is_network_addr:
                matched += 1
        
        # Return confidence as percentage
        total = len(non_null)
        return matched / total if total > 0 else 0.0
    
    def normalize(self, values):
        """
        For network addresses, return original values unchanged (detection only)
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        # Return as-is without any modifications
        return values
