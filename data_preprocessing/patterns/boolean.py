import re
import pandas as pd
import numpy as np
from .base import BasePattern


class BooleanPattern(BasePattern):
    semantic_type = "boolean"
    detected_format = None  # Will store the detected format for column renaming (e.g., "yes/no", "active/inactive")
    
    regex_patterns = [
        # True/False variants
        r'^(?:true|True|TRUE)$',
        r'^(?:false|False|FALSE)$',
        # 1/0 (numeric boolean)
        r'^[01]$',
        # On/Off
        r'^(?:on|On|ON)$',
        r'^(?:off|Off|OFF)$',
        # Yes/No variants (only explicit yes/no, not categorical)
        r'^(?:yes|Yes|YES)$',
        r'^(?:no|No|NO)$',
        # Status patterns
        r'^(?:active|Active|ACTIVE)$',
        r'^(?:inactive|Inactive|INACTIVE)$',
        r'^(?:enabled|Enabled|ENABLED)$',
        r'^(?:disabled|Disabled|DISABLED)$',
        r'^(?:online|Online|ONLINE)$',
        r'^(?:offline|Offline|OFFLINE)$',
        r'^(?:running|Running|RUNNING)$',
        r'^(?:stopped|Stopped|STOPPED)$',
        # Approval patterns
        r'^(?:approved|Approved|APPROVED)$',
        r'^(?:rejected|Rejected|REJECTED)$',
        r'^(?:accepted|Accepted|ACCEPTED)$',
        r'^(?:declined|Declined|DECLINED)$',
        r'^(?:pass|Pass|PASS)$',
        r'^(?:fail|Fail|FAIL)$',
        r'^(?:passed|Passed|PASSED)$',
        r'^(?:failed|Failed|FAILED)$',
        r'^(?:success|Success|SUCCESS)$',
        r'^(?:failure|Failure|FAILURE)$',
        r'^(?:granted|Granted|GRANTED)$',
        r'^(?:denied|Denied|DENIED)$',
        # Flag patterns
        r'^(?:set|Set|SET)$',
        r'^(?:unset|Unset|UNSET)$',
        # Availability patterns
        r'^(?:available|Available|AVAILABLE)$',
        r'^(?:unavailable|Unavailable|UNAVAILABLE)$',
        r'^(?:in stock|In Stock|IN STOCK)$',
        r'^(?:out of stock|Out of Stock|OUT OF STOCK)$',
        r'^(?:in-stock|In-Stock|IN-STOCK)$',
        r'^(?:out-of-stock|Out-of-Stock|OUT-OF-STOCK)$',
        r'^(?:present|Present|PRESENT)$',
        r'^(?:absent|Absent|ABSENT)$',
        r'^(?:ready|Ready|READY)$',
        r'^(?:not ready|Not Ready|NOT READY)$',
    ]
    
    # All TRUE values (map to 1)
    TRUE_VALUES = [
        'true', '1', 'on', 't',
        'yes', 'y',
        'active', 'enabled', 'online', 'running',
        'approved', 'accepted', 'pass', 'passed', 'success', 'granted',
        'set',
        'available', 'in stock', 'in-stock', 'present', 'ready'
    ]
    
    # All FALSE values (map to 0)
    FALSE_VALUES = [
        'false', '0', 'off', 'f',
        'no', 'n',
        'inactive', 'disabled', 'offline', 'stopped',
        'rejected', 'declined', 'fail', 'failed', 'failure', 'denied',
        'unset',
        'unavailable', 'out of stock', 'out-of-stock', 'absent', 'not ready'
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0
        
        matched = 0
        total = 0
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if isinstance(value, str) and value.strip().lower() in ['null', 'nan', 'none', '', '?']:
                continue
                
            total += 1
            value_str = str(value).strip().lower()
            
            if value_str in self.TRUE_VALUES or value_str in self.FALSE_VALUES:
                matched += 1
        
        if total == 0:
            return 0

        match_ratio = matched / total

        # STRICT: Every single non-null value must be boolean-like.
        # A column like Price with "0", "$3.99", "$0.99" must NOT match.
        if match_ratio < 1.0:
            return 0.0

        # Collect the distinct non-null boolean-like values
        clean_vals = set()
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            val_str = str(value).strip().lower()
            if val_str in ['null', 'nan', 'none', '', '?']:
                continue
            clean_vals.add(val_str)

        # Must have at most 2 distinct values
        if len(clean_vals) > 2:
            return 0.0

        # Must have BOTH a true-like AND a false-like value present.
        # A column with only 0s (or only 1s) is NOT demonstrably boolean —
        # it could be a numeric column where values happen to be 0.
        has_true = bool(clean_vals & set(self.TRUE_VALUES))
        has_false = bool(clean_vals & set(self.FALSE_VALUES))

        if has_true and has_false:
            return 1.0

        # Only one side present (e.g. all 0s) → very low confidence
        return 0.3
    
    def normalize(self, values):
        # First pass: detect format type and collect parsed data
        format_counts = {}
        parsed_data = []
        
        for val in values:
            if pd.isna(val):
                parsed_data.append((np.nan, None))
                continue
                
            val_str = str(val).strip().lower()
            if val_str in ['null', 'nan', 'none', '', '?']:
                parsed_data.append((np.nan, None))
                continue
            
            # Determine boolean value and original format
            if val_str in self.TRUE_VALUES:
                bool_val = 1  # Always use 1 for true
                
                # Detect format type for column renaming
                if val_str in ['true', 't']:
                    fmt = 'true/false'
                elif val_str in ['yes', 'y']:
                    fmt = 'yes/no'
                elif val_str == '1':
                    fmt = '1/0'
                elif val_str == 'on':
                    fmt = 'on/off'
                elif val_str in ['active', 'inactive']:
                    fmt = 'active/inactive'
                elif val_str in ['enabled', 'disabled']:
                    fmt = 'enabled/disabled'
                elif val_str in ['online', 'offline']:
                    fmt = 'online/offline'
                elif val_str in ['running', 'stopped']:
                    fmt = 'running/stopped'
                elif val_str in ['approved', 'rejected']:
                    fmt = 'approved/rejected'
                elif val_str in ['accepted', 'rejected']:
                    fmt = 'accepted/rejected'
                elif val_str == 'declined':
                    fmt = 'accepted/declined'
                elif val_str in ['pass', 'passed', 'fail', 'failed']:
                    fmt = 'pass/fail'
                elif val_str in ['success', 'failure']:
                    fmt = 'success/failure'
                elif val_str in ['granted', 'denied']:
                    fmt = 'granted/denied'
                elif val_str in ['set', 'unset']:
                    fmt = 'set/unset'
                elif val_str in ['available', 'unavailable']:
                    fmt = 'available/unavailable'
                elif val_str in ['in stock', 'in-stock', 'out of stock', 'out-of-stock']:
                    fmt = 'in stock/out of stock'
                elif val_str in ['present', 'absent']:
                    fmt = 'present/absent'
                elif val_str in ['ready', 'not ready']:
                    fmt = 'ready/not ready'
                else:
                    fmt = 'true/false'
                    
                parsed_data.append((bool_val, fmt))
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
                
            elif val_str in self.FALSE_VALUES:
                bool_val = 0  # Always use 0 for false
                
                # Detect format type for column renaming
                if val_str in ['false', 'f']:
                    fmt = 'true/false'
                elif val_str in ['no', 'n']:
                    fmt = 'yes/no'
                elif val_str == '0':
                    fmt = '1/0'
                elif val_str == 'off':
                    fmt = 'on/off'
                elif val_str == 'inactive':
                    fmt = 'active/inactive'
                elif val_str == 'disabled':
                    fmt = 'enabled/disabled'
                elif val_str == 'offline':
                    fmt = 'online/offline'
                elif val_str == 'stopped':
                    fmt = 'running/stopped'
                elif val_str == 'rejected':
                    fmt = 'accepted/rejected'
                elif val_str == 'declined':
                    fmt = 'accepted/declined'
                elif val_str in ['fail', 'failed']:
                    fmt = 'pass/fail'
                elif val_str == 'failure':
                    fmt = 'success/failure'
                elif val_str == 'denied':
                    fmt = 'granted/denied'
                elif val_str == 'unset':
                    fmt = 'set/unset'
                elif val_str == 'unavailable':
                    fmt = 'available/unavailable'
                elif val_str in ['out of stock', 'out-of-stock']:
                    fmt = 'in stock/out of stock'
                elif val_str == 'absent':
                    fmt = 'present/absent'
                elif val_str == 'not ready':
                    fmt = 'ready/not ready'
                else:
                    fmt = 'true/false'
                    
                parsed_data.append((bool_val, fmt))
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            else:
                parsed_data.append((np.nan, None))
        
        # Determine most common format for column naming
        if format_counts:
            most_common_format = max(format_counts, key=format_counts.get)
            self.detected_format = most_common_format
        else:
            self.detected_format = '1/0'
        
        # Second pass: convert all to 1/0 (already done in first pass)
        result = [data[0] for data in parsed_data]  # Just extract the 1/0 value
        # Use Int64 (nullable integer) to preserve integer type while supporting NaN
        return pd.Series(result, index=values.index, dtype='Int64')
