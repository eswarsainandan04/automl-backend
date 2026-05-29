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

        # Availability patterns

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
        # Keep original values unchanged.
        # This pattern is only used for detection; no boolean remapping or format tagging.
        self.detected_format = None
        return values.copy()
