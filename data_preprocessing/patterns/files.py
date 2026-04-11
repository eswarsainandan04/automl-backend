"""
Files Pattern - Detects filenames with extensions
Detection only - does not normalize
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern

class FilesPattern(BasePattern):
    
    regex_patterns = [
        # Common programming files
        r'^[a-zA-Z0-9_\-\.]+\.py$',
        r'^[a-zA-Z0-9_\-\.]+\.java$',
        r'^[a-zA-Z0-9_\-\.]+\.c$',
        r'^[a-zA-Z0-9_\-\.]+\.cpp$',
        r'^[a-zA-Z0-9_\-\.]+\.h$',
        r'^[a-zA-Z0-9_\-\.]+\.js$',
        r'^[a-zA-Z0-9_\-\.]+\.ts$',
        r'^[a-zA-Z0-9_\-\.]+\.jsx$',
        r'^[a-zA-Z0-9_\-\.]+\.tsx$',
        r'^[a-zA-Z0-9_\-\.]+\.php$',
        r'^[a-zA-Z0-9_\-\.]+\.rb$',
        r'^[a-zA-Z0-9_\-\.]+\.go$',
        r'^[a-zA-Z0-9_\-\.]+\.rs$',
        r'^[a-zA-Z0-9_\-\.]+\.swift$',
        r'^[a-zA-Z0-9_\-\.]+\.kt$',
        r'^[a-zA-Z0-9_\-\.]+\.scala$',
        
        # Data files
        r'^[a-zA-Z0-9_\-\.]+\.csv$',
        r'^[a-zA-Z0-9_\-\.]+\.json$',
        r'^[a-zA-Z0-9_\-\.]+\.xml$',
        r'^[a-zA-Z0-9_\-\.]+\.yaml$',
        r'^[a-zA-Z0-9_\-\.]+\.yml$',
        r'^[a-zA-Z0-9_\-\.]+\.sql$',
        r'^[a-zA-Z0-9_\-\.]+\.db$',
        r'^[a-zA-Z0-9_\-\.]+\.sqlite$',
        
        # Document files
        r'^[a-zA-Z0-9_\-\.]+\.txt$',
        r'^[a-zA-Z0-9_\-\.]+\.md$',
        r'^[a-zA-Z0-9_\-\.]+\.pdf$',
        r'^[a-zA-Z0-9_\-\.]+\.doc$',
        r'^[a-zA-Z0-9_\-\.]+\.docx$',
        r'^[a-zA-Z0-9_\-\.]+\.xls$',
        r'^[a-zA-Z0-9_\-\.]+\.xlsx$',
        r'^[a-zA-Z0-9_\-\.]+\.ppt$',
        r'^[a-zA-Z0-9_\-\.]+\.pptx$',
        
        # Web files
        r'^[a-zA-Z0-9_\-\.]+\.html$',
        r'^[a-zA-Z0-9_\-\.]+\.htm$',
        r'^[a-zA-Z0-9_\-\.]+\.css$',
        r'^[a-zA-Z0-9_\-\.]+\.scss$',
        r'^[a-zA-Z0-9_\-\.]+\.sass$',
        r'^[a-zA-Z0-9_\-\.]+\.less$',
        
        # Image files
        r'^[a-zA-Z0-9_\-\.]+\.jpg$',
        r'^[a-zA-Z0-9_\-\.]+\.jpeg$',
        r'^[a-zA-Z0-9_\-\.]+\.png$',
        r'^[a-zA-Z0-9_\-\.]+\.gif$',
        r'^[a-zA-Z0-9_\-\.]+\.bmp$',
        r'^[a-zA-Z0-9_\-\.]+\.svg$',
        r'^[a-zA-Z0-9_\-\.]+\.ico$',
        r'^[a-zA-Z0-9_\-\.]+\.webp$',
        
        # Video files
        r'^[a-zA-Z0-9_\-\.]+\.mp4$',
        r'^[a-zA-Z0-9_\-\.]+\.avi$',
        r'^[a-zA-Z0-9_\-\.]+\.mov$',
        r'^[a-zA-Z0-9_\-\.]+\.mkv$',
        r'^[a-zA-Z0-9_\-\.]+\.wmv$',
        r'^[a-zA-Z0-9_\-\.]+\.flv$',
        
        # Audio files
        r'^[a-zA-Z0-9_\-\.]+\.mp3$',
        r'^[a-zA-Z0-9_\-\.]+\.wav$',
        r'^[a-zA-Z0-9_\-\.]+\.flac$',
        r'^[a-zA-Z0-9_\-\.]+\.aac$',
        r'^[a-zA-Z0-9_\-\.]+\.ogg$',
        r'^[a-zA-Z0-9_\-\.]+\.wma$',
        
        # Archive files
        r'^[a-zA-Z0-9_\-\.]+\.zip$',
        r'^[a-zA-Z0-9_\-\.]+\.rar$',
        r'^[a-zA-Z0-9_\-\.]+\.tar$',
        r'^[a-zA-Z0-9_\-\.]+\.gz$',
        r'^[a-zA-Z0-9_\-\.]+\.7z$',
        r'^[a-zA-Z0-9_\-\.]+\.bz2$',
        
        # Executable files
        r'^[a-zA-Z0-9_\-\.]+\.exe$',
        r'^[a-zA-Z0-9_\-\.]+\.dll$',
        r'^[a-zA-Z0-9_\-\.]+\.so$',
        r'^[a-zA-Z0-9_\-\.]+\.app$',
        r'^[a-zA-Z0-9_\-\.]+\.dmg$',
        
        # Config files
        r'^[a-zA-Z0-9_\-\.]+\.ini$',
        r'^[a-zA-Z0-9_\-\.]+\.conf$',
        r'^[a-zA-Z0-9_\-\.]+\.config$',
        r'^[a-zA-Z0-9_\-\.]+\.env$',
        
        # Log / data / backup files
        r'^[a-zA-Z0-9_\-\.]+\.log$',
        r'^[a-zA-Z0-9_\-\.]+\.dat$',
        r'^[a-zA-Z0-9_\-\.]+\.bak$',
        r'^[a-zA-Z0-9_\-\.]+\.tmp$',
        r'^[a-zA-Z0-9_\-\.]+\.lock$',
        r'^[a-zA-Z0-9_\-\.]+\.pid$',
        r'^[a-zA-Z0-9_\-\.]+\.out$',
        r'^[a-zA-Z0-9_\-\.]+\.err$',
        
        # Generic catch-all: any name.extension (1-10 char extension)
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = 'filename'
    
    def detect(self, values):
        """
        Detect if column contains filenames
        Returns confidence as percentage of valid filenames
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
        
        # Check how many values match filename patterns
        matched = 0
        for val in non_null:
            # Skip missing value indicators
            if val.lower() in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                continue
            
            # Skip if looks like a path (contains / or \)
            if '/' in val or '\\' in val:
                continue
            
            # Check against filename regex patterns (case-insensitive)
            is_file = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val, re.IGNORECASE):
                    is_file = True
                    break
            
            if is_file:
                matched += 1
        
        # Return confidence as percentage
        total = len(non_null)
        return matched / total if total > 0 else 0.0
    
    def normalize(self, values):
        """
        For filenames, return original values unchanged (detection only)
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        # Return as-is without any modifications
        return values
