import re
import pandas as pd
import numpy as np
from .base import BasePattern


class NumberSystemsPattern(BasePattern):
    semantic_type = "number_systems"
    
    regex_patterns = [
        # Hexadecimal
        r'^0x[0-9A-Fa-f]+$',  # 0xFF, 0x1A3
        r'^[0-9A-Fa-f]+h$',  # FFh, 1A3h
        # Binary  
        r'^0b[01]+$',  # 0b1010
        r'^[01]+b$',  # 1010b
        r'^[01]{4,}$',  # Plain binary strings (4+ digits to avoid confusion with decimal 0,1,10,11,etc.)
        # Octal
        r'^0o[0-7]+$',  # 0o755
        r'^0[0-7]+$',  # 0755
        # Roman Numerals (comprehensive)
        r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',  # I, V, X, L, C, D, M
        r'^[IVXLCDM]+$',  # Roman numerals (will validate in normalize)
        # Scientific notation
        r'^\d+\.?\d*[eE][+-]?\d+$',  # 1.5e10, 3E-5
        # Text numbers (million, thousand, lakh, crore, etc.)
        r'^\d+(\.\d+)?\s*(million|billion|trillion|thousand|hundred|lakh|lakhs|crore|crores|k|m|b)$',
        r'^(one|two|three|four|five|six|seven|eight|nine|ten)\s*(million|billion|thousand|hundred|lakh|crore)?$',
    ]
    
    # Roman numeral values
    ROMAN_VALUES = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    # Unicode Roman numerals (U+2160 to U+2188)
    UNICODE_ROMAN_MAP = {
        'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4, 'Ⅴ': 5, 'Ⅵ': 6, 'Ⅶ': 7, 'Ⅷ': 8, 'Ⅸ': 9, 'Ⅹ': 10,
        'Ⅺ': 11, 'Ⅻ': 12,  # XI, XII
        'ⅰ': 1, 'ⅱ': 2, 'ⅲ': 3, 'ⅳ': 4, 'ⅴ': 5, 'ⅵ': 6, 'ⅶ': 7, 'ⅷ': 8, 'ⅸ': 9, 'ⅹ': 10,
        'ⅺ': 11, 'ⅻ': 12,  # xi, xii (lowercase)
    }
    
    def _is_valid_roman(self, s: str) -> bool:
        """Check if string is a valid roman numeral"""
        if not s or not all(c in 'IVXLCDM' for c in s.upper()):
            return False
        # Basic validation - detailed conversion in normalize
        return bool(re.match(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', s.upper()))
    
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
            try:
                if isinstance(value, float):
                    value_str = str(value)
                else:
                    value_str = str(value).strip()
            except:
                value_str = str(value).strip()
            
            # Check for Unicode Roman numerals
            if value_str in self.UNICODE_ROMAN_MAP:
                matched += 1
                continue
            
            # Check for roman numerals specifically
            if self._is_valid_roman(value_str):
                matched += 1
                continue
                
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def normalize(self, values):
        def convert_number_system(val):
            """Convert various number systems to decimal"""
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Hexadecimal
                if re.match(r'^0x[0-9A-Fa-f]+$', val_str, re.IGNORECASE):
                    return int(val_str, 16)
                if re.match(r'^[0-9A-Fa-f]+h$', val_str, re.IGNORECASE):
                    return int(val_str[:-1], 16)
                
                # Binary
                if re.match(r'^0b[01]+$', val_str, re.IGNORECASE):
                    return int(val_str, 2)
                if re.match(r'^[01]+b$', val_str, re.IGNORECASE):
                    return int(val_str[:-1], 2)
                # Plain binary strings (4+ digits of only 0s and 1s)
                if re.match(r'^[01]{4,}$', val_str):
                    return int(val_str, 2)
                # Single digits 0 or 1 (ambiguous but treat as decimal in context of number systems)
                if val_str in ['0', '1']:
                    return int(val_str)
                
                # Octal
                if re.match(r'^0o[0-7]+$', val_str, re.IGNORECASE):
                    return int(val_str, 8)
                if re.match(r'^0[0-7]+$', val_str) and len(val_str) > 1:
                    return int(val_str, 8)
                
                # Unicode Roman Numerals (check first - single characters)
                if val_str in self.UNICODE_ROMAN_MAP:
                    return self.UNICODE_ROMAN_MAP[val_str]
                
                # Roman Numerals
                if self._is_valid_roman(val_str):
                    return self._roman_to_int(val_str.upper())
                
                # Scientific notation
                if re.match(r'^\d+\.?\d*[eE][+-]?\d+$', val_str):
                    return int(float(val_str))
                
                # Text numbers (million, lakh, thousand, etc.)
                text_num = self._convert_text_number(val_str)
                if text_num is not None:
                    return text_num
                
                # If nothing matched, try to return as-is
                return val_str
            except:
                return np.nan
        
        return values.apply(convert_number_system)
    
    def _roman_to_int(self, s: str) -> int:
        """Convert roman numeral to integer"""
        if not s:
            return 0
        
        total = 0
        prev_value = 0
        
        for char in reversed(s):
            value = self.ROMAN_VALUES.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        return total
    
    def _convert_text_number(self, val_str: str):
        """Convert text numbers like '1 million', '5 lakh', '2k' to integers"""
        try:
            val_lower = val_str.lower().strip()
            
            # Word to number mapping
            word_numbers = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            
            # Multiplier mapping
            multipliers = {
                'hundred': 100,
                'thousand': 1000,
                'lakh': 100000,
                'lakhs': 100000,
                'crore': 10000000,
                'crores': 10000000,
                'million': 1000000,
                'billion': 1000000000,
                'trillion': 1000000000000,
                'k': 1000,
                'm': 1000000,
                'b': 1000000000
            }
            
            # Check for patterns like "1 million", "5.5 lakh", "2k"
            pattern = r'^(\d+(?:\.\d+)?)\s*(million|billion|trillion|thousand|hundred|lakh|lakhs|crore|crores|k|m|b)$'
            match = re.match(pattern, val_lower)
            if match:
                number = float(match.group(1))
                multiplier_word = match.group(2)
                multiplier = multipliers.get(multiplier_word, 1)
                return int(number * multiplier)
            
            # Check for word numbers like "one million", "five lakh"
            pattern2 = r'^(one|two|three|four|five|six|seven|eight|nine|ten)\s*(million|billion|thousand|hundred|lakh|crore)?$'
            match2 = re.match(pattern2, val_lower)
            if match2:
                number = word_numbers.get(match2.group(1), 1)
                multiplier_word = match2.group(2) if match2.group(2) else None
                multiplier = multipliers.get(multiplier_word, 1) if multiplier_word else 1
                return int(number * multiplier)
            
            return None
        except:
            return None
