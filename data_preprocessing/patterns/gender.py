"""
Gender Pattern - Detects and normalizes gender-related values
Normalizes to 'male', 'female', 'transgender', or 'other' based on most common format in column
"""

import re
import pandas as pd
import numpy as np
from .base import BasePattern
from collections import Counter

class GenderPattern(BasePattern):
    
    regex_patterns = [
        # Male patterns
        r'^m(ale)?$',
        r'^man$',
        r'^boy$',
        r'^gentleman$',
        r'^guy$',
        r'^he$',
        r'^him$',
        r'^his$',
        r'^mr\.?$',
        r'^male$',
        
        # Female patterns
        r'^f(emale)?$',
        r'^woman$',
        r'^girl$',
        r'^lady$',
        r'^she$',
        r'^her$',
        r'^hers$',
        r'^mrs?\.?$',
        r'^miss$',
        r'^ms\.?$',
        r'^female$',
        
        # Transgender patterns
        r'^trans(gender)?$',
        r'^trans\s*m(ale)?$',
        r'^trans\s*f(emale)?$',
        r'^trans\s*man$',
        r'^trans\s*woman$',
        r'^ftm$',
        r'^mtf$',
        
        # Other/Non-binary patterns
        r'^other$',
        r'^non[\s\-]?binary$',
        r'^non[\s\-]?conforming$',
        r'^genderqueer$',
        r'^gender[\s\-]?fluid$',
        r'^agender$',
        r'^bigender$',
        r'^pangender$',
        r'^nb$',
        r'^enby$',
        r'^genderless$',
        r'^neutral$',
        r'^third[\s\-]?gender$',
        r'^two[\s\-]?spirit$',
    ]
    
    # Male value variations
    MALE_VALUES = [
        'm', 'male', 'man', 'boy', 'gentleman', 'guy',
        'he', 'him', 'his', 'mr', 'mr.',
    ]
    
    # Female value variations
    FEMALE_VALUES = [
        'f', 'female', 'woman', 'girl', 'lady',
        'she', 'her', 'hers', 'mrs', 'mrs.', 'mr.', 'miss', 'ms', 'ms.',
    ]
    
    # Transgender value variations
    TRANSGENDER_VALUES = [
        'trans', 'transgender', 'trans male', 'trans female',
        'trans man', 'trans woman', 'transman', 'transwoman',
        'ftm', 'mtf',
    ]
    
    # Other/Non-binary value variations
    OTHER_VALUES = [
        'other', 'non-binary', 'non binary', 'nonbinary',
        'non-conforming', 'non conforming', 'nonconforming',
        'genderqueer', 'gender-fluid', 'gender fluid', 'genderfluid',
        'agender', 'bigender', 'pangender', 'nb', 'enby',
        'genderless', 'neutral', 'third gender', 'third-gender',
        'two spirit', 'two-spirit',
    ]
    
    def __init__(self):
        super().__init__()
        self.detected_format = None
    
    def detect(self, values):
        """
        Detect if column contains gender values
        Returns 1.0 (100%) if all non-null values match gender patterns
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        if values.empty:
            return 0.0
        
        # Check if column is named "gender" or "sex" (case-insensitive)
        column_name = ""
        if hasattr(values, 'name') and values.name:
            column_name = str(values.name).lower()
        
        is_gender_column = any(keyword in column_name for keyword in ['gender', 'sex'])
        
        # Filter out null values and convert to string
        non_null = values.dropna().astype(str).str.strip()
        
        if len(non_null) == 0:
            return 0.0
        
        # Check how many values match gender patterns
        matched = 0
        for val in non_null:
            val_lower = val.lower()
            
            # Skip if value is "?" or similar missing indicators
            if val_lower in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                continue
            
            # Check against all regex patterns
            is_match = False
            for pattern in self.regex_patterns:
                if re.match(pattern, val_lower, re.IGNORECASE):
                    is_match = True
                    break
            
            if is_match:
                matched += 1
        
        # Return confidence as percentage
        total = len(non_null)
        confidence = matched / total if total > 0 else 0.0
        
        # Boost confidence if column is named "gender" or "sex"
        if is_gender_column:
            confidence = min(1.0, confidence * 1.2)  # 20% boost
        
        return confidence
    
    def normalize(self, values):
        """
        Normalize gender values to standard categories: 'male', 'female', 'transgender', or 'other'
        Uses two-pass approach:
        1. Parse values and detect which format is most common
        2. Normalize all values to standard categories
        """
        # Convert to Series if needed
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        
        if values.empty:
            return values
        
        # First pass: parse values and determine formats
        parsed_data = []
        format_counts = Counter()
        
        for idx, val in values.items():
            # Handle NaN/None
            if pd.isna(val):
                parsed_data.append((np.nan, None))
                continue
            
            # Convert to string and clean
            val_str = str(val).strip().lower()
            
            # Handle missing value indicators
            if val_str in ['?', 'na', 'n/a', 'nan', 'null', 'none', '']:
                parsed_data.append((np.nan, None))
                continue
            
            # Determine gender value and format
            gender_val = None
            format_type = None
            
            # Check male values
            if val_str in self.MALE_VALUES:
                gender_val = 'male'
                
                # Determine format type
                if val_str in ['m']:
                    format_type = 'm/f/o/t'
                elif val_str in ['male']:
                    format_type = 'male/female/other/transgender'
                elif val_str in ['man', 'woman']:
                    format_type = 'man/woman/other/transgender'
                elif val_str in ['boy', 'girl']:
                    format_type = 'boy/girl/other/transgender'
                elif val_str in ['he', 'she']:
                    format_type = 'he/she/other/transgender'
                elif val_str in ['him', 'her']:
                    format_type = 'him/her/other/transgender'
                else:
                    format_type = 'male/female/other/transgender'
                
                format_counts[format_type] += 1
                parsed_data.append((gender_val, format_type))
            
            # Check female values
            elif val_str in self.FEMALE_VALUES:
                gender_val = 'female'
                
                # Determine format type
                if val_str in ['f']:
                    format_type = 'm/f/o/t'
                elif val_str in ['female']:
                    format_type = 'male/female/other/transgender'
                elif val_str in ['woman', 'lady']:
                    format_type = 'man/woman/other/transgender'
                elif val_str in ['girl']:
                    format_type = 'boy/girl/other/transgender'
                elif val_str in ['she']:
                    format_type = 'he/she/other/transgender'
                elif val_str in ['her', 'hers']:
                    format_type = 'him/her/other/transgender'
                else:
                    format_type = 'male/female/other/transgender'
                
                format_counts[format_type] += 1
                parsed_data.append((gender_val, format_type))
            
            # Check transgender values
            elif val_str in self.TRANSGENDER_VALUES:
                gender_val = 'transgender'
                format_type = 'male/female/other/transgender'
                format_counts[format_type] += 1
                parsed_data.append((gender_val, format_type))
            
            # Check other/non-binary values
            elif val_str in self.OTHER_VALUES:
                gender_val = 'other'
                format_type = 'male/female/other/transgender'
                format_counts[format_type] += 1
                parsed_data.append((gender_val, format_type))
            
            else:
                # Unknown value
                parsed_data.append((np.nan, None))
        
        # Determine most common format for column naming
        if format_counts:
            most_common_format = max(format_counts, key=format_counts.get)
            self.detected_format = most_common_format
        else:
            self.detected_format = 'male/female/other/transgender'
        
        # Second pass: normalize all to standard gender categories
        result = []
        for gender_val, fmt in parsed_data:
            if pd.isna(gender_val):
                result.append(np.nan)
            else:
                result.append(gender_val)  # Return 'male', 'female', 'transgender', or 'other'
        
        # Return as object dtype to preserve string type with NaN
        return pd.Series(result, index=values.index, dtype='object')
