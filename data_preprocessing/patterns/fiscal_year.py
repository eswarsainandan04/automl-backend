import re
import pandas as pd
import numpy as np
from .base import BasePattern


class FiscalYearPattern(BasePattern):
    semantic_type = "fiscal_year"
    
    regex_patterns = [
        r'^FY?\s*\d{4}$',  # FY2024, F2023
        r'^FY?\s*\d{2}$',  # FY23, F23
        r'^FY?\s*-\d{2,4}$',  # FY-25, F-2024, FY-24
        r'^FY?\s*\d{4}-\d{2,4}$',  # FY2024-25, F2024-2025
        r'^FY?\s*\d{2}-\d{2}$',  # FY23-24, F23-24
        r'^FY?\s*\d{4}/\d{2,4}$',  # FY2024/25, FY2024/2025
        r'^FY?\s*\d{2}/\d{2}$',  # FY23/24
        r'^Fiscal\s*Year\s*\d{4}$',  # Fiscal Year 2024
        r'^\d{4}-\d{2,4}\s*FY?$',  # 2024-25 FY, 2024-25 F, 2024-2025 FY
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        # Check if column is named "fiscal_year", "fiscal", "fy" (case-insensitive)
        column_name = ""
        if hasattr(values, 'name') and values.name:
            column_name = str(values.name).lower()
        
        is_fiscal_column = any(keyword in column_name for keyword in ['fiscal', 'fy'])
        
        matched = 0
        total = 0
        
        for value in values:
            if pd.isna(value):
                continue
            total += 1
            value_str = str(value).strip()
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        
        confidence = matched / total
        
        # Boost confidence if column is named fiscal_year
        if is_fiscal_column:
            confidence = min(1.0, confidence * 1.5)  # 50% boost
        
        return confidence
    
    def normalize(self, values):
        def parse_fiscal_year(val):
            if pd.isna(val):
                return np.nan
            try:
                val_str = str(val).strip()
                
                # Extract all year numbers
                years = re.findall(r'\d{4}|\d{2}', val_str)
                
                if len(years) == 0:
                    return np.nan
                
                # Convert to 4-digit years
                normalized_years = []
                for year_str in years:
                    year = int(year_str)
                    if year < 100:
                        if year < 50:
                            year += 2000
                        else:
                            year += 1900
                    normalized_years.append(year)
                
                # Standardize format to FY-YYYY (single year format for consistency)
                # Use first year as the fiscal year
                start_year = normalized_years[0]
                return f"FY-{start_year}"
            except:
                return np.nan
        
        return values.apply(parse_fiscal_year)
