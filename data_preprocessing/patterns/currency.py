import re
import pandas as pd
import numpy as np
from .base import BasePattern


class CurrencyPattern(BasePattern):
    semantic_type = "currency"
    detected_currency = None  # Will store single currency code for column renaming
    has_mixed_currencies = False  # Flag for multiple currency types in same column
    currency_codes = None  # Series containing currency code for each row (if mixed)
    
    regex_patterns = [
        # Discount/Save formats: "Save $1.23" or "Discount $45.67"
        r'^(?:save|discount|off)\s+\$\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        r'^(?:save|discount|off)\s+€\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        r'^(?:save|discount|off)\s+£\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        r'^(?:save|discount|off)\s+¥\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        r'^(?:save|discount|off)\s+₹\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # US format: $1,234.56 or just $1234.56
        r'^\$\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Amount with $ symbol (postfix, no space): 4$, 1234$
        r'^\d+(?:,\d{3})*(?:\.\d+)?\$$',
        # Amount with space then $ symbol: 4642 $
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s+\$$',
        # EU format: €1.234,56 or €1 234,56 (dot/space thousands, comma decimals)
        r'^\€\s*\d+(?:[., ]\d{3})*(?:,\d{2})?$',
        r'^\€\s*\d+(?:\.\d{3})*(?:,\d{2})?$',
        # Amount with space then € symbol: 4642 €
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s+€$',
        # UK format: £1,234.56
        r'^£\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Amount with space then £ symbol: 4642 £
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s+£$',
        # Japanese Yen: ¥1,234
        r'^¥\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Amount with space then ¥ symbol
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s+¥$',
        # Indian Rupee: ₹1,234.56
        r'^₹\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Amount with space then ₹ symbol
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s+₹$',
        # Indian lakh format: ₹1,00,000
        r'^₹\s*\d+(?:,\d{2,3})+$',
        # South Korean Won: ₩1,234
        r'^₩\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Russian Ruble: ₽1,234.56
        r'^₽\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Indian/Pakistani Rupee: ₨1,234 or Rs 1,234
        r'^₨\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        r'^Rs\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Amount then Rs with space: 6786 Rs
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s+Rs$',
        # Bangladesh Taka: ৳1,234
        r'^৳\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Thai Baht: ฿1,234.56
        r'^฿\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Philippine Peso: ₱1,234.56
        r'^₱\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Vietnamese Dong: ₫1,234
        r'^₫\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Israeli Shekel: ₪1,234.56
        r'^₪\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Nigerian Naira: ₦1,234.56
        r'^₦\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Brazilian Real: R$1,234.56
        r'^R\$\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Indonesian Rupiah: Rp1,234
        r'^Rp\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Nordic currencies (SEK, NOK, DKK, ISK): 1,234 kr or kr 1,234
        r'^\d+(?:,\d{3})*(?:\.\d+)?\s*kr$',
        r'^kr\s*\d+(?:,\d{3})*(?:\.\d+)?$',
        # Code-based formats (currency code before or after number) - with optional commas
        r'^\d+(?:,\d{2,3})*(?:\.\d+)?\s+(USD|EUR|GBP|JPY|INR|AUD|CAD|CHF|KRW|RUB|BDT|THB|PHP|VND|ILS|NGN|BRL|IDR|SEK|NOK|DKK|ISK)$',
        r'^(USD|EUR|GBP|JPY|INR|AUD|CAD|CHF|KRW|RUB|BDT|THB|PHP|VND|ILS|NGN|BRL|IDR|SEK|NOK|DKK|ISK)\s*\d+(?:,\d{2,3})*(?:\.\d+)?$',
        r'^\d+(?:\.\d+)?\s+(USD|EUR|GBP|JPY|INR|AUD|CAD|CHF|KRW|RUB|BDT|THB|PHP|VND|ILS|NGN|BRL|IDR|SEK|NOK|DKK|ISK)$',
        r'^(USD|EUR|GBP|JPY|INR|AUD|CAD|CHF|KRW|RUB|BDT|THB|PHP|VND|ILS|NGN|BRL|IDR|SEK|NOK|DKK|ISK)\s+\d+(?:\.\d+)?$',
    ]
    
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        
        matched = 0
        total = 0
        has_symbol = False  # Track if we find any currency symbols
        
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
            
            # Check if value has currency symbols
            if any(sym in value_str for sym in ['$', '€', '£', '¥', '₹', '₩', '₽', '₨', '৳', '฿', '₱', '₫', '₪', '₦']):
                has_symbol = True
            
            # Check for EU currency format specifically (€ with European separators)
            # Example: "€1.200,50" or "€2.500,00"
            if '€' in value_str or '�' in value_str:  # � is mojibake for €
                # EU format: dot for thousands, comma for decimals
                if re.match(r'^[€�]\s*\d+(?:\.\d{3})*(?:,\d{2})?$', value_str):
                    matched += 1
                    continue
                # Also check space separator variant
                if re.match(r'^[€�]\s*\d+(?:\s\d{3})*(?:,\d{2})?$', value_str):
                    matched += 1
                    continue
                
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        
        # Only accept as currency if we found currency symbols OR currency codes
        # Don't match plain numbers without any currency indicators
        if not has_symbol and matched < total * 0.5:
            return 0.0
        
        return matched / total
    
    def normalize(self, values):
        # Detect which currency symbols/codes are present
        currency_types = set()
        
        # First pass: detect all currency types present
        for val in values:
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if val_str.lower() in ['null', 'nan', 'none', '']:
                continue
            
            # Map symbols to codes
            if '$' in val_str:
                currency_types.add('USD')
            if '€' in val_str or '�' in val_str:  # � is mojibake for €
                currency_types.add('EUR')
            if '£' in val_str:
                currency_types.add('GBP')
            if '¥' in val_str:
                currency_types.add('JPY')
            if '₹' in val_str:
                currency_types.add('INR')
            if '₩' in val_str:
                currency_types.add('KRW')

            if '₽' in val_str:
                currency_types.add('RUB')

            if '₨' in val_str or re.search(r'\bRs\b', val_str):
                currency_types.add('INR')  # Treat as INR by default

            if '৳' in val_str:
                currency_types.add('BDT')

            if '฿' in val_str:
                currency_types.add('THB')

            if '₱' in val_str:
                currency_types.add('PHP')

            if '₫' in val_str:
                currency_types.add('VND')

            if '₪' in val_str:
                currency_types.add('ILS')

            if '₦' in val_str:
                currency_types.add('NGN')

            if 'R$' in val_str:
                currency_types.add('BRL')

            if 'Rp' in val_str:
                currency_types.add('IDR')

            if 'kr' in val_str.lower():
                currency_types.add('NORDIC_KR') 
            
            # Check for currency codes (common ones)
            code_match = re.search(r'\b(USD|EUR|GBP|JPY|INR|AUD|CAD|CHF|KRW|RUB|BDT|THB|PHP|VND|ILS|NGN|BRL|IDR)\b', val_str, re.IGNORECASE)
            if code_match:
                currency_types.add(code_match.group(1).upper())
        
        # Determine if mixed currencies
        if len(currency_types) > 1:
            # MIXED CURRENCIES: Store currency code for each row
            self.has_mixed_currencies = True
            self.detected_currency = None
            
            def extract_currency_code(val):
                """Extract the currency code from value"""
                if pd.isna(val):
                    return np.nan
                val_str = str(val).strip()
                
                # Check symbols
                if '$' in val_str:
                    return 'USD'
                if '€' in val_str or '�' in val_str:
                    return 'EUR'
                if '£' in val_str:
                    return 'GBP'
                if '¥' in val_str:
                    return 'JPY'
                if '₹' in val_str:
                    return 'INR'
                if '₩' in val_str:
                    return 'KRW'
                if '₽' in val_str:
                    return 'RUB'
                if '₨' in val_str or re.search(r'\bRs\b', val_str):
                    return 'INR'
                if '৳' in val_str:
                    return 'BDT'
                if '฿' in val_str:
                    return 'THB'
                if '₱' in val_str:
                    return 'PHP'
                if '₫' in val_str:
                    return 'VND'
                if '₪' in val_str:
                    return 'ILS'
                if '₦' in val_str:
                    return 'NGN'
                if 'R$' in val_str:
                    return 'BRL'
                if 'Rp' in val_str:
                    return 'IDR'
                
                # Check codes
                code_match = re.search(r'\b(USD|EUR|GBP|JPY|INR|AUD|CAD|CHF|KRW|RUB|BDT|THB|PHP|VND|ILS|NGN|BRL|IDR)\b', val_str, re.IGNORECASE)
                if code_match:
                    return code_match.group(1).upper()
                
                return np.nan
            
            # Store currency codes for each row
            self.currency_codes = values.apply(extract_currency_code)
            
        elif len(currency_types) == 1:
            # SINGLE CURRENCY: Normal behavior
            self.has_mixed_currencies = False
            self.detected_currency = list(currency_types)[0]
            self.currency_codes = None
        else:
            # NO CURRENCY DETECTED
            self.has_mixed_currencies = False
            self.detected_currency = None
            self.currency_codes = None
        
        # First pass: Parse all values to detect if any have decimals
        parsed_values = []
        has_float_values = False
        
        for val in values:
            if pd.isna(val):
                parsed_values.append(np.nan)
                continue
                
            try:
                val_str = str(val).strip()
                
                # Remove discount/save prefixes if present
                val_str = re.sub(r'^(?:save|discount|off)\s+', '', val_str, flags=re.IGNORECASE)
                
                # Handle multipliers
                multiplier = 1
                val_lower = val_str.lower()
                
                if 'cr' in val_lower or 'crore' in val_lower:
                    multiplier = 10_000_000
                    val_str = re.sub(r'(cr|crore)', '', val_str, flags=re.IGNORECASE)
                elif 'lakh' in val_lower or 'lac' in val_lower:
                    multiplier = 100_000
                    val_str = re.sub(r'(lakh|lac)', '', val_str, flags=re.IGNORECASE)
                elif 'k' in val_lower and not any(x in val_lower for x in ['usd', 'eur', 'gbp', 'inr', 'jpy']):
                    multiplier = 1000
                    val_str = re.sub(r'k', '', val_str, flags=re.IGNORECASE)
                elif 'm' in val_lower and 'mill' not in val_lower and not any(x in val_lower for x in ['usd', 'eur', 'gbp', 'inr', 'jpy']):
                    multiplier = 1_000_000
                    val_str = re.sub(r'm', '', val_str, flags=re.IGNORECASE)
                
                # Remove ALL non-numeric chars except digits, comma, period, minus
                val_str = re.sub(r'[^\d,.\-]', '', val_str)
                
                # CRITICAL: Detect format PER VALUE (not per column) to handle mixed EU/US formats
                is_eu_format = bool(re.search(r'\d+\.\d{3}', val_str) and re.search(r',\d{2}$', val_str))
                
                if is_eu_format:
                    val_str = val_str.replace('.', '').replace(',', '.')
                elif ',' in val_str and '.' in val_str:
                    last_comma_pos = val_str.rfind(',')
                    last_period_pos = val_str.rfind('.')
                    if last_period_pos > last_comma_pos:
                        val_str = val_str.replace(',', '')
                    else:
                        val_str = val_str.replace('.', '').replace(',', '.')
                elif ',' in val_str:
                    parts = val_str.split(',')
                    if len(parts[-1]) == 2:
                        val_str = val_str.replace(',', '.')
                    else:
                        val_str = val_str.replace(',', '')
                
                number = float(val_str) * multiplier
                parsed_values.append(number)
                
                # Check if this number has decimal part
                if number != int(number):
                    has_float_values = True
                    
            except:
                parsed_values.append(np.nan)
        
        # Second pass: Format based on whether any float values detected
        def format_number(num):
            if pd.isna(num):
                return np.nan
            if has_float_values:
                # If ANY value has decimals, format ALL as float with 2 decimals
                return round(num, 2)
            else:
                # If ALL values are integers, keep as int
                if num == int(num):
                    return int(num)
                else:
                    return num
        
        return pd.Series([format_number(num) for num in parsed_values], index=values.index)
