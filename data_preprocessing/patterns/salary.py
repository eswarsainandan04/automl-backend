import re
import pandas as pd
import numpy as np
from collections import Counter
from .base import BasePattern


class SalaryPattern(BasePattern):
    semantic_type = "salary"
    
    # Currency symbols and their codes
    CURRENCY_SYMBOLS = {
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP',
        '₹': 'INR',
        'Rs': 'INR',
        'RS': 'INR',
        'INR': 'INR',
        'USD': 'USD',
        'EUR': 'EUR',
        'GBP': 'GBP',
        'JPY': 'JPY',
        '¥': 'JPY',
        'CNY': 'CNY',
        'AUD': 'AUD',
        'CAD': 'CAD',
        'dollars': 'USD',
        'dollar': 'USD',
        'Dollars': 'USD',
        'Dollar': 'USD',
        'usd': 'USD',
        'eur': 'EUR',
        'gbp': 'GBP',
        'inr': 'INR',
    }
    
    # Exchange rates to INR (as of example - you may want to use an API)
    EXCHANGE_RATES_TO_INR = {
        'USD': 83.12,
        'EUR': 91.23,
        'GBP': 105.45,
        'INR': 1.0,
        'JPY': 0.56,
        'CNY': 11.52,
        'AUD': 55.23,
        'CAD': 61.45,
    }
    
    # Time period multipliers to convert to per year
    TIME_PERIOD_MULTIPLIERS = {
        'hour': 2080,      # 40 hours/week * 52 weeks
        'hr': 2080,
        'h': 2080,
        'day': 260,        # ~260 working days per year
        'daily': 260,
        'pd': 260,
        'week': 52,
        'weekly': 52,
        'wk': 52,
        'pw': 52,
        'month': 12,
        'monthly': 12,
        'mo': 12,
        'pm': 12,
        'year': 1,
        'yearly': 1,
        'annual': 1,
        'annually': 1,
        'annum': 1,
        'yr': 1,
        'pa': 1,
        'p.a.': 1,
        'per annum': 1,
    }
    
    regex_patterns = [
    # 1️⃣ Currency symbol + amount + per/slash + time period (e.g., $200 per annum, $ 10 per day)
        r'[₹$€£¥]\s*\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s*(?:per|/|-|–)\s*(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk)\b',

    # 2️⃣ Currency symbol + amount + time period word (e.g., €10000 yearly, $3000 monthly)
        r'[₹$€£¥]\s*\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s+(?:yearly|monthly|hourly|daily|weekly|annually)\b',

    # 3️⃣ Currency symbol + amount + /abbreviation (e.g., $3000/m, $50/hr)
        r'[₹$€£¥]\s*\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s*/\s*(?:m|mo|h|hr|d|w|wk|y|yr)\b',

    # 4️⃣ Amount + currency CODE + time period (case insensitive) - e.g., "4000 usd pm", "30 dollars per annum"
        r'\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s+(?:Rs|RS|INR|USD|EUR|GBP|JPY|CNY|AUD|CAD|usd|eur|gbp|inr|rs|dollars?|Dollars?)\s+(?:per\s+|/\s*|–\s*|-\s*)?(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk|pa|pm|pw|pd)\b',

    # 5️⃣ Currency CODE + amount + per/slash/space + time period (e.g., "USD 25 / hour", "usd 4000 pm", "USD 20 per hour")
        r'(?:Rs|RS|INR|USD|EUR|GBP|JPY|CNY|AUD|CAD|usd|eur|gbp|inr|rs|dollars?|Dollars?)\s+\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s+(?:per\s+|/\s*|–\s*|-\s*)?(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk|pa|pm|pw|pd)\b',

    # 6️⃣ Currency symbol + amount + suffix (k/M/L/Cr) + period (e.g., $50k/year)
        r'[₹$€£¥]\s*\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s*(?:k|K|L|Cr|cr)\s*(?:per|/|-|–)?\\s*(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk)\b',

    # 7️⃣ Amount + suffix (k) + currency symbol + time period (e.g., "1.5k $ / month")
        r'\d+(?:\.\d{1,2})?\s*(?:k|K)\s+[₹$€£¥]\s*(?:per|/|-|–)\s*(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk)\b',

    # 8️⃣ Amount (with suffix) + currency CODE + period
        r'\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s*(?:k|K|L|Cr|cr)\s*(?:Rs|RS|INR|USD|EUR|GBP|JPY|CNY|AUD|CAD)\s*(?:per|/|-|–)?\s*(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk)\b',
    
    # 9️⃣ LPA pattern (Lakh Per Annum) - e.g., "8 LPA", "12.5 LPA"
        r'\d+(?:\.\d{1,2})?\s*LPA\b',
    
    # 🔟 Amount + Lakh/Cr + per period - e.g., "5 Lakh per month"
        r'\d+(?:\.\d{1,2})?\s*(?:lakh|lakhs|crore|crores)\s*(?:per|/|-|–)?\s*(?:year|yr|annum|annually|month|mo|monthly)\b',
    
    # 1️⃣1️⃣ Amount + space + currency symbol + per/time - e.g., "12 $ per month", "89 $ per year"
        r'\d+(?:[,\s]\d{2,3})*(?:\.\d{1,2})?\s+[₹$€£¥]\s*(?:per|/|-|–)\s*(?:year|yr|annum|annually|month|mo|monthly|hour|hr|h|day|daily|week|weekly|wk)\b',
    

    
    ]

    
    
    def detect(self, values) -> float:
        """
        Detect if column contains salary values with currency symbols and/or time periods.
        """
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
            value_str = str(value).strip()
            
            # Check if it matches any salary pattern
            for pattern in self.regex_patterns:
                if re.search(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        
        if total == 0:
            return 0.0
        return matched / total
    
    def _analyze_salary_distribution(self, values):
        """
        Analyze salary column to determine which scenario applies:
        1. Contains hourly OR weekly rates -> 2 columns (salary, salary_time_unit)
        2. Mixed currency AND mixed time units (non-hourly/weekly) -> 3 columns (salary, salary_time_unit, salary_currency)
        3. Mixed currency but same time unit -> 1 column (salary_{most_common_currency}_per_{time_unit})
        4. Same currency but mixed time units -> 1 column (salary_{currency}_per_{most_common_time_unit})
        
        Returns:
            dict with keys:
                - scenario: 'has_hourly_weekly', 'mixed_both', 'mixed_currency', 'mixed_time', or 'uniform'
                - currencies: set of currencies found
                - time_periods: set of time periods found
                - most_common_currency: most frequent currency
                - most_common_time: most frequent time period
        """
        currencies = []
        time_periods = []
        
        for value in values:
            if pd.notna(value):
                val_str = str(value).strip()
                
                # Extract currency
                currency = self._extract_currency(val_str)
                currencies.append(currency)
                
                # Extract time period
                period, _ = self._extract_time_period(val_str)
                time_periods.append(period)
        
        # Get unique currencies and time periods
        unique_currencies = set(currencies)
        unique_time_periods = set(time_periods)
        
        # Get most common
        from collections import Counter
        currency_counts = Counter(currencies)
        time_counts = Counter(time_periods)
        
        most_common_currency = currency_counts.most_common(1)[0][0] if currency_counts else 'USD'
        most_common_time = time_counts.most_common(1)[0][0] if time_counts else 'year'
        
        # PRIORITY: Check if contains hourly or weekly rates
        hourly_weekly_periods = {'hour', 'hr', 'h', 'week', 'weekly', 'wk', 'pw'}
        has_hourly_or_weekly = bool(unique_time_periods & hourly_weekly_periods)
        
        if has_hourly_or_weekly:
            # If contains hourly or weekly, use special scenario
            scenario = 'has_hourly_weekly'
        else:
            # Original logic for other cases
            has_mixed_currency = len(unique_currencies) > 1
            has_mixed_time = len(unique_time_periods) > 1
            
            if has_mixed_currency and has_mixed_time:
                scenario = 'mixed_both'
            elif has_mixed_currency:
                scenario = 'mixed_currency'
            elif has_mixed_time:
                scenario = 'mixed_time'
            else:
                scenario = 'uniform'
        
        return {
            'scenario': scenario,
            'currencies': unique_currencies,
            'time_periods': unique_time_periods,
            'most_common_currency': most_common_currency,
            'most_common_time': most_common_time,
            'currency_counts': currency_counts,
            'time_counts': time_counts
        }
    
    def _extract_currency(self, value_str):
        """Extract currency code from a salary string."""
        value_str = str(value_str).strip()
        value_lower = value_str.lower()
        
        # Check for LPA (Lakh Per Annum) - default to INR
        if re.search(r'\bLPA\b', value_str, re.IGNORECASE):
            return 'INR'
        
        # Check for Lakh/Crore patterns - default to INR
        if re.search(r'\b(lakh|lakhs|crore|crores)\b', value_str, re.IGNORECASE):
            return 'INR'
        
        # Check for "dollars" or "dollar" keyword
        if re.search(r'\bdollars?\b', value_str, re.IGNORECASE):
            return 'USD'
        
        # Check for currency symbols and codes (case-insensitive for codes)
        for symbol, code in self.CURRENCY_SYMBOLS.items():
            # For symbols, do exact match
            if len(symbol) <= 2 and symbol in value_str:
                return code
            # For codes/words, do case-insensitive word boundary match
            elif len(symbol) > 2 and re.search(rf'\b{re.escape(symbol)}\b', value_str, re.IGNORECASE):
                return code
        
        # Default to USD if no currency found
        return 'USD'
    
    def _extract_time_period(self, value_str):
        """Extract time period from a salary string."""
        value_str = str(value_str).lower()
        
        # Check for LPA (Lakh Per Annum) - defaults to per annum (year)
        if re.search(r'\blpa\b', value_str):
            return 'year', 1
        
        # Check for time period keywords (in order of specificity)
        # More specific patterns first to avoid false matches
        for period, multiplier in self.TIME_PERIOD_MULTIPLIERS.items():
            # Look for the period keyword with word boundaries
            if re.search(rf'\b{re.escape(period)}\b', value_str):
                return period, multiplier
        
        # Default to per year if no period found (for values like "$3000")
        return 'year', 1
    
    def _extract_amount(self, value_str):
        """Extract numeric amount from salary string."""
        value_str = str(value_str).strip()
        
        # Check for multiplier suffixes (k, K, L, Cr, etc.)
        # BE VERY SPECIFIC to avoid matching words like "monthly", "per", etc.
        multiplier = 1
        val_lower = value_str.lower()
        
        # Check for LPA pattern first (most specific)
        lpa_match = re.search(r'(\d+(?:\.\d{1,2})?)\s*lpa\b', val_lower)
        if lpa_match:
            base_amount = float(lpa_match.group(1))
            return base_amount * 100_000  # 1 LPA = 1 Lakh = 100,000
        
        # Check for these patterns in order (most specific first)
        if re.search(r'\d+\s*(cr|crore|crores)\b', val_lower):
            multiplier = 10_000_000  # Crore
            value_str = re.sub(r'(cr|crore|crores)\b', '', value_str, flags=re.IGNORECASE)
        elif re.search(r'\d+\s*(l|lakh|lakhs)\b', val_lower):
            multiplier = 100_000  # Lakh
            value_str = re.sub(r'(l|lakh|lakhs)\b', '', value_str, flags=re.IGNORECASE)
        elif re.search(r'\d+\s*(m|million)\b', val_lower) and 'month' not in val_lower:
            multiplier = 1_000_000  # Million (but not "monthly")
            value_str = re.sub(r'(m|million)\b', '', value_str, flags=re.IGNORECASE)
        elif re.search(r'\d+\s*k\b', val_lower):
            multiplier = 1_000  # Thousand
            value_str = re.sub(r'k\b', '', value_str, flags=re.IGNORECASE)
        
        # Remove all non-numeric characters except digits, dots, and commas
        # This includes ALL currency symbols: $, €, £, ₹, ¥, and mojibake like ?
        cleaned = re.sub(r'[^\d,.]', '', value_str)
        
        # Remove commas (both thousand separators and any other commas)
        cleaned = cleaned.replace(',', '')
        
        try:
            amount = float(cleaned) * multiplier
            return amount
        except:
            return None
    
    def _detect_most_common_currency(self, values):
        """Detect the most common currency in the column."""
        currency_counts = Counter()
        
        for value in values:
            if pd.notna(value):
                currency = self._extract_currency(str(value))
                currency_counts[currency] += 1
        
        if currency_counts:
            return currency_counts.most_common(1)[0][0]
        return 'INR'  # Default to INR
    
    def normalize(self, values):
        """
        Normalize salary values based on detected scenario:
        Scenario 0: Contains hourly/weekly rates -> create 2 columns (salary, salary_time_unit)
        Scenario 1: Mixed currency AND mixed time units -> create 3 columns
        Scenario 2: Mixed currency but same time unit -> convert to most common currency
        Scenario 3: Same currency but mixed time units -> convert to most common time unit
        
        Returns Series with appropriate metadata for column creation.
        """
        # Analyze the distribution
        analysis = self._analyze_salary_distribution(values)
        scenario = analysis['scenario']
        
        print(f"         [SALARY SCENARIO] Detected: {scenario}")
        print(f"         [SALARY INFO] Currencies: {analysis['currencies']}, Time periods: {analysis['time_periods']}")
        
        # Store metadata for column handler
        self.scenario = scenario
        self.analysis = analysis
        
        if scenario == 'has_hourly_weekly':
            # Scenario 0: Contains hourly/weekly rates
            # Create 2 columns: salary (amount in most common currency), salary_time_unit
            return self._normalize_has_hourly_weekly(values, analysis)
        
        elif scenario == 'mixed_both':
            # Scenario 1: Mixed currency AND mixed time units
            # Create 3 columns: salary (amount), salary_time_unit, salary_currency
            return self._normalize_mixed_both(values, analysis)
        
        elif scenario == 'mixed_currency':
            # Scenario 2: Mixed currencies but same time unit
            # Convert to most common currency: salary_{currency}_per_{time_unit}
            return self._normalize_mixed_currency(values, analysis)
        
        elif scenario == 'mixed_time':
            # Scenario 3: Same currency but mixed time units
            # Convert to most common time unit: salary_{currency}_per_{time_unit}
            return self._normalize_mixed_time(values, analysis)
        
        else:
            # Uniform: single currency and single time unit
            # Use simple normalization
            return self._normalize_uniform(values, analysis)
    
    def _normalize_has_hourly_weekly(self, values, analysis):
        """
        Scenario 0: Contains hourly/weekly rates
        Creates: salary (amount in most common currency), salary_time_unit
        """
        target_currency = analysis['most_common_currency']
        
        salary_amounts = []
        salary_time_units = []
        
        for val in values:
            if pd.isna(val):
                salary_amounts.append(np.nan)
                salary_time_units.append(np.nan)
                continue
            
            try:
                val_str = str(val).strip()
                
                # Extract components
                source_currency = self._extract_currency(val_str)
                period, _ = self._extract_time_period(val_str)
                amount = self._extract_amount(val_str)
                
                if amount is None:
                    salary_amounts.append(np.nan)
                    salary_time_units.append(np.nan)
                else:
                    # Convert currency to target currency
                    if source_currency != target_currency:
                        if source_currency in self.EXCHANGE_RATES_TO_INR:
                            amount_in_inr = amount * self.EXCHANGE_RATES_TO_INR.get(source_currency, 1.0)
                        else:
                            amount_in_inr = amount
                        
                        if target_currency in self.EXCHANGE_RATES_TO_INR:
                            amount = amount_in_inr / self.EXCHANGE_RATES_TO_INR.get(target_currency, 1.0)
                        else:
                            amount = amount_in_inr
                    
                    salary_amounts.append(int(round(amount)))
                    # Normalize time unit names
                    period_normalized = self._normalize_time_period_name(period)
                    salary_time_units.append(period_normalized)
                    
            except Exception:
                salary_amounts.append(np.nan)
                salary_time_units.append(np.nan)
        
        # Store metadata for column handler
        self.target_currency = target_currency
        self.salary_time_units = pd.Series(salary_time_units, index=values.index)
        
        return pd.Series(salary_amounts, index=values.index)
    
    def _normalize_mixed_both(self, values, analysis):
        """
        Scenario 1: Mixed currency AND mixed time units
        Creates: salary (amount only), salary_time_unit, salary_currency
        """
        salary_amounts = []
        salary_time_units = []
        salary_currencies = []
        
        for val in values:
            if pd.isna(val):
                salary_amounts.append(np.nan)
                salary_time_units.append(np.nan)
                salary_currencies.append(np.nan)
                continue
            
            try:
                val_str = str(val).strip()
                
                # Extract components
                currency = self._extract_currency(val_str)
                period, _ = self._extract_time_period(val_str)
                amount = self._extract_amount(val_str)
                
                if amount is None:
                    salary_amounts.append(np.nan)
                    salary_time_units.append(np.nan)
                    salary_currencies.append(np.nan)
                else:
                    salary_amounts.append(int(round(amount)))
                    # Normalize time unit names
                    period_normalized = self._normalize_time_period_name(period)
                    salary_time_units.append(period_normalized)
                    salary_currencies.append(currency)
                    
            except Exception:
                salary_amounts.append(np.nan)
                salary_time_units.append(np.nan)
                salary_currencies.append(np.nan)
        
        # Store additional data for column handler
        self.salary_time_units = pd.Series(salary_time_units, index=values.index)
        self.salary_currencies = pd.Series(salary_currencies, index=values.index)
        
        return pd.Series(salary_amounts, index=values.index)
    
    def _normalize_mixed_currency(self, values, analysis):
        """
        Scenario 2: Mixed currencies but same time unit
        Converts all to most common currency with same time unit
        Column name: salary_{most_common_currency}_per_{time_unit}
        """
        target_currency = analysis['most_common_currency']
        # Get the single time period (since all are same)
        target_period = list(analysis['time_periods'])[0]
        
        # Store metadata for column naming
        self.target_currency = target_currency
        self.target_period = self._normalize_time_period_name(target_period)
        
        normalized_amounts = []
        
        for val in values:
            if pd.isna(val):
                normalized_amounts.append(np.nan)
                continue
            
            try:
                val_str = str(val).strip()
                
                # Extract components
                source_currency = self._extract_currency(val_str)
                amount = self._extract_amount(val_str)
                
                if amount is None:
                    normalized_amounts.append(np.nan)
                    continue
                
                # Convert to target currency
                if source_currency != target_currency:
                    if source_currency in self.EXCHANGE_RATES_TO_INR:
                        amount_in_inr = amount * self.EXCHANGE_RATES_TO_INR.get(source_currency, 1.0)
                    else:
                        amount_in_inr = amount
                    
                    if target_currency in self.EXCHANGE_RATES_TO_INR:
                        amount = amount_in_inr / self.EXCHANGE_RATES_TO_INR.get(target_currency, 1.0)
                    else:
                        amount = amount_in_inr
                
                normalized_amounts.append(int(round(amount)))
                
            except Exception:
                normalized_amounts.append(np.nan)
        
        return pd.Series(normalized_amounts, index=values.index)
    
    def _normalize_mixed_time(self, values, analysis):
        """
        Scenario 3: Same currency but mixed time units
        Converts all to most common time unit with same currency
        Column name: salary_{currency}_per_{most_common_time_unit}
        """
        # Get the single currency (since all are same)
        target_currency = list(analysis['currencies'])[0]
        target_period = analysis['most_common_time']
        target_multiplier = self.TIME_PERIOD_MULTIPLIERS.get(target_period, 1)
        
        # Store metadata for column naming
        self.target_currency = target_currency
        self.target_period = self._normalize_time_period_name(target_period)
        
        normalized_amounts = []
        
        for val in values:
            if pd.isna(val):
                normalized_amounts.append(np.nan)
                continue
            
            try:
                val_str = str(val).strip()
                
                # Extract components
                period, period_multiplier = self._extract_time_period(val_str)
                amount = self._extract_amount(val_str)
                
                if amount is None:
                    normalized_amounts.append(np.nan)
                    continue
                
                # Convert to target time period
                # First convert to annual, then to target period
                annual_amount = amount * period_multiplier
                target_amount = annual_amount / target_multiplier
                
                normalized_amounts.append(int(round(target_amount)))
                
            except Exception:
                normalized_amounts.append(np.nan)
        
        return pd.Series(normalized_amounts, index=values.index)
    
    def _normalize_uniform(self, values, analysis):
        """
        Uniform scenario: single currency and single time unit
        Just extract the amounts, keep currency and time unit in column name
        """
        target_currency = list(analysis['currencies'])[0]
        target_period = list(analysis['time_periods'])[0]
        
        # Store metadata for column naming
        self.target_currency = target_currency
        self.target_period = self._normalize_time_period_name(target_period)
        
        normalized_amounts = []
        
        for val in values:
            if pd.isna(val):
                normalized_amounts.append(np.nan)
                continue
            
            try:
                val_str = str(val).strip()
                amount = self._extract_amount(val_str)
                
                if amount is None:
                    normalized_amounts.append(np.nan)
                else:
                    normalized_amounts.append(int(round(amount)))
                    
            except Exception:
                normalized_amounts.append(np.nan)
        
        return pd.Series(normalized_amounts, index=values.index)
    
    def _normalize_time_period_name(self, period):
        """
        Normalize time period names to standard forms for column naming.
        """
        period_lower = period.lower()
        
        if period_lower in ['hour', 'hr', 'h', 'hourly']:
            return 'hour'
        elif period_lower in ['day', 'daily']:
            return 'day'
        elif period_lower in ['week', 'weekly', 'wk']:
            return 'week'
        elif period_lower in ['month', 'monthly', 'mo']:
            return 'month'
        elif period_lower in ['year', 'yearly', 'annual', 'annually', 'annum', 'yr', 'pa', 'p.a.', 'per annum']:
            return 'year'
        else:
            return period_lower
