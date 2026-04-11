import re
import pandas as pd
from collections import Counter
from datetime import datetime
from .base import BasePattern


class TimePattern(BasePattern):
    semantic_type = "time"

    # ------------------------------------------------------------------
    # REGEX PATTERNS FOR DETECTION
    # ------------------------------------------------------------------
    regex_patterns = [

        # 24-hour formats
        r'^\d{1,2}:\d{2}$',
        r'^\d{1,2}:\d{2}:\d{2}$',

        # 12-hour formats with AM/PM
        r'^\d{1,2}:\d{2}\s*(AM|PM|am|pm)$',
        r'^\d{1,2}:\d{2}:\d{2}\s*(AM|PM|am|pm)$',
        r'^\d{1,2}\s*(AM|PM|am|pm)$',

        # Attached AM/PM
        r'^\d{1,2}:\d{2}(AM|PM|am|pm)$',
        r'^\d{1,2}(AM|PM|am|pm)$',

        # Compact numeric


        # Special words
        r'^(midnight|noon|midday|morning|afternoon|evening|night)$'
    ]

    # ------------------------------------------------------------------
    # KEYWORD MAPPING
    # ------------------------------------------------------------------
    time_keywords = {
        "midnight": "00:00",
        "noon": "12:00",
        "midday": "12:00",
        "morning": "09:00",
        "afternoon": "15:00",
        "evening": "18:00",
        "night": "21:00"
    }

    # ------------------------------------------------------------------
    # DETECT TIME COLUMN CONFIDENCE
    # ------------------------------------------------------------------
    def detect(self, values) -> float:

        if len(values) == 0:
            return 0.0

        column_name = getattr(values, "name", "").lower()

        # Hard stop for year-like columns
        if "year" in column_name:
            return 0.0

        matched = 0
        total = 0

        for value in values:

            total += 1

            if pd.isna(value):
                continue

            value_str = str(value).strip()

            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    if self._is_valid_time(value_str):
                        matched += 1
                    break

        if total == 0:
            return 0.0

        return matched / total

    # ------------------------------------------------------------------
    # STRICT VALIDATION
    # ------------------------------------------------------------------
    def _is_valid_time(self, value_str: str) -> bool:

        value_str = value_str.strip()

        # Handle keyword
        if value_str.lower() in self.time_keywords:
            return True

        # Normalize AM/PM spacing
        value_str = re.sub(r'(?i)(am|pm)', r' \1', value_str)
        value_str = re.sub(r'\s+', ' ', value_str).strip()

        # Try parsing 12-hour
        try:
            dt = datetime.strptime(value_str, "%I:%M %p")
            return True
        except:
            pass

        # Try parsing 12-hour with seconds
        try:
            dt = datetime.strptime(value_str, "%I:%M:%S %p")
            return True
        except:
            pass

        # Try parsing 24-hour
        try:
            dt = datetime.strptime(value_str, "%H:%M")
            return True
        except:
            pass

        # Try parsing 24-hour with seconds
        try:
            dt = datetime.strptime(value_str, "%H:%M:%S")
            return True
        except:
            pass

        # Try compact HHMM
        if re.match(r'^\d{4}$', value_str):
            try:
                datetime.strptime(value_str, "%H%M")
                return True
            except:
                return False

        # Try compact HHMMSS
        if re.match(r'^\d{6}$', value_str):
            try:
                datetime.strptime(value_str, "%H%M%S")
                return True
            except:
                return False

        return False

    # ------------------------------------------------------------------
    # FORCE RAILWAY TIME CONVERSION
    # ------------------------------------------------------------------
    def _to_railway_time(self, value_str: str):

        if pd.isna(value_str):
            return pd.NA

        value_str = str(value_str).strip()

        # Keyword handling
        if value_str.lower() in self.time_keywords:
            return self.time_keywords[value_str.lower()]

        # Normalize AM/PM spacing
        value_str = re.sub(r'(?i)(am|pm)', r' \1', value_str)
        value_str = re.sub(r'\s+', ' ', value_str).strip()

        # 1️⃣ 12-hour with seconds
        try:
            dt = datetime.strptime(value_str, "%I:%M:%S %p")
            return dt.strftime("%H:%M")
        except:
            pass

        # 2️⃣ 12-hour without seconds
        try:
            dt = datetime.strptime(value_str, "%I:%M %p")
            return dt.strftime("%H:%M")
        except:
            pass

        # 3️⃣ 24-hour with seconds
        try:
            dt = datetime.strptime(value_str, "%H:%M:%S")
            return dt.strftime("%H:%M")
        except:
            pass

        # 4️⃣ 24-hour normal
        try:
            dt = datetime.strptime(value_str, "%H:%M")
            return dt.strftime("%H:%M")
        except:
            pass

        # 5️⃣ Compact HHMM
        if re.match(r'^\d{4}$', value_str):
            try:
                dt = datetime.strptime(value_str, "%H%M")
                return dt.strftime("%H:%M")
            except:
                return pd.NA

        # 6️⃣ Compact HHMMSS
        if re.match(r'^\d{6}$', value_str):
            try:
                dt = datetime.strptime(value_str, "%H%M%S")
                return dt.strftime("%H:%M")
            except:
                return pd.NA

        return pd.NA

    # ------------------------------------------------------------------
    # NORMALIZE COLUMN TO RAILWAY TIME (HH:MM)
    # ------------------------------------------------------------------
    def normalize(self, values):

        def convert(val):

            if pd.isna(val):
                return pd.NA

            val_str = str(val).strip()

            if not self._is_valid_time(val_str):
                return pd.NA

            railway = self._to_railway_time(val_str)

            if railway is None:
                return pd.NA

            return railway

        result = values.apply(convert)

        return result

    # ------------------------------------------------------------------
    # EXTRA: STRICT CLEAN MODE (OPTIONAL FUTURE USE)
    # ------------------------------------------------------------------
    def normalize_strict(self, values):

        """
        Strict version:
        - Only allows valid times
        - Invalid values -> NULL
        - Always HH:MM format
        """

        cleaned = []

        for val in values:

            if pd.isna(val):
                cleaned.append(pd.NA)
                continue

            val_str = str(val).strip()

            if not self._is_valid_time(val_str):
                cleaned.append(pd.NA)
                continue

            converted = self._to_railway_time(val_str)

            if converted is None:
                cleaned.append(pd.NA)
            else:
                cleaned.append(converted)

        return pd.Series(cleaned, index=values.index)
