import re
import pandas as pd
from collections import Counter
from datetime import datetime
from .base import BasePattern


class DatetimePattern(BasePattern):
    semantic_type = "datetime"

    # ------------------------------------------------------------------
    # REGEX PATTERNS FOR DETECTION
    # ------------------------------------------------------------------
    regex_patterns = [
        # ISO 8601 and standard formats
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}$',
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}\.\d+$',
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}Z$',
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$',
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}$',
        r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}$',

        # Slash separator formats
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}/\d{1,2}/\d{2}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}/\d{1,2}/\d{2}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}/\d{1,2}/\d{2}\s+(noon|midnight)$',
        r'^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}\s*(AM|PM)$',

        # Hyphen separator formats
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}\s*(AM|PM)$',

        # Dot separator formats
        r'^\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}[ap]m$',
        r'^\d{4}\.\d{1,2}\.\d{1,2}\s+\d{1,2}\.\d{2}$',
        r'^\d{4}\.\d{1,2}\.\d{1,2}\s+\d{1,2}:\d{2}$',

        # Alternative separators (em-dash, en-dash, bullet, pipe, tilde)
        r'^\d{1,2}[—–•|~]\d{1,2}[—–•|~]\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{4}[—–•|~]\d{1,2}[—–•|~]\d{1,2}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}[—–•|~]\d{1,2}[—–•|~]\d{4}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}[—–•|~]\d{1,2}[—–•|~]\d{4}\s+\d{1,2}:\d{2}[ap]m$',
        r'^\d{1,2}[—–•|~]\d{1,2}[—–•|~]\d{2}\s+\d{1,2}:\d{2}[ap]m$',

        # Short month name formats (e.g. "13 mar 2025 2:30 PM", "22 may 2025 3:45 PM")
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s+\d{4}\s+\d{1,2}:\d{2}$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s+\d{4}\s+(noon|midnight)$',
        r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{1,2}\s*(AM|PM)$',
        r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2}\s+\d{1,2}:\d{2}$',

        # DD-Mon-YY and DD-Mon-YYYY with time
        r'^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\s+\d{1,2}:\d{2}$',
        r'^\d{4}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{1,2}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{4}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{1,2}\s+\d{1,2}:\d{2}$',

        # Mon/DD/YYYY with time
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}$',

        # Full month name formats (e.g. "july 14, 2022 6:30 AM", "December 25, 2025 12:00 AM")
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\s+\d{1,2}[ap]m$',
        r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\s+\d{1,2}:\d{2}$',
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s+(noon|midnight)$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}$',

        # Ordinal day with full/short month name and time
        r'^\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}(st|nd|rd|th)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)$',
        r'^\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}$',

        # With era suffix (AD, CE, BC, BCE)
        r'^\d{4}-\d{1,2}-\d{1,2}\s+(AD|CE|BC|BCE)\s+\d{1,2}:\d{2}:\d{2}\.\d+$',
        r'^\d{4}-\d{1,2}-\d{1,2}\s+(AD|CE|BC|BCE)\s+\d{1,2}:\d{2}:\d{2}$',
        r'^\d{4}-\d{1,2}-\d{1,2}\s+(AD|CE|BC|BCE)\s+\d{1,2}:\d{2}$',
        r'^\d{1,2}\.\d{1,2}\.\d{4}\s+(AD|CE|BC|BCE)\s+\d{1,2}:\d{2}:\d{2}$',

        # Compact numeric datetime (YYYYMMDDHHmmss)
        r'^\d{14}$',

        # Quoted datetime
        r'^"[^"]*\d{1,2}[,\s]+\d{4}\s+\d{1,2}:\d{2}[^"]*"$',
    ]

    # ------------------------------------------------------------------
    # TIME KEYWORD MAPPING (from time.py)
    # ------------------------------------------------------------------
    time_keywords = {
        'midnight':      '00:00:00',
        'noon':          '12:00:00',
        'midday':        '12:00:00',
        'afternoon':     '12:00:00',
        'morning':       '09:00:00',
        'early morning': '06:00:00',
        'evening':       '18:00:00',
        'night':         '21:00:00',
    }

    # ------------------------------------------------------------------
    # FORMAT PATTERNS (for detect_format)
    # ------------------------------------------------------------------
    format_patterns = {
        'YYYY-MM-DDTHH:MM:SS':    (r'^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$', '%Y-%m-%dT%H:%M:%S'),
        'YYYY-MM-DD HH:MM:SS':    (r'^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$', '%Y-%m-%d %H:%M:%S'),
        'DD-MM-YYYY HH:MM:SS':    (r'^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2}):(\d{2})$', '%d-%m-%Y %H:%M:%S'),
        'DD-MM-YYYY HH:MM':       (r'^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2})$', '%d-%m-%Y %H:%M'),
        'DD/MM/YYYY HH:MM:SS':    (r'^(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2}):(\d{2})$', '%d/%m/%Y %H:%M:%S'),
        'YYYY/MM/DD HH:MM:SS':    (r'^(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$', '%Y/%m/%d %H:%M:%S'),
        'DD/MM/YYYY HH:MM AM/PM': (r'^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})\s*(AM|PM)$', None),
        'DD Mon YYYY HH:MM':      (r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\s+\d{1,2}:\d{2}$', None),
        'Month DD, YYYY HH:MM':   (r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\s*(AM|PM)?$', None),
    }

    # ------------------------------------------------------------------
    # DETECT CONFIDENCE SCORE
    # ------------------------------------------------------------------
    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0
        matched = 0
        for value in values:
            value_str = str(value).strip()
            for pattern in self.regex_patterns:
                if re.match(pattern, value_str, re.IGNORECASE):
                    matched += 1
                    break
        return matched / len(values)

    # ------------------------------------------------------------------
    # DETECT FORMAT NAME FOR A SINGLE VALUE
    # ------------------------------------------------------------------
    def detect_format(self, value_str: str):
        for format_name, (pattern, _) in self.format_patterns.items():
            if re.match(pattern, value_str, re.IGNORECASE):
                return format_name
        return None

    # ------------------------------------------------------------------
    # VALIDATE TIME COMPONENT (from time.py)
    # ------------------------------------------------------------------
    def _is_valid_time_component(self, hour: int, minute: int, second: int = 0) -> bool:
        return (0 <= hour <= 23) and (0 <= minute <= 59) and (0 <= second <= 59)

    # ------------------------------------------------------------------
    # STRIP ORDINALS (from date.py)
    # ------------------------------------------------------------------
    def _strip_ordinal(self, s: str) -> str:
        return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s, flags=re.I)

    # ------------------------------------------------------------------
    # PRE-PROCESS RAW STRING BEFORE PARSING
    # ------------------------------------------------------------------
    def _preprocess(self, val_str: str) -> str:
        # Strip surrounding quotes
        val_str = val_str.strip('"').strip("'")

        # Strip ordinals (e.g. 1st → 1)
        val_str = self._strip_ordinal(val_str)

        # Compact format: YYYYMMDDHHmmss → YYYY-MM-DD HH:MM:SS
        if re.match(r'^\d{14}$', val_str):
            val_str = (
                f"{val_str[0:4]}-{val_str[4:6]}-{val_str[6:8]} "
                f"{val_str[8:10]}:{val_str[10:12]}:{val_str[12:14]}"
            )

        # Replace alternative date separators with hyphen
        val_str = re.sub(r'[—–•|~]', '-', val_str)

        # Replace time keywords (longest match first to avoid partial matches)
        for keyword, time_val in sorted(self.time_keywords.items(), key=lambda x: -len(x[0])):
            kw_pattern = re.compile(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)', re.IGNORECASE)
            if kw_pattern.search(val_str):
                val_str = kw_pattern.sub(time_val, val_str)
                break

        # Replace dot separators in date part (protect time dots like HH.MM)
        time_dot_match = re.search(r'\s\d{1,2}\.\d{2}(?:\.\d{2})?$', val_str)
        if time_dot_match:
            time_part = time_dot_match.group(0).replace('.', ':', 2).lstrip()
            val_str = val_str[:time_dot_match.start()] + ' ' + time_part
        else:
            parts = val_str.split()
            if len(parts) >= 2:
                parts[0] = parts[0].replace('.', '-')
                val_str = ' '.join(parts)

        return val_str

    # ------------------------------------------------------------------
    # PARSE PREPROCESSED STRING TO DATETIME
    # ------------------------------------------------------------------
    def _parse_datetime(self, preprocessed: str):
        try:
            dt = pd.to_datetime(
                preprocessed,
                errors='coerce',
                dayfirst=True,
                infer_datetime_format=True
            )
            return dt
        except Exception:
            return pd.NaT

    # ------------------------------------------------------------------
    # VALIDATE PARSED DATETIME
    # ------------------------------------------------------------------
    def _is_valid_datetime(self, dt) -> bool:
        if dt is pd.NaT or pd.isna(dt):
            return False
        try:
            _ = dt.strftime('%d-%m-%Y %H:%M:%S')
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # NORMALIZE VALUES → DD-MM-YYYY HH:MM:SS
    # ------------------------------------------------------------------
    def normalize(self, values):
        """
        Normalize datetime values to DD-MM-YYYY HH:MM:SS.

        Rules (from your example):
          12/02/2026 11:30 AM   → 12-02-2026 11:30:00
          13 mar 2025 2:30 PM   → 13-03-2025 14:30:00
          30/09/2021 12:00 AM   → 30-09-2021 00:00:00
          15 aug 2026 45:00     → NULL  (invalid minute 45 in hour position)
          22 may 2025 3:45 PM   → 22-05-2025 15:45:00
          july 14, 2022 6:30 AM → 14-07-2022 06:30:00
          31 feb 2026 4:16 PM   → NULL  (impossible date)
        """
        INVALID_TOKENS = {"nan", "null", "", "none", "unknown", "nat"}
        OUTPUT_FORMAT = '%d-%m-%Y %H:%M:%S'

        def convert(val):
            if pd.isna(val):
                return "NULL"

            val_str = str(val).strip()

            if val_str.lower() in INVALID_TOKENS:
                return "NULL"

            # Must match at least one detection pattern
            if not any(re.match(p, val_str, re.IGNORECASE) for p in self.regex_patterns):
                return "NULL"

            # Preprocess (normalize separators, keywords, ordinals)
            preprocessed = self._preprocess(val_str)

            # Parse to datetime
            dt = self._parse_datetime(preprocessed)

            # Reject impossible dates (e.g. 31 Feb)
            if not self._is_valid_datetime(dt):
                return "NULL"

            # Strict time validation (catches things like hour=45)
            try:
                if not self._is_valid_time_component(dt.hour, dt.minute, dt.second):
                    return "NULL"
            except Exception:
                return "NULL"

            return dt.strftime(OUTPUT_FORMAT)

        return values.apply(convert)

    # ------------------------------------------------------------------
    # PROFILE (diagnostic summary)
    # ------------------------------------------------------------------
    def profile(self, values) -> dict:
        """
        Returns a diagnostic summary dict:
          total, matched_by_regex, valid_normalized,
          null_count, confidence, format_distribution
        """
        total = len(values)
        matched = 0
        format_dist = Counter()

        normalized = self.normalize(values)
        null_count = int((normalized == "NULL").sum())
        valid = total - null_count

        for original in values:
            val_str = str(original).strip()
            for p in self.regex_patterns:
                if re.match(p, val_str, re.IGNORECASE):
                    matched += 1
                    fmt = self.detect_format(val_str)
                    if fmt:
                        format_dist[fmt] += 1
                    break

        return {
            "total": total,
            "matched_by_regex": matched,
            "valid_normalized": valid,
            "null_count": null_count,
            "confidence": matched / total if total else 0.0,
            "format_distribution": dict(format_dist),
        }