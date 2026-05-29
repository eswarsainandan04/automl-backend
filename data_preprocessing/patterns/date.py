import re
import pandas as pd
from collections import Counter
from .base import BasePattern


class DatePattern(BasePattern):

    semantic_type = "date"

    regex_patterns = [

        # Standard numeric
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',
        r'^\d{1,2}[-/]\d{1,2}[-/]\d{2}$',

        r'^\d{4}\.\d{1,2}\.\d{1,2}$',
        r'^\d{1,2}\.\d{1,2}\.\d{4}$',

        r'^\d{4}[-./|•]\d{2}[-./|•]\d{2}$',
        r'^\d{2}[-./|•]\d{2}[-./|•]\d{4}$',

        # textual dates
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}$',
        r'^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}$',

        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}$',
        r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$',

        r'^\d{1,2}(st|nd|rd|th)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$',
        r'^\d{1,2}(st|nd|rd|th)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}$',

        # month-year
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}$',

        # year month
        r'^\d{4}\s+(January|February|March|April|May|June|July|August|September|October|November|December)$',
        r'^\d{4}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*$',

        # separators
        r'^(January|February|March|April|May|June|July|August|September|October|November|December)[-/_]\d{4}$',
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/_]\d{4}$',

        r'^\d{4}[-/_](January|February|March|April|May|June|July|August|September|October|November|December)$',
        r'^\d{4}[-/_](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*$',

        # YYYYMM
        r'^(19|20)\d{2}(0[1-9]|1[0-2])$',

        # MM-YYYY
        r'^(0?[1-9]|1[0-2])[-/.]\d{4}$',

        # YYYY-MM
        r'^\d{4}[-/.](0?[1-9]|1[0-2])$',

        # YYYYMMDD
        r'^\d{8}$'
    ]

    # -----------------------------------------------------

    def _matches_regex(self, s):
        return any(re.match(p, s, re.I) for p in self.regex_patterns)

    # -----------------------------------------------------

    def _is_valid_date(self, value):

        s = str(value).strip()

        # handle YYYYMM
        if re.match(r'^(19|20)\d{2}(0[1-9]|1[0-2])$', s):
            try:
                pd.to_datetime(s, format="%Y%m")
                return True
            except:
                return False

        # handle YYYYMMDD
        if re.match(r'^\d{8}$', s):
            try:
                pd.to_datetime(s, format="%Y%m%d")
                return True
            except:
                return False

        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)

        return not pd.isna(dt)

    # -----------------------------------------------------

    def detect(self, values):

        if len(values) == 0:
            return 0.0

        matched = 0

        for v in values:

            s = str(v).strip()

            if not self._matches_regex(s):
                continue

            if self._is_valid_date(s):
                matched += 1

        return matched / len(values)

    # -----------------------------------------------------

    def normalize(self, values):

        INVALID_TOKENS = {"nan", "null", "", "none", "unknown"}

        def is_missing(v):
            return pd.isna(v) or str(v).strip().lower() in INVALID_TOKENS

        def strip_ordinal(s):
            return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s, flags=re.I)

        def parse_date(s):

            s = strip_ordinal(s)

            if re.match(r'^(19|20)\d{2}(0[1-9]|1[0-2])$', s):
                return pd.to_datetime(s, format="%Y%m", errors="coerce")

            if re.match(r'^\d{8}$', s):
                return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

            return pd.to_datetime(s, errors="coerce", dayfirst=True)

        def is_textual(s):
            return bool(re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', s, re.I))

        # detect dominant pattern
        votes = []

        for v in values:

            if is_missing(v):
                continue

            s = str(v).strip()

            if not self._matches_regex(s):
                continue

            if not self._is_valid_date(s):
                continue

            votes.append("textual" if is_textual(s) else "numeric")

        dominant = Counter(votes).most_common(1)
        dominant = dominant[0][0] if dominant else "numeric"

        # conversion
        def convert(v):

            if is_missing(v):
                return "NULL"

            s = str(v).strip()

            if not self._matches_regex(s):
                return "NULL"

            if not self._is_valid_date(s):
                return "NULL"

            dt = parse_date(s)

            if pd.isna(dt):
                return "NULL"

            if dominant == "numeric":
                return dt.strftime("%d-%m-%Y")

            # textual output
            day = dt.day
            month = dt.strftime("%b").lower()
            year = dt.year

            def suffix(d):
                if 11 <= d <= 13:
                    return "th"
                return {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")

            return f"{day}{suffix(day)} {month} {year}"

        return values.apply(convert)