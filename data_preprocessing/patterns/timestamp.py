import re
import pandas as pd
import numpy as np
from .base import BasePattern


class TimestampPattern(BasePattern):

    semantic_type = "timestamp"

    regex_patterns = [

        # Unix timestamps
        r'^\d{10}$',
        r'^\d{13}$',
        r'^\d{16}$',
        r'^1[0-9]{9}$',
        r'^1[0-9]{12}$',

        # ISO
        r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$',
        r'^\d{8}T\d{6}(Z)?$',
        r'^\d{4}/\d{2}/\d{2}[ T]\d{2}:\d{2}:\d{2}$',
        r'^\d{4}\.\d{2}\.\d{2}[ T]\d{2}:\d{2}:\d{2}$',

        r'^\d{4}-\d{2}-\d{2}$',
        r'^\d{8}$',

        r'^\d{4}[—–•|~]\d{2}[—–•|~]\d{2}[ T]\d{2}:\d{2}:\d{2}$',
        r'^\d{2}[—–•|~]\d{2}[—–•|~]\d{4}[ T]\d{2}:\d{2}:\d{2}$',

        r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\s?(AM|PM|am|pm)$',

        r'^\d{4}-\d{2}-\d{2}\s?(AD|CE|BC|BCE)\s?\d{2}:\d{2}:\d{2}$',

        # Month name timestamps
        r'^(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)[a-z]*[\s-]\d{1,2}[,\s-]\d{4}[ T]\d{2}:\d{2}:\d{2}$',

        r'^\d{1,2}[\s-](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)[a-z]*[\s-]\d{4}[ T]\d{2}:\d{2}:\d{2}$',

        r'^(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)[a-z]*[\s-]\d{1,2}[,\s-]\d{2,4}$',

        r'^\d{1,2}[\s-](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)[a-z]*[\s-]\d{2,4}$',

        r'^\d{4}[/-](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)[/-]\d{1,2}$',

        r'^(Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)[a-z]*[\s-]\d{4}$',

        r'^\d{4}[ ](Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)$',
    ]

    # -----------------------------------------------------

    def _matches_regex(self, s):
        return any(re.match(p, s, re.I) for p in self.regex_patterns)

    # -----------------------------------------------------

    def _is_valid_timestamp(self, value):

        s = str(value).strip()

        try:

            # UNIX seconds
            if re.fullmatch(r'\d{10}', s):
                ts = int(s)
                dt = pd.to_datetime(ts, unit='s', errors='coerce')
                return not pd.isna(dt)

            # UNIX milliseconds
            if re.fullmatch(r'\d{13}', s):
                ts = int(s)
                dt = pd.to_datetime(ts, unit='ms', errors='coerce')
                return not pd.isna(dt)

            # UNIX microseconds
            if re.fullmatch(r'\d{16}', s):
                ts = int(s)
                dt = pd.to_datetime(ts, unit='us', errors='coerce')
                return not pd.isna(dt)

            # ISO / textual timestamps
            dt = pd.to_datetime(s, errors="coerce")

            return not pd.isna(dt)

        except:
            return False

    # -----------------------------------------------------

    def detect(self, values):

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
                    value_str = str(int(value))
                else:
                    value_str = str(value).strip()
            except:
                value_str = str(value).strip()

            if not self._matches_regex(value_str):
                continue

            if self._is_valid_timestamp(value_str):
                matched += 1

        if total == 0:
            return 0.0

        return matched / total

    # -----------------------------------------------------

    def normalize(self, values):

        def to_standard(val):

            if pd.isna(val):
                return ''

            if isinstance(val, float):
                val_str = str(int(val)).strip()
            else:
                val_str = str(val).strip()

            if not self._matches_regex(val_str):
                return ''

            if not self._is_valid_timestamp(val_str):
                return ''

            return val_str

        return values.apply(to_standard)