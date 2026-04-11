import re
import pandas as pd
from .base import BasePattern


class BinaryPattern(BasePattern):
    semantic_type = "binary"

    # Common binary encodings:
    # - single bit: 0 / 1
    # - bitstrings: 0101 / 11110000
    # - python-style: 0b11 / 0b1010
    regex_patterns = [
        r"^[01]$",
        r"^[01]{2,}$",
        r"^0b[01]+$",
    ]

    def detect(self, values) -> float:
        if len(values) == 0:
            return 0.0

        matched = 0
        total = 0

        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue

            value_str = str(value).strip()
            if not value_str:
                continue

            if value_str.lower() in ["null", "nan", "none"]:
                continue

            total += 1
            value_norm = value_str.lower()

            if any(re.match(p, value_norm) for p in self.regex_patterns):
                matched += 1

        if total == 0:
            return 0.0

        return matched / total

    def normalize(self, values):
        return values
