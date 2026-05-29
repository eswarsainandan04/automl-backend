import pandas as pd
import re
from .base import BasePattern

class TextPattern(BasePattern):
	"""Pattern for pure text strings (letters, spaces, punctuation only - NO digits)"""
	semantic_type = "text"
	
	regex_patterns = [r'^[A-Za-z\s\.,;:!?\-\'\"]+$']  # Only letters, spaces, punctuation

	def detect(self, values):
		"""Detects if values are pure text strings (no numbers, no specialized patterns)"""
		if not isinstance(values, pd.Series):
			values = pd.Series(values)
		
		# Column name awareness - boost confidence for columns named "text"
		column_name = ''
		if hasattr(values, 'name') and values.name is not None:
			column_name = str(values.name).lower()
		confidence_boost = 0.5 if 'text' in column_name else 0.0
		
		v = values.dropna()
		if len(v) == 0:
			return 0.0
		
		pure_text_count = 0
		num_like = 0
		mixed_alphanumeric = 0
		specialized_pattern_like = 0
		
		for x in v:
			val_str = str(x).strip()
			
			# Skip empty
			if not val_str:
				continue
			
			# Check if numeric
			try:
				float(val_str.replace(',', '').replace('$', '').replace('%', ''))
				num_like += 1
				continue
			except:
				pass
			
			# Check if contains digits (mixed alphanumeric - should be varchar)
			if re.search(r'\d', val_str):
				mixed_alphanumeric += 1
				continue
			
			# Check if looks like specialized patterns (coordinates, timezone, etc.)
			# Latitude/Longitude: contains degree symbol or N/S/E/W with numbers
			if re.search(r'°|[NSEW]\s*$|^\s*[NSEW]|\d+[\'\"]\s*[NSEW]', val_str):
				specialized_pattern_like += 1
			# Coordinates: lat, lon pattern or comma-separated numbers with directions
			elif re.search(r'lat|lon|geo:|,\s*-?\d+\.\d+|^\d+\s*[NSEW]\s+\d+\s*[NSEW]', val_str, re.IGNORECASE):
				specialized_pattern_like += 1
			# Timezone: UTC, GMT, or timezone abbreviations with offsets
			elif re.search(r'\bUTC\b|\bGMT\b|^[A-Z]{2,5}\s+(UTC)?\d+:?\d{0,2}$|^[A-Z]{2,5}\s*[+-]\d|^UTC\d+', val_str):
				specialized_pattern_like += 1
			# Pure text (only letters, spaces, punctuation - no digits)
			elif re.match(r'^[A-Za-z\s\.,;:!?\-\'\"]+$', val_str):
				pure_text_count += 1
		
		# If more than 50% look like specialized patterns, return low confidence
		if specialized_pattern_like / len(v) > 0.5:
			return min(1.0, 0.3 + confidence_boost)  # Low confidence - let specialized patterns win
		
		# If numeric-like, return 0
		if num_like / len(v) > 0.8:
			return 0.0
		
		# If mixed alphanumeric (should be varchar), return low confidence
		if mixed_alphanumeric / len(v) > 0.5:
			return min(1.0, 0.2 + confidence_boost)  # Let varchar pattern win
		
		# If mostly pure text (no digits), return high confidence
		# Lowered threshold from 0.7 to 0.5 to match other patterns
		base_confidence = pure_text_count / len(v)
		if base_confidence >= 0.5:
			return min(1.0, base_confidence + confidence_boost)
		
		return min(1.0, confidence_boost)  # Return boost even if base is low

	def normalize(self, values):
		# No normalization, return as-is
		return values
