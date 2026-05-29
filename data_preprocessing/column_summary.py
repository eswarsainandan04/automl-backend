"""
Column Summary Generator

Adds per-column "stats" to *_profiling.json files based on the cleaned dataset.
This runs after structural_type detection in the preprocessing pipeline.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from datetime import date
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
	from .supabase_storage import download_file, download_json, list_files, upload_json
except ImportError:
	from supabase_storage import download_file, download_json, list_files, upload_json


def _silent_print(*_args, **_kwargs):
	return None


print = _silent_print


_TEXT_STOPWORDS = {
	"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "at",
	"by", "from", "is", "are", "was", "were", "be", "this", "that", "it", "as",
	"but", "not", "if", "they", "their", "you", "your", "we", "our", "i",
}

_DELIMITERS_RE = re.compile(r"[;,|]")


def _json_safe(value: Any) -> Any:
	if value is None:
		return None
	if isinstance(value, (np.integer, np.floating)):
		return value.item()
	if isinstance(value, np.bool_):
		return bool(value)
	if isinstance(value, (pd.Timestamp,)):
		return value.isoformat()
	if isinstance(value, (date,)):
		return value.isoformat()
	if isinstance(value, (float, int, bool, str)):
		return value
	if pd.isna(value):
		return None
	return str(value)


def _safe_float(value: Any) -> Optional[float]:
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _safe_int(value: Any) -> Optional[int]:
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series, errors="coerce")


def _coerce_datetime(series: pd.Series) -> pd.Series:
	return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def _value_counts(series: pd.Series, top_n: int = 5) -> List[Dict[str, Any]]:
	if series.empty:
		return []
	counts = series.value_counts(dropna=True).head(top_n)
	total = int(series.shape[0])
	results = []
	for value, cnt in counts.items():
		results.append(
			{
				"value": _json_safe(value),
				"count": int(cnt),
				"percentage": float(round((cnt / total) * 100, 4)) if total else 0.0,
			}
		)
	return results


def _entropy(series: pd.Series) -> float:
	if series.empty:
		return 0.0
	counts = series.value_counts(dropna=True)
	total = counts.sum()
	probs = counts / total
	entropy = -float(np.sum(probs * np.log2(probs))) if total else 0.0
	return float(round(entropy, 6))


def _detect_datetime_granularity(dt_series: pd.Series) -> str:
	if dt_series.empty:
		return "unknown"

	hours = dt_series.dt.hour
	minutes = dt_series.dt.minute
	seconds = dt_series.dt.second
	months = dt_series.dt.month
	days = dt_series.dt.day

	if hours.nunique() == 1 and minutes.nunique() == 1 and seconds.nunique() == 1:
		if int(hours.iloc[0]) == 0 and int(minutes.iloc[0]) == 0 and int(seconds.iloc[0]) == 0:
			if months.nunique() == 1 and days.nunique() == 1:
				if int(months.iloc[0]) == 1 and int(days.iloc[0]) == 1:
					return "year"
				return "month"
			return "day"
		if minutes.nunique() == 1 and seconds.nunique() == 1:
			if int(minutes.iloc[0]) == 0 and int(seconds.iloc[0]) == 0:
				return "hour"
			if int(seconds.iloc[0]) == 0:
				return "minute"
	if seconds.nunique() == 1:
		return "minute"
	return "second"


def _detect_time_granularity(dt_series: pd.Series) -> str:
	if dt_series.empty:
		return "unknown"
	seconds = dt_series.dt.second
	minutes = dt_series.dt.minute
	if seconds.nunique() == 1 and int(seconds.iloc[0]) == 0:
		if minutes.nunique() == 1 and int(minutes.iloc[0]) == 0:
			return "hour"
		return "minute"
	return "second"


def _distribution_type(skewness: Optional[float]) -> str:
	if skewness is None:
		return "unknown"
	if abs(skewness) < 0.5:
		return "approximately_normal"
	if skewness >= 1.0:
		return "right_skewed"
	if skewness <= -1.0:
		return "left_skewed"
	if skewness > 0:
		return "moderately_right_skewed"
	return "moderately_left_skewed"


def _build_numeric_stats(series: pd.Series) -> Dict[str, Any]:
	numeric = _coerce_numeric(series).dropna()
	count = int(numeric.shape[0])
	if count == 0:
		return {
			"count": 0,
			"min": None,
			"max": None,
			"range": None,
			"sum": None,
			"mean": None,
			"median": None,
			"mode": None,
			"std": None,
			"variance": None,
			"q1": None,
			"q2": None,
			"q3": None,
			"iqr": None,
			"p5": None,
			"p95": None,
			"skewness": None,
			"kurtosis": None,
			"zero_count": 0,
			"negative_count": 0,
			"outlier_count": 0,
			"distribution_type": "unknown",
		}

	min_val = float(numeric.min())
	max_val = float(numeric.max())
	q1 = float(numeric.quantile(0.25))
	q2 = float(numeric.quantile(0.50))
	q3 = float(numeric.quantile(0.75))
	iqr = q3 - q1

	mode_series = numeric.mode(dropna=True)
	mode_val = float(mode_series.iloc[0]) if not mode_series.empty else None

	std_val = float(numeric.std()) if count > 1 else None
	var_val = float(numeric.var()) if count > 1 else None
	skew_val = float(numeric.skew()) if count >= 3 else None
	kurt_val = float(numeric.kurt()) if count >= 4 else None

	p5 = float(numeric.quantile(0.05))
	p95 = float(numeric.quantile(0.95))

	outlier_count = 0
	if iqr > 0:
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		outlier_count = int(((numeric < lower) | (numeric > upper)).sum())

	return {
		"count": count,
		"min": min_val,
		"max": max_val,
		"range": max_val - min_val,
		"sum": float(numeric.sum()),
		"mean": float(numeric.mean()),
		"median": q2,
		"mode": mode_val,
		"std": std_val,
		"variance": var_val,
		"q1": q1,
		"q2": q2,
		"q3": q3,
		"iqr": iqr,
		"p5": p5,
		"p95": p95,
		"skewness": skew_val,
		"kurtosis": kurt_val,
		"zero_count": int((numeric == 0).sum()),
		"negative_count": int((numeric < 0).sum()),
		"outlier_count": outlier_count,
		"distribution_type": _distribution_type(skew_val),
	}


def _build_categorical_stats(series: pd.Series) -> Dict[str, Any]:
	non_null = series.dropna()
	count = int(non_null.shape[0])
	if count == 0:
		return {
			"mode": None,
			"mode_frequency": 0,
			"mode_percentage": 0.0,
			"top_values": [],
			"cardinality_type": "unknown",
			"multi_value_column": False,
			"entropy": 0.0,
			"dominance_ratio": 0.0,
		}

	counts = non_null.value_counts(dropna=True)
	mode_val = counts.index[0]
	mode_freq = int(counts.iloc[0])
	mode_pct = float(round((mode_freq / count) * 100, 4)) if count else 0.0

	unique_count = int(non_null.nunique(dropna=True))
	if unique_count <= 10:
		cardinality_type = "low"
	elif unique_count <= 50:
		cardinality_type = "medium"
	else:
		cardinality_type = "high"

	multi_value_ratio = 0.0
	if count > 0:
		multi_value_ratio = float((non_null.astype(str).str.contains(_DELIMITERS_RE)).mean())
	multi_value_column = multi_value_ratio >= 0.1

	dominance_ratio = float(round((mode_freq / count), 6)) if count else 0.0

	return {
		"mode": _json_safe(mode_val),
		"mode_frequency": mode_freq,
		"mode_percentage": mode_pct,
		"top_values": _value_counts(non_null, top_n=5),
		"cardinality_type": cardinality_type,
		"multi_value_column": bool(multi_value_column),
		"entropy": _entropy(non_null),
		"dominance_ratio": dominance_ratio,
	}


def _build_text_stats(series: pd.Series) -> Dict[str, Any]:
	non_null = series.dropna().astype(str)
	count = int(non_null.shape[0])
	if count == 0:
		return {
			"count": 0,
			"min_length": None,
			"max_length": None,
			"avg_length": None,
			"avg_word_count": None,
			"duplicate_percentage": 0.0,
			"language": "unknown",
			"top_keywords": [],
			"logical_text_type": "unknown",
			"recommended_search": "unknown",
			"embedding_ready": False,
		}

	lengths = non_null.str.len()
	word_counts = non_null.apply(lambda x: len(re.findall(r"\b\w+\b", x)))

	unique_count = int(non_null.nunique())
	duplicate_pct = float(round(((count - unique_count) / count) * 100, 4)) if count else 0.0

	tokens = []
	for text in non_null:
		tokens.extend(re.findall(r"[A-Za-z]{3,}", text.lower()))
	filtered = [t for t in tokens if t not in _TEXT_STOPWORDS]
	top_keywords = [t for t, _ in Counter(filtered).most_common(5)]

	avg_length = float(lengths.mean())
	avg_word_count = float(word_counts.mean())

	if avg_length > 100 or avg_word_count > 15:
		logical_text_type = "long_form"
	elif avg_length <= 30:
		logical_text_type = "short_label"
	elif avg_length <= 60:
		logical_text_type = "short_text"
	else:
		logical_text_type = "medium_text"

	recommended_search = "full_text" if logical_text_type in {"long_form", "medium_text"} else "keyword"
	embedding_ready = bool(count >= 5 and avg_length >= 20)

	language = "unknown"
	if tokens:
		english_hits = sum(1 for t in tokens if t in _TEXT_STOPWORDS)
		if english_hits >= 5:
			language = "english"

	return {
		"count": count,
		"min_length": int(lengths.min()),
		"max_length": int(lengths.max()),
		"avg_length": float(round(avg_length, 4)),
		"avg_word_count": float(round(avg_word_count, 4)),
		"duplicate_percentage": duplicate_pct,
		"language": language,
		"top_keywords": top_keywords,
		"logical_text_type": logical_text_type,
		"recommended_search": recommended_search,
		"embedding_ready": embedding_ready,
	}


def _build_date_stats(series: pd.Series) -> Dict[str, Any]:
	dt = _coerce_datetime(series).dropna()
	count = int(dt.shape[0])
	if count == 0:
		return {
			"count": 0,
			"min_date": None,
			"max_date": None,
			"date_range_days": None,
			"detected_granularity": "unknown",
			"top_years": [],
			"top_months": [],
			"missing_dates_detected": False,
			"future_dates_count": 0,
		}

	dates = dt.dt.date
	min_date = dates.min()
	max_date = dates.max()
	date_range_days = (max_date - min_date).days if min_date and max_date else None

	years = dt.dt.year
	months = dt.dt.month
	top_years = _value_counts(years, top_n=5)
	top_months = _value_counts(months, top_n=5)

	detected_granularity = _detect_datetime_granularity(dt)

	missing_dates_detected = False
	if date_range_days is not None and date_range_days >= 1 and detected_granularity == "day":
		unique_days = int(dates.nunique())
		missing_dates_detected = unique_days < (date_range_days + 1)

	future_dates_count = int((dates > date.today()).sum())

	return {
		"count": count,
		"min_date": min_date.isoformat() if min_date else None,
		"max_date": max_date.isoformat() if max_date else None,
		"date_range_days": _safe_int(date_range_days),
		"detected_granularity": detected_granularity,
		"top_years": top_years,
		"top_months": top_months,
		"missing_dates_detected": bool(missing_dates_detected),
		"future_dates_count": future_dates_count,
	}


def _build_time_stats(series: pd.Series) -> Dict[str, Any]:
	dt = _coerce_datetime(series).dropna()
	count = int(dt.shape[0])
	if count == 0:
		return {
			"count": 0,
			"min_time": None,
			"max_time": None,
			"detected_granularity": "unknown",
			"top_hours": [],
			"peak_hour": None,
			"business_hours_percentage": 0.0,
		}

	seconds = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
	min_idx = seconds.idxmin() if not seconds.empty else None
	max_idx = seconds.idxmax() if not seconds.empty else None
	min_time = dt.loc[min_idx].time() if min_idx is not None else None
	max_time = dt.loc[max_idx].time() if max_idx is not None else None

	hours = dt.dt.hour
	top_hours = _value_counts(hours, top_n=5)
	peak_hour = int(hours.value_counts().idxmax()) if count else None

	business_hours_pct = float(round(((hours.between(9, 17)).mean() * 100), 4)) if count else 0.0

	return {
		"count": count,
		"min_time": min_time.isoformat() if min_time else None,
		"max_time": max_time.isoformat() if max_time else None,
		"detected_granularity": _detect_time_granularity(dt),
		"top_hours": top_hours,
		"peak_hour": peak_hour,
		"business_hours_percentage": business_hours_pct,
	}


def _build_datetime_stats(series: pd.Series) -> Dict[str, Any]:
	dt = _coerce_datetime(series).dropna()
	count = int(dt.shape[0])
	if count == 0:
		return {
			"count": 0,
			"min_datetime": None,
			"max_datetime": None,
			"range_days": None,
			"detected_granularity": "unknown",
			"top_years": [],
			"top_months": [],
			"top_weekdays": [],
			"top_hours": [],
			"peak_hour": None,
			"business_hours_percentage": 0.0,
			"weekend_percentage": 0.0,
		}

	min_dt = dt.min()
	max_dt = dt.max()
	range_days = (max_dt - min_dt).days if min_dt is not None and max_dt is not None else None

	years = dt.dt.year
	months = dt.dt.month
	weekdays = dt.dt.weekday
	hours = dt.dt.hour

	weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
	top_weekdays_raw = _value_counts(weekdays, top_n=7)
	top_weekdays = [
		{**entry, "value": weekday_names[int(entry["value"])]}
		for entry in top_weekdays_raw
		if entry.get("value") is not None
	]

	top_hours = _value_counts(hours, top_n=5)
	peak_hour = int(hours.value_counts().idxmax()) if count else None

	business_hours_pct = float(round(((hours.between(9, 17)).mean() * 100), 4)) if count else 0.0
	weekend_pct = float(round(((weekdays >= 5).mean() * 100), 4)) if count else 0.0

	return {
		"count": count,
		"min_datetime": min_dt.isoformat() if min_dt is not None else None,
		"max_datetime": max_dt.isoformat() if max_dt is not None else None,
		"range_days": _safe_int(range_days),
		"detected_granularity": _detect_datetime_granularity(dt),
		"top_years": _value_counts(years, top_n=5),
		"top_months": _value_counts(months, top_n=5),
		"top_weekdays": top_weekdays,
		"top_hours": top_hours,
		"peak_hour": peak_hour,
		"business_hours_percentage": business_hours_pct,
		"weekend_percentage": weekend_pct,
	}


def _identifier_format(values: pd.Series) -> str:
	sample = values.dropna().astype(str).head(50).tolist()
	if not sample:
		return "unknown"

	uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
						 r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")
	if all(uuid_re.match(v) for v in sample):
		return "uuid"
	if all(v.isdigit() for v in sample):
		return "numeric"
	if all("@" in v for v in sample):
		return "email"
	if all(v.startswith("http://") or v.startswith("https://") or v.startswith("www.") for v in sample):
		return "url"
	if all(re.match(r"^[A-Za-z0-9]+$", v) for v in sample):
		return "alphanumeric"
	return "mixed"


def _is_sequential_numeric(values: pd.Series) -> bool:
	sample = values.dropna().head(50)
	if sample.empty:
		return False
	try:
		nums = [int(float(v)) for v in sample]
	except (TypeError, ValueError):
		return False
	if len(nums) < 3:
		return False
	diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
	return len(set(diffs)) == 1 and diffs[0] != 0


def _build_identifier_stats(series: pd.Series, col_meta: Dict[str, Any]) -> Dict[str, Any]:
	non_null = series.dropna()
	count = int(non_null.shape[0])
	unique_count = int(non_null.nunique(dropna=True)) if count else 0
	duplicate_count = int(count - unique_count) if count else 0
	uniqueness_ratio = float(round((unique_count / count), 6)) if count else 0.0

	lengths = non_null.astype(str).str.len() if count else pd.Series([], dtype=int)
	min_length = int(lengths.min()) if not lengths.empty else None
	max_length = int(lengths.max()) if not lengths.empty else None

	semantic_type = str(col_meta.get("semantic_type", "")).lower()

	return {
		"count": count,
		"duplicate_count": duplicate_count,
		"uniqueness_ratio": uniqueness_ratio,
		"identifier_format": _identifier_format(non_null),
		"min_length": min_length,
		"max_length": max_length,
		"is_sequential": _is_sequential_numeric(non_null),
		"collision_detected": duplicate_count > 0,
		"semantic_identifier_type": semantic_type or "unknown",
	}


def _effective_type(structural_type: str, semantic_type: str) -> str:
	stype = (structural_type or "").strip().lower()
	sem = (semantic_type or "").strip().lower()

	if sem in {"date", "time", "datetime", "timestamp"}:
		return sem
	return stype or "unknown"


def _build_stats(series: pd.Series, col_meta: Dict[str, Any]) -> Dict[str, Any]:
	structural_type = str(col_meta.get("structural_type", "")).lower()
	semantic_type = str(col_meta.get("semantic_type", "")).lower()
	effective_type = _effective_type(structural_type, semantic_type)

	if effective_type == "numeric":
		return _build_numeric_stats(series)
	if effective_type == "categorical":
		return _build_categorical_stats(series)
	if effective_type == "text":
		return _build_text_stats(series)
	if effective_type == "date":
		return _build_date_stats(series)
	if effective_type == "time":
		return _build_time_stats(series)
	if effective_type in {"datetime", "timestamp"}:
		return _build_datetime_stats(series)
	if effective_type == "identifier":
		return _build_identifier_stats(series, col_meta)

	return {}


def process_user_datasets(user_id: str, session_id: str) -> None:
	"""Add column stats into each *_profiling.json for the session."""
	meta_prefix = f"meta_data/{user_id}/{session_id}"
	output_prefix = f"output/{user_id}/{session_id}"

	try:
		meta_files = list_files(meta_prefix)
	except Exception:
		meta_files = []

	profiling_files = [f for f in meta_files if f.endswith("_profiling.json")]

	if not profiling_files:
		print(f"[column_summary] No profiling files found for {user_id}/{session_id}")
		return

	for pf in profiling_files:
		base = pf.replace("_profiling.json", "")
		profiling_path = f"{meta_prefix}/{pf}"
		cleaned_path = f"{output_prefix}/{base}_cleaned.csv"

		try:
			profiling = download_json(profiling_path)
		except Exception as exc:
			print(f"[column_summary] Failed to load profiling: {profiling_path} ({exc})")
			continue

		try:
			content = download_file(cleaned_path)
			df = pd.read_csv(BytesIO(content))
		except Exception as exc:
			print(f"[column_summary] Failed to load cleaned CSV: {cleaned_path} ({exc})")
			continue

		updated = False
		for col in profiling.get("column_wise_summary", []):
			col_name = col.get("column_name")
			if not col_name or col_name not in df.columns:
				continue
			col["stats"] = _build_stats(df[col_name], col)
			updated = True

		if updated:
			upload_json(profiling_path, profiling)
			print(f"[column_summary] Updated stats in {profiling_path}")


if __name__ == "__main__":
	import sys

	if len(sys.argv) < 3:
		print("Usage: python column_summary.py <user_id> <session_id>")
		sys.exit(1)

	process_user_datasets(sys.argv[1], sys.argv[2])
