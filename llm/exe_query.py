"""
Execute DuckDB queries for LLM chart responses.
"""

import os
import time
import tempfile
from typing import Any, Dict

import duckdb

from data_preprocessing.supabase_storage import download_file, list_files, upload_json


def _coerce_value(value: Any) -> Any:
	if hasattr(value, "item"):
		try:
			return value.item()
		except Exception:
			return value
	return value


def execute_chart_query(
	user_id: str,
	session_id: str,
	message_id: str,
	chart_payload: Dict[str, Any],
) -> Dict[str, Any]:
	if not isinstance(chart_payload, dict):
		return chart_payload

	sql_query = str(chart_payload.get("sql_query", "")).strip()
	if not sql_query:
		chart_payload["status"] = chart_payload.get("status", "pending")
		return chart_payload

	start = time.perf_counter()
	try:
		output_files = list_files(f"output/{user_id}/{session_id}")
	except Exception:
		output_files = []

	cleaned_files = [f for f in output_files if f.endswith("_cleaned.csv")]
	if not cleaned_files:
		chart_payload["status"] = "failed"
		chart_payload["data"] = []
		chart_payload["execution"] = {
			"rows_returned": 0,
			"execution_time_ms": int((time.perf_counter() - start) * 1000),
		}
		return chart_payload

	try:
		with tempfile.TemporaryDirectory() as tmpdir:
			con = duckdb.connect(database=":memory:")
			for fname in cleaned_files:
				try:
					blob = download_file(f"output/{user_id}/{session_id}/{fname}")
				except Exception:
					continue

				local_path = os.path.join(tmpdir, fname)
				os.makedirs(os.path.dirname(local_path), exist_ok=True)
				with open(local_path, "wb") as f:
					f.write(blob)

				duck_path = local_path.replace("\\", "/").replace("'", "''")
				view_name = fname.replace("\"", "\"\"")
				con.execute(
					f"CREATE OR REPLACE VIEW \"{view_name}\" AS "
					f"SELECT * FROM read_csv_auto('{duck_path}')"
				)

			df = con.execute(sql_query).fetchdf()
		x_key = (chart_payload.get("encoding") or {}).get("x")
		y_key = (chart_payload.get("encoding") or {}).get("y")

		if (not x_key or not y_key) and df is not None:
			cols = list(df.columns)
			if len(cols) >= 2:
				x_key = x_key or cols[0]
				y_key = y_key or cols[1]
				chart_payload["encoding"] = {"x": x_key, "y": y_key}

		data = []
		if x_key and y_key and x_key in df.columns and y_key in df.columns:
			for _, row in df.iterrows():
				data.append({
					"x": _coerce_value(row[x_key]),
					"y": _coerce_value(row[y_key]),
				})
		else:
			data = [
				{k: _coerce_value(v) for k, v in rec.items()}
				for rec in df.to_dict(orient="records")
			]

		elapsed_ms = int((time.perf_counter() - start) * 1000)
		chart_payload["data"] = data
		chart_payload["execution"] = {
			"rows_returned": int(len(df)),
			"execution_time_ms": elapsed_ms,
		}
		chart_payload["status"] = "completed"

		upload_json(
			f"meta_data/{user_id}/{session_id}/messages/{message_id}.json",
			chart_payload,
		)
		return chart_payload
	except Exception as exc:
		elapsed_ms = int((time.perf_counter() - start) * 1000)
		chart_payload["status"] = "failed"
		chart_payload["data"] = []
		chart_payload["execution"] = {
			"rows_returned": 0,
			"execution_time_ms": elapsed_ms,
		}
		chart_payload["error"] = str(exc)
		try:
			upload_json(
				f"meta_data/{user_id}/{session_id}/messages/{message_id}.json",
				chart_payload,
			)
		except Exception:
			pass
		return chart_payload
