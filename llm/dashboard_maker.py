"""
Dashboard chart planner.

Generates a chart plan JSON payload for a given chat message and
stores it under meta_data/{user_id}/{session_id}/messages/{message_id}.json.
"""

import json
import os
from typing import Any, Dict, Optional

from data_preprocessing.supabase_storage import upload_json
from llm.prompt import CHART_SYSTEM_PROMPT, CHART_USER_PROMPT_TEMPLATE
from llm.chat import (
	_compact_metadata_for_prompt,
	_env_int,
	_extract_json,
	_groq_chat,
	_load_metadata,
	_sanitize_chart_type,
	_sanitize_encoding,
)


def build_chart_plan(
	user_id: str,
	session_id: str,
	message_id: str,
	user_query: str,
	metadata: Optional[Dict[str, Any]] = None,
	compact_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	if not user_id or not session_id or not message_id:
		raise ValueError("user_id, session_id, and message_id are required")

	metadata = metadata if isinstance(metadata, dict) else _load_metadata(user_id, session_id)
	compact_metadata = (
		compact_metadata
		if isinstance(compact_metadata, dict)
		else _compact_metadata_for_prompt(metadata)
	)

	metadata_json = json.dumps(compact_metadata, indent=2, ensure_ascii=True)
	cleaned_files = compact_metadata.get("cleaned_files", [])
	cleaned_paths = compact_metadata.get("cleaned_paths", [])

	model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
	chart_prompt = CHART_USER_PROMPT_TEMPLATE.format(
		user_query=user_query,
		metadata_json=metadata_json,
		cleaned_files=json.dumps(cleaned_files, indent=2, ensure_ascii=True),
		cleaned_paths=json.dumps(cleaned_paths, indent=2, ensure_ascii=True),
	)

	chart_content = _groq_chat(
		[
			{"role": "system", "content": CHART_SYSTEM_PROMPT},
			{"role": "user", "content": chart_prompt},
		],
		model_name,
		temperature=0.1,
		max_tokens=_env_int("GROQ_MAX_TOKENS_CHART", 1200),
	)

	chart_payload = _extract_json(chart_content)
	if not isinstance(chart_payload, dict):
		chart_payload = {}

	_sanitize_chart_type(chart_payload)
	_sanitize_encoding(chart_payload)
	if "status" not in chart_payload:
		chart_payload["status"] = "pending"
	if "data" not in chart_payload:
		chart_payload["data"] = []
	if "execution" not in chart_payload:
		chart_payload["execution"] = {"rows_returned": 0, "execution_time_ms": 0}

	upload_json(
		f"meta_data/{user_id}/{session_id}/messages/{message_id}.json",
		chart_payload,
	)

	return chart_payload
