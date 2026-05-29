"""
LLM Chat API
"""

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List

import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from jose import jwt as jose_jwt
from pydantic import BaseModel

from config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER
from data_preprocessing.supabase_storage import download_json, list_files, upload_json
from jwt_handler import ALGORITHM, SECRET_KEY
from llm.prompt import CHAT_SYSTEM_PROMPT, CHAT_USER_PROMPT_TEMPLATE

router = APIRouter(prefix="/llm", tags=["LLM"])
security = HTTPBearer()
ALLOWED_CHART_TYPES = {"pie", "bar", "hist", "line", "scatter"}
_CHART_TYPE_PATTERNS = {
    "pie": re.compile(r"\bpie(\s*chart)?\b", re.IGNORECASE),
    "bar": re.compile(r"\bbar(\s*chart)?\b", re.IGNORECASE),
    "hist": re.compile(r"\bhistogram\b|\bhist\s*(plot|chart)?\b", re.IGNORECASE),
    "line": re.compile(r"\bline(\s*chart)?\b", re.IGNORECASE),
    "scatter": re.compile(r"\bscatter(\s*plot|\s*chart)?\b", re.IGNORECASE),
}
_NUMERIC_SEMANTIC_TYPES = {
    "currency", "percentage", "speed", "energy", "power", "pressure",
    "capacity", "density", "area", "distance", "weight", "volume",
    "temperature", "angle", "salary", "price", "revenue", "expense",
}


class ChatRequest(BaseModel):
    session_id: str
    user_query: str


class ChartUpdateBody(BaseModel):
    chart: Dict[str, Any]


def _email_from_token(token: str) -> str:
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _user_id_by_email_db(email: str) -> str:
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return str(row[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}")


def _get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    try:
        payload = jose_jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("user_id")
    if user_id:
        return str(user_id)

    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return _user_id_by_email_db(email)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    snippet = text[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return {}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _compact_metadata_for_prompt(metadata: Dict[str, Any]) -> Dict[str, Any]:
    max_cols = _env_int("LLM_MAX_COLUMNS", 40)
    max_samples = _env_int("LLM_MAX_SAMPLE_VALUES", 3)
    max_files = _env_int("LLM_MAX_CLEANED_FILES", 10)

    profiling = metadata.get("profiling", {}) if isinstance(metadata, dict) else {}
    constraints = metadata.get("constraints", {}) if isinstance(metadata, dict) else {}
    cleaned_files = metadata.get("cleaned_files", []) if isinstance(metadata, dict) else []
    cleaned_paths = metadata.get("cleaned_paths", []) if isinstance(metadata, dict) else []

    compact = {
        "datasets": [],
        "constraints": {},
        "cleaned_files": list(cleaned_files)[:max_files],
        "cleaned_files_total": len(cleaned_files) if isinstance(cleaned_files, list) else 0,
        "cleaned_paths": list(cleaned_paths)[:max_files],
        "cleaned_paths_total": len(cleaned_paths) if isinstance(cleaned_paths, list) else 0,
    }

    if isinstance(profiling, dict):
        for dataset_name, dataset in profiling.items():
            if not isinstance(dataset, dict):
                continue
            entry = {
                "name": dataset_name,
                "number_of_rows": dataset.get("number_of_rows"),
                "number_of_columns": dataset.get("number_of_columns"),
                "columns": [],
            }
            col_summary = dataset.get("column_wise_summary", []) or []
            for col in col_summary[:max_cols]:
                if not isinstance(col, dict):
                    continue
                entry["columns"].append({
                    "column_name": col.get("column_name"),
                    "semantic_type": col.get("semantic_type"),
                    "structural_type": col.get("structural_type"),
                    "inferred_dtype": col.get("inferred_dtype"),
                    "null_percentage": col.get("null_percentage"),
                    "unique_count": col.get("unique_count"),
                    "sample_values": (col.get("sample_values") or [])[:max_samples],
                })
            compact["datasets"].append(entry)

    if isinstance(constraints, dict):
        keys = list(constraints.keys())
        for key in keys[:max_files]:
            compact["constraints"][key] = constraints.get(key)
        if len(keys) > max_files:
            compact["constraints_truncated"] = True

    return compact


def _parse_error_message(detail: str) -> str:
    if not detail:
        return ""
    try:
        data = json.loads(detail)
        message = data.get("error", {}).get("message")
        if message:
            return str(message)
    except Exception:
        pass
    return detail


def _retry_after_ms_from_error(message: str, headers: Any = None) -> int | None:
    if headers is not None:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return int(float(retry_after) * 1000)
            except (TypeError, ValueError):
                pass
    if not message:
        return None
    match = re.search(r"try again in (\d+)(ms|s)", message, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        return value if unit == "ms" else value * 1000
    return None


def _compute_backoff_ms(attempt: int, base_ms: int = 250, cap_ms: int = 4000) -> int:
    delay = min(base_ms * (2 ** attempt), cap_ms)
    jitter = random.uniform(0, delay * 0.2)
    return int(delay + jitter)


def _groq_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured")

    if max_tokens is None:
        max_tokens = _env_int("GROQ_MAX_TOKENS", 800)

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AutoML-Backend/1.0",
        },
    )

    last_error_detail = ""
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                break
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            last_error_detail = detail
            message = _parse_error_message(detail).lower()
            is_rate_limit = exc.code == 429 or "rate limit" in message or "rate_limit" in message
            if is_rate_limit and attempt < max_retries:
                retry_ms = _retry_after_ms_from_error(message, exc.headers)
                if retry_ms is None:
                    retry_ms = _compute_backoff_ms(attempt)
                time.sleep(max(retry_ms, 50) / 1000)
                continue
            raise HTTPException(status_code=502, detail=f"Groq API error: {detail}")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Groq API request failed: {exc}")
    else:
        raise HTTPException(status_code=502, detail=f"Groq API error: {last_error_detail}")

    try:
        data = json.loads(body)
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail="Groq API returned an invalid response")


def _load_metadata(user_id: str, session_id: str) -> Dict[str, Any]:
    profiling: Dict[str, Any] = {}
    constraints: Dict[str, Any] = {}

    try:
        meta_files = list_files(f"meta_data/{user_id}/{session_id}")
    except Exception:
        meta_files = []

    for fname in meta_files:
        if fname.endswith("_profiling.json"):
            base = fname[:-len("_profiling.json")]
            try:
                profiling[base] = download_json(f"meta_data/{user_id}/{session_id}/{fname}")
            except Exception:
                continue
        elif fname.endswith("_constraints.json"):
            base = fname[:-len("_constraints.json")]
            try:
                constraints[base] = download_json(f"meta_data/{user_id}/{session_id}/{fname}")
            except Exception:
                continue

    try:
        output_files = list_files(f"output/{user_id}/{session_id}")
    except Exception:
        output_files = []

    cleaned_files = [f for f in output_files if f.endswith("_cleaned.csv")]
    cleaned_paths = [f"output/{user_id}/{session_id}/{f}" for f in cleaned_files]

    return {
        "profiling": profiling,
        "constraints": constraints,
        "cleaned_files": cleaned_files,
        "cleaned_paths": cleaned_paths,
    }


def _load_messages(user_id: str, session_id: str) -> List[Dict[str, Any]]:
    try:
        data = download_json(f"meta_data/{user_id}/{session_id}/messages.json")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass
    return []


def _next_message_id(messages: List[Dict[str, Any]]) -> str:
    max_id = 0
    for msg in messages:
        mid = str(msg.get("message_id", ""))
        match = re.match(r"msg_(\d+)", mid)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"msg_{max_id + 1:06d}"


def _append_message(user_id: str, session_id: str, entry: Dict[str, Any]) -> None:
    messages = _load_messages(user_id, session_id)
    messages.append(entry)
    upload_json(f"meta_data/{user_id}/{session_id}/messages.json", messages)


def _sanitize_chart_type(chart_payload: Dict[str, Any]) -> None:
    chart_type = str(chart_payload.get("chart_type", "")).lower().strip()
    if chart_type == "histogram":
        chart_type = "hist"
    if chart_type not in ALLOWED_CHART_TYPES:
        chart_payload["chart_type"] = "bar"
    else:
        chart_payload["chart_type"] = chart_type


def _strip_wrapping_quotes(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {"\"", "'"}:
        return trimmed[1:-1]
    return trimmed


def _sanitize_encoding(chart_payload: Dict[str, Any]) -> None:
    encoding = chart_payload.get("encoding")
    if not isinstance(encoding, dict):
        return
    if "x" in encoding:
        encoding["x"] = _strip_wrapping_quotes(encoding.get("x"))
    if "y" in encoding:
        encoding["y"] = _strip_wrapping_quotes(encoding.get("y"))


def _detect_chart_type_request(query: str) -> str | None:
    if not query:
        return None
    for chart_type, pattern in _CHART_TYPE_PATTERNS.items():
        if pattern.search(query):
            return chart_type
    return None


def _column_is_numeric(col_meta: Dict[str, Any]) -> bool:
    structural_type = str(col_meta.get("structural_type", "")).lower().strip()
    semantic_type = str(col_meta.get("semantic_type", "")).lower().strip()
    inferred_dtype = str(col_meta.get("inferred_dtype", "")).lower().strip()

    if structural_type in {"numeric", "integer", "float", "number"}:
        return True
    if inferred_dtype in {"integer", "float", "numeric_string", "number"}:
        return True
    if semantic_type in _NUMERIC_SEMANTIC_TYPES:
        return True
    return False


def _column_inventory(metadata: Dict[str, Any]) -> Dict[str, int]:
    profiling = metadata.get("profiling", {}) if isinstance(metadata, dict) else {}
    if not isinstance(profiling, dict):
        return {"numeric": 0, "categorical": 0, "date": 0, "total": 0}

    counts = {"numeric": 0, "categorical": 0, "date": 0, "total": 0}

    for dataset in profiling.values():
        if not isinstance(dataset, dict):
            continue
        for col in dataset.get("column_wise_summary", []) or []:
            if not isinstance(col, dict):
                continue
            counts["total"] += 1
            structural_type = str(col.get("structural_type", "")).lower().strip()
            semantic_type = str(col.get("semantic_type", "")).lower().strip()
            inferred_dtype = str(col.get("inferred_dtype", "")).lower().strip()

            if _column_is_numeric(col):
                counts["numeric"] += 1
                continue

            if structural_type in {"categorical", "text"}:
                counts["categorical"] += 1
                continue
            if inferred_dtype in {"text", "categorical"}:
                counts["categorical"] += 1
                continue
            if semantic_type in {"categorical", "enum"}:
                counts["categorical"] += 1
                continue

            if semantic_type in {"date", "time", "datetime", "timestamp"}:
                counts["date"] += 1
                continue
            if structural_type in {"date", "datetime", "timestamp", "time"}:
                counts["date"] += 1
                continue
            if inferred_dtype == "datetime":
                counts["date"] += 1

    return counts


def _chart_is_possible(chart_type: str, inventory: Dict[str, int]) -> tuple[bool, str]:
    numeric = inventory.get("numeric", 0)
    categorical = inventory.get("categorical", 0)
    date = inventory.get("date", 0)
    total = inventory.get("total", 0)

    if total < 2:
        return False, "Charts require at least two columns (x and y), but this dataset does not have enough columns."

    if chart_type == "hist":
        if numeric > 0:
            return True, ""
        return False, "A histogram needs numeric columns, but none were detected in this dataset."
    if chart_type == "scatter":
        if numeric >= 2:
            return True, ""
        return False, "A scatter plot needs at least two numeric columns, but this dataset does not have enough."
    if chart_type == "line":
        if numeric >= 2 or (numeric >= 1 and date >= 1):
            return True, ""
        return False, "A line chart needs a numeric series and an ordered axis (numeric or date)."
    if chart_type in {"bar", "pie"}:
        if categorical > 0:
            return True, ""
        return False, "A bar/pie chart needs categorical columns, but none were detected in this dataset."

    return True, ""


def decide_chat_response(user_id: str, session_id: str, user_query: str) -> Dict[str, Any]:
    session_id = (session_id or "").strip()
    user_query = (user_query or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not user_query:
        raise HTTPException(status_code=400, detail="user_query is required")

    metadata = _load_metadata(user_id, session_id)
    compact_metadata = _compact_metadata_for_prompt(metadata)
    metadata_json = json.dumps(compact_metadata, indent=2, ensure_ascii=True)

    requested_type = _detect_chart_type_request(user_query)
    requested_chart_possible = False
    if requested_type:
        inventory = _column_inventory(metadata)
        possible, _reason = _chart_is_possible(requested_type, inventory)
        requested_chart_possible = possible
        if not possible:
            messages = _load_messages(user_id, session_id)
            message_id = _next_message_id(messages)
            message_text = "It is not possible."
            entry = {
                "message_id": message_id,
                "user_query": user_query,
                "response_type": "text",
                "message": message_text,
                "created_at": _now_iso(),
            }
            _append_message(user_id, session_id, entry)
            return {
                "message_id": message_id,
                "response_type": "text",
                "message": message_text,
                "metadata": metadata,
                "compact_metadata": compact_metadata,
            }

    model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    chat_prompt = CHAT_USER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        metadata_json=metadata_json,
    )

    content = _groq_chat(
        [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": chat_prompt},
        ],
        model_name,
        temperature=0.2,
        max_tokens=_env_int("GROQ_MAX_TOKENS_CHAT", 800),
    )
    parsed = _extract_json(content)

    response_type = str(parsed.get("response_type", "text")).lower()
    if response_type not in {"text", "chart"}:
        response_type = "text"
    if requested_type and requested_chart_possible:
        response_type = "chart"

    message_text = parsed.get("message") or content.strip()

    messages = _load_messages(user_id, session_id)
    message_id = _next_message_id(messages)

    entry = {
        "message_id": message_id,
        "user_query": user_query,
        "response_type": response_type,
        "message": message_text,
        "created_at": _now_iso(),
    }
    _append_message(user_id, session_id, entry)

    return {
        "message_id": message_id,
        "response_type": response_type,
        "message": message_text,
        "metadata": metadata,
        "compact_metadata": compact_metadata,
    }


@router.get("/messages/{session_id}")
def list_messages(session_id: str, user_id: str = Depends(_get_current_user_id)):
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    messages = _load_messages(user_id, session_id)
    enriched = []
    for msg in messages:
        item = dict(msg)
        if item.get("response_type") == "chart":
            message_id = str(item.get("message_id", "")).strip()
            if message_id:
                try:
                    item["chart"] = download_json(
                        f"meta_data/{user_id}/{session_id}/messages/{message_id}.json"
                    )
                except Exception:
                    item["chart"] = None
        enriched.append(item)

    return {"messages": enriched}


@router.patch("/messages/{session_id}/{message_id}")
def update_chart(
    session_id: str,
    message_id: str,
    body: ChartUpdateBody,
    user_id: str = Depends(_get_current_user_id),
):
    session_id = (session_id or "").strip()
    message_id = (message_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id is required")
    if not isinstance(body.chart, dict):
        raise HTTPException(status_code=400, detail="chart payload is required")

    _sanitize_chart_type(body.chart)
    _sanitize_encoding(body.chart)

    upload_json(
        f"meta_data/{user_id}/{session_id}/messages/{message_id}.json",
        body.chart,
    )

    return {"message_id": message_id, "chart": body.chart}


