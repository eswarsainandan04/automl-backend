"""
LLM Orchestrator

Flow:
1) chat decision
2) dashboard chart planning (if response_type == "chart")
3) execute chart query
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends

from llm.chat import ChatRequest, _get_current_user_id, decide_chat_response
from llm.dashboard_maker import build_chart_plan
from llm.exe_query import execute_chart_query

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post("/chat")
def chat(body: ChatRequest, user_id: str = Depends(_get_current_user_id)) -> Dict[str, Any]:
    decision = decide_chat_response(
        user_id=user_id,
        session_id=body.session_id,
        user_query=body.user_query,
    )

    chart_payload = None
    if decision.get("response_type") == "chart":
        chart_payload = build_chart_plan(
            user_id=user_id,
            session_id=body.session_id,
            message_id=decision["message_id"],
            user_query=body.user_query,
            metadata=decision.get("metadata"),
            compact_metadata=decision.get("compact_metadata"),
        )
        chart_payload = execute_chart_query(
            user_id=user_id,
            session_id=body.session_id,
            message_id=decision["message_id"],
            chart_payload=chart_payload,
        )

    return {
        "message_id": decision["message_id"],
        "response_type": decision["response_type"],
        "message": decision["message"],
        "chart": chart_payload,
    }
