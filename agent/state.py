from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# Intent 값: intent_router 노드가 결정한다.
#   "sql"    → normalized_products 직접 조회
#   "ingest" → Kaggle 적재 후 SQL 조회
#   "llm"    → LLM 해석/추천 응답
Intent = Literal["sql", "ingest", "llm"]


class AgentState(TypedDict):
    """StateGraph 전체에서 공유되는 상태."""

    # --- 입력 (항상 필수) ---
    query: str

    # --- intent_router 가 채운다 ---
    intent: NotRequired[Intent]

    # --- sql_query 노드가 채운다 ---
    sql_rows: NotRequired[list[dict]]

    # --- llm_respond 노드가 채운다 ---
    response: NotRequired[str]

    # --- 어느 노드든 오류 발생 시 채운다 ---
    error: NotRequired[str | None]
