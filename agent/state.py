from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# Intent 값: classify_intent 노드가 결정한다.
#   "sql" → check_loaded → (run_ingestion →) run_sql → generate_response
#   "llm" → generate_response
Intent = Literal["sql", "llm"]


class AgentState(TypedDict):
    """StateGraph 전체에서 공유되는 상태."""

    # --- 입력 (항상 필수) ---
    query: str

    # --- classify_intent 가 채운다 ---
    intent: NotRequired[Intent]
    category: NotRequired[str | None]       # 사람이 읽을 수 있는 카테고리명 (e.g. "Headphones")
    csv_filename: NotRequired[str | None]   # Kaggle CSV 파일명 (e.g. "Headphones.csv")

    # --- check_loaded 가 채운다 ---
    data_loaded: NotRequired[bool]          # True → run_sql, False → run_ingestion

    # --- run_sql 이 채운다 ---
    sql_rows: NotRequired[list[dict]]

    # --- generate_response 가 채운다 ---
    response: NotRequired[str]

    # --- 어느 노드든 오류 발생 시 채운다 ---
    error: NotRequired[str | None]
