from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# Intent 값: classify_intent 노드가 결정한다.
#   "sql"        → check_loaded → run_sql → generate_response
#   "llm"        → generate_response
#   "web_search" → web_search (Tavily) → generate_response
Intent = Literal["sql", "llm", "web_search"]


class AgentState(TypedDict):
    """StateGraph 전체에서 공유되는 상태."""

    # --- 입력 (항상 필수) ---
    query: str

    # --- 경로 선택 ---
    use_react: NotRequired[bool]            # True → ReAct 경로, False(기본) → 파이프라인 경로

    # --- classify_intent 가 채운다 (파이프라인 경로) ---
    intent: NotRequired[Intent]
    category: NotRequired[str | None]       # 사람이 읽을 수 있는 카테고리명 (e.g. "Headphones")
    csv_filename: NotRequired[str | None]   # Kaggle CSV 파일명 (e.g. "Headphones.csv")

    # --- check_loaded 가 채운다 (파이프라인 경로) ---
    data_loaded: NotRequired[bool]          # 참고용. 요청 중 ingestion 은 수행하지 않는다.

    # --- generate_query_plan / run_sql 이 채운다 (파이프라인 경로) ---
    query_plan: NotRequired[dict]           # LLM structured output 기반 검색 계획
    query_plan_error: NotRequired[str | None]
    query_plan_llm_calls: NotRequired[int]
    sql_rows: NotRequired[list[dict]]

    # --- web_search 노드가 채운다 (web_search 인텐트) ---
    web_results: NotRequired[list[dict]]    # [{title, url, content}, ...]

    # --- generate_response / react_reason 이 채운다 ---
    response: NotRequired[str]
    response_llm_calls: NotRequired[int]
    llm_input_tokens: NotRequired[int]
    llm_output_tokens: NotRequired[int]
    llm_total_tokens: NotRequired[int]

    # --- 어느 노드든 오류 발생 시 채운다 ---
    error: NotRequired[str | None]

    # --- ReAct 경로 전용 ---
    react_messages: NotRequired[list]       # LangChain BaseMessage 목록 (대화 이력)
    react_iterations: NotRequired[int]      # 무한루프 방지 카운터
