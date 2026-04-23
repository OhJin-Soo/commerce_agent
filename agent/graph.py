"""StateGraph 조립 모듈.

Usage::

    from agent import build_graph, GraphDeps
    from db.session import AsyncSessionLocal
    from langchain_ollama import ChatOllama

    deps = GraphDeps(
        session_factory=AsyncSessionLocal,
        llm=ChatOllama(model="llama3.1:8b"),
        dataset_handle="asaniczka/amazon-products-dataset-2023-1-4m-products",
    )
    graph = build_graph(deps)

    # 파이프라인 경로 (기존, 기본값)
    result = await graph.ainvoke({"query": "이어폰 5만원 이하"})

    # ReAct 경로 (use_react=True)
    result = await graph.ainvoke({"query": "이어폰 5만원 이하", "use_react": True})

그래프 토폴로지::

    START
      │
      ▼ (use_react=False, 기본)           (use_react=True)
    classify_intent                      react_reason ◄──────────────────────┐
      │                                    │ (tool_calls)                    │
      ├── csv_filename 있음 → check_loaded  └──► react_act ──────────────────┘
      │     └── loaded 여부와 무관하게 run_sql → generate_response → END
      └── csv_filename 없음 ──────────────► generate_response → END
                                           │ (tool_calls 없음)
                                           └──► END
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.nodes import (
    make_check_loaded_node,
    make_classify_intent_node,
    make_generate_query_plan_node,
    make_generate_response_node,
    make_run_sql_node,
    make_web_search_node,
)
from agent.react_nodes import (
    make_react_act_node,
    make_react_reason_node,
    route_after_react_reason,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class GraphDeps:
    """build_graph() 에 필요한 외부 의존성 묶음."""

    session_factory: async_sessionmaker
    llm: BaseChatModel
    dataset_handle: str = "lokeshparab/amazon-products-dataset"
    ingest_source_site: str = "kaggle/amazon-products"
    ingest_nrows: int | None = None    # None = 전체, 정수 = 행 수 제한
    exchange_rate: float = 16.0        # INR → KRW 환율 (앱 시작 시 실시간 갱신)
    tavily_api_key: str | None = None  # Tavily 검색 API 키 (없으면 web_search 경로 비활성)


# ---------------------------------------------------------------------------
# 파이프라인 경로 라우팅 함수 (기존)
# ---------------------------------------------------------------------------

def _route_after_classify(state: AgentState) -> str:
    """인텐트와 csv_filename 에 따라 다음 노드를 결정한다.

    - web_search 인텐트 → web_search 노드 (Tavily 호출)
    - 그 외 → generate_query_plan 노드에서 structured output 시도
    """
    if state.get("intent") == "web_search":
        return "web_search"
    return "generate_query_plan"


def _route_after_query_plan(state: AgentState) -> str:
    """QueryPlan 이후 카테고리가 있으면 DB 경로, 없으면 LLM 직행."""
    return "check_loaded" if state.get("csv_filename") else "generate_response"


def _route_check_loaded(state: AgentState) -> str:
    return "run_sql"


# ---------------------------------------------------------------------------
# 최상단 경로 분기 (파이프라인 vs ReAct)
# ---------------------------------------------------------------------------

def _route_entry(state: AgentState) -> str:
    """use_react=True 이면 ReAct 경로, 아니면 기존 파이프라인 경로."""
    return "react_reason" if state.get("use_react") else "classify_intent"


# ---------------------------------------------------------------------------
# 그래프 빌더
# ---------------------------------------------------------------------------

def build_graph(deps: GraphDeps):
    """
    토폴로지::

        START
          ├── use_react=False → classify_intent (파이프라인 경로)
          │     ├── intent=web_search → web_search (Tavily) → generate_response → END
          │     ├── csv_filename 있음 → check_loaded
          │     │     └── run_sql → generate_response → END
          │     └── csv_filename 없음 ──────────────────────► generate_response → END
          └── use_react=True  → react_reason (ReAct 경로)
                ├── tool_calls 있음 → react_act → react_reason (루프)
                └── tool_calls 없음 ───────────────────────────────► END
    """
    workflow = StateGraph(AgentState)

    # ── 파이프라인 노드 ────────────────────────────────────────────────────
    workflow.add_node("classify_intent",   make_classify_intent_node())
    workflow.add_node("generate_query_plan", make_generate_query_plan_node(deps.llm))
    workflow.add_node("web_search",        make_web_search_node(deps.tavily_api_key))
    workflow.add_node("check_loaded",      make_check_loaded_node(deps.session_factory))
    workflow.add_node("run_sql",           make_run_sql_node(deps.session_factory, exchange_rate=deps.exchange_rate))
    workflow.add_node("generate_response", make_generate_response_node(deps.llm, exchange_rate=deps.exchange_rate))

    # ── ReAct 노드 ────────────────────────────────────────────────────────
    workflow.add_node("react_reason", make_react_reason_node(deps.llm))
    workflow.add_node("react_act",    make_react_act_node(
        session_factory=deps.session_factory,
        dataset_handle=deps.dataset_handle,
        ingest_nrows=deps.ingest_nrows,
        exchange_rate=deps.exchange_rate,
        tavily_api_key=deps.tavily_api_key,
    ))

    # ── 진입점: use_react 플래그로 경로 분기 ──────────────────────────────
    workflow.add_conditional_edges(
        START,
        _route_entry,
        {"classify_intent": "classify_intent", "react_reason": "react_reason"},
    )

    # ── 파이프라인 엣지 ───────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "web_search":        "web_search",
            "generate_query_plan": "generate_query_plan",
        },
    )
    workflow.add_edge("web_search", "generate_response")
    workflow.add_conditional_edges(
        "generate_query_plan",
        _route_after_query_plan,
        {"check_loaded": "check_loaded", "generate_response": "generate_response"},
    )
    workflow.add_conditional_edges(
        "check_loaded",
        _route_check_loaded,
        {"run_sql": "run_sql"},
    )
    workflow.add_edge("run_sql",           "generate_response")
    workflow.add_edge("generate_response", END)

    # ── ReAct 엣지 ────────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "react_reason",
        route_after_react_reason,
        {"react_act": "react_act", "__end__": END},
    )
    workflow.add_edge("react_act", "react_reason")  # Thought → Act → Observe → Thought …

    return workflow.compile()
