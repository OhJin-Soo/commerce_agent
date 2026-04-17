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
    result = await graph.ainvoke({"query": "이어폰 5만원 이하"})

그래프 토폴로지::

    START
      │
      ▼
    classify_intent ── "llm" ───────────────────────────► generate_response ──► END
      │
      └── "sql" ──► check_loaded ── loaded=True ────────► run_sql ──► generate_response ──► END
                        │
                        └── loaded=False ──► run_ingestion ──► run_sql ──► generate_response ──► END
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.nodes import (
    make_check_loaded_node,
    make_classify_intent_node,
    make_generate_response_node,
    make_run_ingestion_node,
    make_run_sql_node,
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


# ---------------------------------------------------------------------------
# 라우팅 함수
# ---------------------------------------------------------------------------

def _route_after_classify(state: AgentState) -> str:
    """csv_filename 이 있으면 DB 경로, 없으면 LLM 직행.

    intent(sql/llm)는 응답 스타일을 결정할 뿐 DB 조회 여부와 무관하다.
    카테고리 키워드가 없는 쿼리("커머스 사용법 알려줘" 등)만 DB를 건너뛴다.
    """
    return "check_loaded" if state.get("csv_filename") else "generate_response"


def _route_check_loaded(state: AgentState) -> str:
    return "run_sql" if state.get("data_loaded", False) else "run_ingestion"


# ---------------------------------------------------------------------------
# 그래프 빌더
# ---------------------------------------------------------------------------

def build_graph(deps: GraphDeps):
    """
    토폴로지::

        classify_intent
          ├── csv_filename 있음 → check_loaded
          │     ├── loaded=True  → run_sql → generate_response → END
          │     └── loaded=False → run_ingestion → run_sql → generate_response → END
          └── csv_filename 없음 ──────────────────► generate_response → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent",   make_classify_intent_node())
    workflow.add_node("check_loaded",      make_check_loaded_node(deps.session_factory))
    workflow.add_node("run_ingestion",     make_run_ingestion_node(
        session_factory=deps.session_factory,
        dataset_handle=deps.dataset_handle,
        ingest_nrows=deps.ingest_nrows,
    ))
    workflow.add_node("run_sql",           make_run_sql_node(deps.session_factory))
    workflow.add_node("generate_response", make_generate_response_node(deps.llm))

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"check_loaded": "check_loaded", "generate_response": "generate_response"},
    )
    workflow.add_conditional_edges(
        "check_loaded",
        _route_check_loaded,
        {"run_sql": "run_sql", "run_ingestion": "run_ingestion"},
    )
    workflow.add_edge("run_ingestion",     "run_sql")
    workflow.add_edge("run_sql",           "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
