"""StateGraph 조립 모듈.

GraphDeps 에 의존성을 묶어서 build_graph() 에 전달하면
컴파일된 LangGraph 를 반환한다.

Usage::

    from agent import build_graph, GraphDeps
    from db.session import AsyncSessionLocal
    from langchain_ollama import ChatOllama
    from pipeline.kaggle_load import KaggleDatasetConfig

    deps = GraphDeps(
        session_factory=AsyncSessionLocal,
        llm=ChatOllama(model="llama3.1:8b"),
        ingest_config=KaggleDatasetConfig(
            handle="asaniczka/amazon-products-dataset-2023-1-4m-products"
        ),
    )
    graph = build_graph(deps)
    result = await graph.ainvoke({"query": "이어폰 5만원 이하"})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.nodes import (
    make_intent_router,
    make_kaggle_ingest_node,
    make_llm_respond_node,
    make_sql_node,
)
from agent.state import AgentState
from pipeline.kaggle_load import KaggleDatasetConfig

logger = logging.getLogger(__name__)

# 기본 Kaggle 데이터셋 (Phase 1)
_DEFAULT_INGEST_CONFIG = KaggleDatasetConfig(
    handle="asaniczka/amazon-products-dataset-2023-1-4m-products",
    nrows=50_000,
)


@dataclass
class GraphDeps:
    """build_graph() 에 필요한 외부 의존성 묶음."""

    session_factory: async_sessionmaker
    llm: BaseChatModel
    ingest_config: KaggleDatasetConfig = field(default_factory=lambda: _DEFAULT_INGEST_CONFIG)
    ingest_source_site: str = "kaggle/amazon-products"


def _route_intent(state: AgentState) -> str:
    """intent_router 노드 다음 분기 결정 함수."""
    return state.get("intent", "sql")


def build_graph(deps: GraphDeps):
    """노드를 조립하고 컴파일된 StateGraph 를 반환한다.

    그래프 토폴로지::

        START
          │
          ▼
        intent_router ──── "sql" ──────► sql_query ──► END
          │                                  ▲
          ├── "ingest" ──► kaggle_ingest ────┘
          │
          └── "llm" ─────► llm_respond ──► END
    """
    workflow = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # 노드 등록 (팩토리가 의존성을 클로저로 캡처)
    # ------------------------------------------------------------------
    workflow.add_node("intent_router", make_intent_router())
    workflow.add_node("sql_query", make_sql_node(deps.session_factory))
    workflow.add_node(
        "kaggle_ingest",
        make_kaggle_ingest_node(
            session_factory=deps.session_factory,
            config=deps.ingest_config,
            source_site=deps.ingest_source_site,
        ),
    )
    workflow.add_node("llm_respond", make_llm_respond_node(deps.llm))

    # ------------------------------------------------------------------
    # 엣지 연결
    # ------------------------------------------------------------------
    workflow.set_entry_point("intent_router")

    # intent_router → 분기
    workflow.add_conditional_edges(
        "intent_router",
        _route_intent,
        {
            "sql": "sql_query",
            "ingest": "kaggle_ingest",
            "llm": "llm_respond",
        },
    )

    # ingest 완료 후 SQL 조회
    workflow.add_edge("kaggle_ingest", "sql_query")

    # 종료 노드
    workflow.add_edge("sql_query", END)
    workflow.add_edge("llm_respond", END)

    return workflow.compile()
