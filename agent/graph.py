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
from pipeline.kaggle_load import KaggleDatasetConfig

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# 라우팅 함수
# ---------------------------------------------------------------------------

def _route_intent(state: AgentState) -> str:
    """classify_intent 다음 분기."""
    return state.get("intent", "sql")


def _route_check_loaded(state: AgentState) -> str:
    """check_loaded 다음 분기."""
    return "run_sql" if state.get("data_loaded", False) else "run_ingestion"


# ---------------------------------------------------------------------------
# 그래프 빌더
# ---------------------------------------------------------------------------

def build_graph(deps: GraphDeps):
    """노드를 조립하고 컴파일된 StateGraph 를 반환한다."""
    workflow = StateGraph(AgentState)

    # ------------------------------------------------------------------
    # 노드 등록 (팩토리가 의존성을 클로저로 캡처)
    # ------------------------------------------------------------------
    workflow.add_node("classify_intent",   make_classify_intent_node())
    workflow.add_node("check_loaded",      make_check_loaded_node(deps.session_factory))
    workflow.add_node("run_ingestion",     make_run_ingestion_node(
        session_factory=deps.session_factory,
        config=deps.ingest_config,
        source_site=deps.ingest_source_site,
    ))
    workflow.add_node("run_sql",           make_run_sql_node(deps.session_factory))
    workflow.add_node("generate_response", make_generate_response_node(deps.llm))

    # ------------------------------------------------------------------
    # 엣지 연결
    # ------------------------------------------------------------------
    workflow.set_entry_point("classify_intent")

    # classify_intent → sql 경로 / llm 경로 분기
    workflow.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "sql": "check_loaded",
            "llm": "generate_response",
        },
    )

    # check_loaded → 데이터 있으면 run_sql, 없으면 run_ingestion
    workflow.add_conditional_edges(
        "check_loaded",
        _route_check_loaded,
        {
            "run_sql":        "run_sql",
            "run_ingestion":  "run_ingestion",
        },
    )

    # run_ingestion 완료 후 SQL 조회
    workflow.add_edge("run_ingestion", "run_sql")

    # 모든 경로가 generate_response 에서 합류
    workflow.add_edge("run_sql",           "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()
