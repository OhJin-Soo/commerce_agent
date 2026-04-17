"""StateGraph smoke tests.

외부 의존성 없이 단독 실행 가능:
- AsyncSession  →  AsyncMock (실제 DB 불필요)
- LLM           →  AsyncMock (Ollama 불필요)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph import GraphDeps, build_graph
from agent.nodes import _build_sql, make_intent_router
from agent.state import AgentState
from pipeline.kaggle_load import KaggleDatasetConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_session_factory(rows: list[dict] | None = None) -> MagicMock:
    """execute() 가 지정된 rows 를 반환하는 가짜 세션 팩토리."""
    rows = rows or []

    mock_row_mappings = [MagicMock(_mapping=row) for row in rows]

    result = MagicMock()
    result.fetchall.return_value = mock_row_mappings

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()

    # async context manager: `async with session_factory() as session`
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = ctx
    return factory


def _make_llm(response_text: str = "테스트 응답입니다.") -> AsyncMock:
    """ainvoke() 가 고정 응답을 반환하는 가짜 LLM."""
    ai_msg = MagicMock()
    ai_msg.content = response_text
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=ai_msg)
    return llm


def _make_deps(rows: list[dict] | None = None, llm_text: str = "mock") -> GraphDeps:
    return GraphDeps(
        session_factory=_make_session_factory(rows),
        llm=_make_llm(llm_text),
        ingest_config=KaggleDatasetConfig(handle="owner/test-dataset"),
    )


# ---------------------------------------------------------------------------
# intent_router 단위 테스트
# ---------------------------------------------------------------------------

class TestIntentRouter:
    @pytest.fixture
    def router(self):
        return make_intent_router()

    @pytest.mark.parametrize("query,expected", [
        ("이어폰 5만원 이하 조회",           "sql"),
        ("노트북 10만원 이상",              "sql"),
        ("브랜드별 가격 알려줘",             "sql"),
        ("이어폰 추천해줘",                 "llm"),
        ("이 제품 왜 비싼 거야",            "llm"),
        ("두 제품 비교해줘",               "llm"),
        ("데이터 없어서 ingest 해야 해",     "ingest"),
        ("kaggle 데이터 가져와",           "ingest"),
    ])
    async def test_routing(self, router, query, expected):
        state: AgentState = {"query": query}
        result = await router(state)
        assert result["intent"] == expected


# ---------------------------------------------------------------------------
# _build_sql 단위 테스트
# ---------------------------------------------------------------------------

class TestBuildSql:
    def test_price_upper_bound(self):
        sql = _build_sql("이어폰 5만원 이하")
        assert "price <= 50000" in sql

    def test_price_lower_bound(self):
        sql = _build_sql("노트북 100만원 이상")
        assert "price >= 1000000" in sql

    def test_no_condition_returns_select_all(self):
        sql = _build_sql("전체 상품 보여줘")
        assert "WHERE" not in sql
        assert "normalized_products" in sql

    def test_category_filter(self):
        sql = _build_sql("이어폰 목록")
        assert "Electronics" in sql


# ---------------------------------------------------------------------------
# 전체 그래프 smoke 테스트
# ---------------------------------------------------------------------------

class TestGraph:
    def test_graph_compiles(self):
        """build_graph() 가 예외 없이 컴파일된다."""
        deps = _make_deps()
        graph = build_graph(deps)
        assert graph is not None

    async def test_sql_intent_flow(self):
        """sql 인텐트: sql_query 노드가 실행되고 sql_rows 가 채워진다."""
        product_row = {"id": 1, "name": "Sony WH-1000XM5", "price": 249000, "rating": 4.4}
        deps = _make_deps(rows=[product_row])
        graph = build_graph(deps)

        result = await graph.ainvoke({"query": "이어폰 30만원 이하"})

        assert result["intent"] == "sql"
        assert len(result["sql_rows"]) == 1
        assert result["sql_rows"][0]["name"] == "Sony WH-1000XM5"

    async def test_llm_intent_flow(self):
        """llm 인텐트: llm_respond 노드가 실행되고 response 가 채워진다."""
        deps = _make_deps(llm_text="Sony WH-1000XM5를 추천합니다.")
        graph = build_graph(deps)

        result = await graph.ainvoke({"query": "이어폰 추천해줘"})

        assert result["intent"] == "llm"
        assert result["response"] == "Sony WH-1000XM5를 추천합니다."

    async def test_ingest_then_sql_flow(self):
        """ingest 인텐트: kaggle_ingest → sql_query 순으로 실행된다."""
        product_row = {"id": 2, "name": "Apple AirPods Pro", "price": 199000, "rating": 4.8}
        deps = _make_deps(rows=[product_row])

        graph = build_graph(deps)

        with patch("kagglehub.dataset_download", return_value="/tmp/fake"):
            with patch("pipeline.kaggle_load.KaggleLoadFilter.process", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = []  # 빈 결과로 ingest 단계만 통과
                result = await graph.ainvoke({"query": "kaggle 데이터 가져와"})

        assert result["intent"] == "ingest"
        # ingest 후 sql_query 도 실행됨 → sql_rows 존재
        assert "sql_rows" in result

    async def test_sql_error_sets_error_field(self):
        """sql_query 실패 시 error 필드가 채워진다."""
        session_factory = _make_session_factory()
        # execute 가 예외를 던지도록 override
        ctx = session_factory.return_value
        ctx.__aenter__.return_value.execute = AsyncMock(side_effect=RuntimeError("DB 연결 실패"))

        deps = GraphDeps(
            session_factory=session_factory,
            llm=_make_llm(),
            ingest_config=KaggleDatasetConfig(handle="owner/test"),
        )
        graph = build_graph(deps)

        result = await graph.ainvoke({"query": "노트북 가격 조회"})

        assert result.get("error") is not None
        assert "DB 연결 실패" in result["error"]
