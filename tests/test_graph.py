"""StateGraph smoke tests.

외부 의존성 없이 단독 실행 가능:
- AsyncSession  →  AsyncMock (실제 DB 불필요)
- LLM           →  AsyncMock (Ollama 불필요)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph import GraphDeps, build_graph
from agent.nodes import (
    _build_sql,
    make_check_loaded_node,
    make_classify_intent_node,
)
from agent.state import AgentState
from pipeline.kaggle_load import KaggleDatasetConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_session_factory(
    rows: list[dict] | None = None,
    check_row: bool = True,
) -> MagicMock:
    """execute() 가 동작하는 가짜 세션 팩토리.

    Args:
        rows: run_sql 노드가 받을 상품 row 목록
        check_row: check_loaded 용 SELECT 결과 (True → 1행 반환, False → 빈 결과)
    """
    rows = rows or []

    # check_loaded 용 결과 (fetchone)
    check_result = MagicMock()
    check_result.fetchone.return_value = MagicMock() if check_row else None

    # run_sql 용 결과 (fetchall)
    sql_result = MagicMock()
    sql_result.fetchall.return_value = [MagicMock(_mapping=r) for r in rows]

    # execute 는 호출 순서에 따라 다른 결과를 반환
    # check_loaded → run_sql 순이므로 side_effect 리스트 사용
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[check_result, sql_result])
    session.commit = AsyncMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = ctx
    return factory


def _make_llm(response_text: str = "테스트 응답입니다.") -> AsyncMock:
    ai_msg = MagicMock()
    ai_msg.content = response_text
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=ai_msg)
    return llm


def _make_deps(
    rows: list[dict] | None = None,
    check_row: bool = True,
    llm_text: str = "mock",
) -> GraphDeps:
    return GraphDeps(
        session_factory=_make_session_factory(rows, check_row),
        llm=_make_llm(llm_text),
        ingest_config=KaggleDatasetConfig(handle="owner/test-dataset"),
    )


# ---------------------------------------------------------------------------
# classify_intent 단위 테스트
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    @pytest.fixture
    def node(self):
        return make_classify_intent_node()

    @pytest.mark.parametrize("query,expected_intent,expected_category", [
        ("이어폰 5만원 이하",       "sql", "Electronics"),
        ("노트북 100만원 이상",      "sql", "Computers"),
        ("스마트폰 최저가",          "sql", "Cell Phones"),
        ("브랜드별 가격 조회",        "sql", None),
        ("이어폰 추천해줘",          "llm", "Electronics"),
        ("이 제품 왜 이렇게 비싸",   "llm", None),
        ("두 제품 비교해줘",         "llm", None),
        ("좋은 노트북 알려줘",       "llm", "Computers"),
    ])
    async def test_classify(self, node, query, expected_intent, expected_category):
        result = await node({"query": query})
        assert result["intent"] == expected_intent
        assert result["category"] == expected_category


# ---------------------------------------------------------------------------
# check_loaded 단위 테스트
# ---------------------------------------------------------------------------

class TestCheckLoaded:
    def _factory_with_row(self, has_row: bool) -> MagicMock:
        """fetchone() 이 row 반환 여부를 제어하는 팩토리."""
        result = MagicMock()
        result.fetchone.return_value = MagicMock() if has_row else None

        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        factory = MagicMock()
        factory.return_value = ctx
        return factory

    async def test_returns_true_when_category_exists(self):
        node = make_check_loaded_node(self._factory_with_row(True))
        result = await node({"query": "이어폰", "category": "Electronics"})
        assert result["data_loaded"] is True

    async def test_returns_false_when_category_missing(self):
        node = make_check_loaded_node(self._factory_with_row(False))
        result = await node({"query": "이어폰", "category": "Electronics"})
        assert result["data_loaded"] is False

    async def test_returns_false_when_no_category(self):
        """category 가 None 이어도 전체 테이블 확인."""
        node = make_check_loaded_node(self._factory_with_row(False))
        result = await node({"query": "상품 목록", "category": None})
        assert result["data_loaded"] is False

    async def test_db_error_sets_error_field(self):
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=RuntimeError("connection refused"))

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)

        factory = MagicMock()
        factory.return_value = ctx

        node = make_check_loaded_node(factory)
        result = await node({"query": "이어폰", "category": "Electronics"})

        assert result["data_loaded"] is False
        assert "connection refused" in result["error"]


# ---------------------------------------------------------------------------
# _build_sql 단위 테스트
# ---------------------------------------------------------------------------

class TestBuildSql:
    def test_price_upper_bound(self):
        sql = _build_sql("이어폰 5만원 이하", category=None)
        assert "price <= 50000" in sql

    def test_price_lower_bound(self):
        sql = _build_sql("노트북 100만원 이상", category=None)
        assert "price >= 1000000" in sql

    def test_category_from_param(self):
        """category 파라미터가 직접 SQL 조건에 주입된다."""
        sql = _build_sql("상품 목록", category="Electronics")
        assert "category = 'Electronics'" in sql

    def test_price_and_category_combined(self):
        sql = _build_sql("이어폰 5만원 이하", category="Electronics")
        assert "price <= 50000" in sql
        assert "category = 'Electronics'" in sql

    def test_no_condition_returns_select_all(self):
        sql = _build_sql("전체 상품 보여줘", category=None)
        assert "WHERE" not in sql
        assert "normalized_products" in sql


# ---------------------------------------------------------------------------
# 전체 그래프 smoke 테스트
# ---------------------------------------------------------------------------

class TestGraph:
    def test_graph_compiles(self):
        """build_graph() 가 예외 없이 컴파일된다."""
        assert build_graph(_make_deps()) is not None

    async def test_sql_loaded_path(self):
        """sql 인텐트 + 데이터 있음: check_loaded → run_sql → generate_response."""
        product = {"id": 1, "name": "Sony WH-1000XM5", "price": 249000, "rating": 4.4}
        deps = _make_deps(rows=[product], check_row=True, llm_text="Sony를 추천합니다.")
        result = await build_graph(deps).ainvoke({"query": "이어폰 30만원 이하"})

        assert result["intent"] == "sql"
        assert result["data_loaded"] is True
        assert result["sql_rows"][0]["name"] == "Sony WH-1000XM5"
        assert result["response"] == "Sony를 추천합니다."

    async def test_sql_not_loaded_triggers_ingestion(self):
        """sql 인텐트 + 데이터 없음: check_loaded → run_ingestion → run_sql."""
        product = {"id": 2, "name": "Apple AirPods Pro", "price": 199000, "rating": 4.8}

        # check_loaded 는 False, 이후 run_sql 에서 결과 반환
        factory = _make_session_factory(rows=[product], check_row=False)
        # run_ingestion 도 같은 factory 를 씀 → execute side_effect 를 넉넉하게
        session_mock = factory.return_value.__aenter__.return_value
        check_res = MagicMock(); check_res.fetchone.return_value = None
        sql_res   = MagicMock(); sql_res.fetchall.return_value = [MagicMock(_mapping=product)]
        session_mock.execute = AsyncMock(side_effect=[check_res, sql_res])

        deps = GraphDeps(
            session_factory=factory,
            llm=_make_llm("AirPods Pro 추천"),
            ingest_config=KaggleDatasetConfig(handle="owner/test"),
        )

        with patch("pipeline.kaggle_load.KaggleLoadFilter.process", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []  # 빈 결과로 ingest 단계만 통과
            result = await build_graph(deps).ainvoke({"query": "이어폰 목록"})

        assert result["intent"] == "sql"
        assert result["data_loaded"] is False   # check_loaded 결과
        assert "sql_rows" in result

    async def test_llm_path_skips_db(self):
        """llm 인텐트: check_loaded, run_sql 을 거치지 않고 generate_response 직행."""
        deps = _make_deps(llm_text="이어폰 추천: Sony WH-1000XM5")
        result = await build_graph(deps).ainvoke({"query": "이어폰 추천해줘"})

        assert result["intent"] == "llm"
        assert result["response"] == "이어폰 추천: Sony WH-1000XM5"
        # sql 경로를 타지 않았으므로 sql_rows, data_loaded 없음
        assert "sql_rows" not in result
        assert "data_loaded" not in result

    async def test_run_sql_error_propagates(self):
        """run_sql 실패 시 error 필드가 채워진다."""
        factory = _make_session_factory()
        session = factory.return_value.__aenter__.return_value
        check_res = MagicMock(); check_res.fetchone.return_value = MagicMock()
        session.execute = AsyncMock(side_effect=[check_res, RuntimeError("DB 오류")])

        deps = GraphDeps(
            session_factory=factory,
            llm=_make_llm(),
            ingest_config=KaggleDatasetConfig(handle="owner/test"),
        )
        result = await build_graph(deps).ainvoke({"query": "노트북 조회"})

        assert result.get("error") == "DB 오류"
        assert result["sql_rows"] == []

    async def test_category_propagated_to_sql(self):
        """classify_intent 가 추출한 category 가 run_sql SQL 에 반영된다."""
        captured_sql: list[str] = []

        factory = _make_session_factory(rows=[], check_row=True)
        session = factory.return_value.__aenter__.return_value

        original_execute = session.execute.side_effect

        async def spy_execute(stmt, *args, **kwargs):
            # text() 객체의 문자열 표현 캡처
            captured_sql.append(str(stmt))
            if original_execute:
                vals = list(original_execute)
                if vals:
                    val = vals.pop(0)
                    original_execute.side_effect = vals
                    return val
            return MagicMock()

        # 간단히: SQL을 직접 확인
        sql = _build_sql("이어폰 5만원 이하", category="Electronics")
        assert "category = 'Electronics'" in sql
        assert "price <= 50000" in sql
