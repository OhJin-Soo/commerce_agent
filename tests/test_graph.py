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
    _source_site_from,
    make_check_loaded_node,
    make_classify_intent_node,
)
from agent.state import AgentState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_session_factory(
    rows: list[dict] | None = None,
    check_row: bool = True,
) -> MagicMock:
    rows = rows or []

    check_result = MagicMock()
    check_result.fetchone.return_value = MagicMock() if check_row else None

    sql_result = MagicMock()
    sql_result.fetchall.return_value = [MagicMock(_mapping=r) for r in rows]

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[check_result, sql_result])
    session.commit = AsyncMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = ctx
    return factory


def _make_llm(response_text: str = "테스트 응답") -> AsyncMock:
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
    )


# ---------------------------------------------------------------------------
# _source_site_from
# ---------------------------------------------------------------------------

class TestSourceSiteFrom:
    def test_derives_source_site(self):
        assert _source_site_from("Headphones.csv") == "kaggle/amazon-products/Headphones"

    def test_handles_spaces(self):
        assert _source_site_from("Washing Machines.csv") == "kaggle/amazon-products/Washing Machines"

    def test_none_returns_none(self):
        assert _source_site_from(None) is None


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    @pytest.fixture
    def node(self):
        return make_classify_intent_node()

    @pytest.mark.parametrize("query,expected_intent,expected_csv,expected_cat", [
        ("이어폰 5만원 이하",    "sql", "Headphones.csv",         "Headphones"),
        ("헤드폰 추천해줘",      "llm", "Headphones.csv",         "Headphones"),
        ("스피커 목록",          "sql", "Speakers.csv",           "Speakers"),
        ("tv 50만원 이하",       "sql", "Televisions.csv",        "Televisions"),
        ("냉장고 어때",          "llm", "Refrigerators.csv",      "Refrigerators"),
        ("노트북 100만원 이상",  "sql", "All Electronics.csv",    "All Electronics"),
        ("요가 매트",            "sql", "Yoga.csv",               "Yoga"),
        ("브랜드별 가격 조회",   "sql", None,                     None),  # 키워드 매핑 없음
    ])
    async def test_classify(self, node, query, expected_intent, expected_csv, expected_cat):
        result = await node({"query": query})
        assert result["intent"] == expected_intent
        assert result["csv_filename"] == expected_csv
        assert result["category"] == expected_cat


# ---------------------------------------------------------------------------
# check_loaded
# ---------------------------------------------------------------------------

class TestCheckLoaded:
    def _factory(self, has_row: bool) -> MagicMock:
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

    async def test_loaded_when_data_exists(self):
        node = make_check_loaded_node(self._factory(True))
        result = await node({"query": "이어폰", "csv_filename": "Headphones.csv"})
        assert result["data_loaded"] is True

    async def test_not_loaded_when_no_data(self):
        node = make_check_loaded_node(self._factory(False))
        result = await node({"query": "이어폰", "csv_filename": "Headphones.csv"})
        assert result["data_loaded"] is False

    async def test_no_csv_filename_returns_false(self):
        """csv_filename 없으면 DB 조회 없이 False."""
        node = make_check_loaded_node(self._factory(True))
        result = await node({"query": "상품 목록", "csv_filename": None})
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
        result = await node({"query": "이어폰", "csv_filename": "Headphones.csv"})
        assert result["data_loaded"] is False
        assert "connection refused" in result["error"]

    async def test_uses_source_site_not_category(self):
        """source_site = 'kaggle/amazon-products/Headphones' 로 조회하는지 확인."""
        captured: list[str] = []

        result_mock = MagicMock()
        result_mock.fetchone.return_value = MagicMock()
        session = AsyncMock()

        async def spy_execute(stmt, params=None):
            captured.append(str(stmt))
            return result_mock

        session.execute = spy_execute
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        factory = MagicMock()
        factory.return_value = ctx

        node = make_check_loaded_node(factory)
        await node({"query": "이어폰", "csv_filename": "Headphones.csv"})

        assert any("source_site" in s for s in captured)


# ---------------------------------------------------------------------------
# _build_sql
# ---------------------------------------------------------------------------

class TestBuildSql:
    def test_price_upper_bound(self):
        assert "price <= 50000" in _build_sql("이어폰 5만원 이하")

    def test_price_lower_bound(self):
        assert "price >= 1000000" in _build_sql("노트북 100만원 이상")

    def test_source_site_filter(self):
        sql = _build_sql("이어폰", source_site="kaggle/amazon-products/Headphones")
        assert "source_site = 'kaggle/amazon-products/Headphones'" in sql

    def test_price_and_source_site_combined(self):
        sql = _build_sql("이어폰 5만원 이하", source_site="kaggle/amazon-products/Headphones")
        assert "price <= 50000" in sql
        assert "source_site" in sql

    def test_no_condition(self):
        sql = _build_sql("전체 상품")
        assert "WHERE" not in sql


# ---------------------------------------------------------------------------
# 전체 그래프
# ---------------------------------------------------------------------------

class TestGraph:
    def test_graph_compiles(self):
        assert build_graph(_make_deps()) is not None

    async def test_sql_loaded_path(self):
        """이어폰 쿼리 → Headphones.csv 선택 → loaded → run_sql → generate_response."""
        product = {"id": 1, "name": "Sony WH-1000XM5", "price": 249000, "rating": 4.4}
        deps = _make_deps(rows=[product], check_row=True, llm_text="Sony를 추천합니다.")

        result = await build_graph(deps).ainvoke({"query": "이어폰 30만원 이하"})

        assert result["intent"] == "sql"
        assert result["csv_filename"] == "Headphones.csv"
        assert result["data_loaded"] is True
        assert result["sql_rows"][0]["name"] == "Sony WH-1000XM5"
        assert result["response"] == "Sony를 추천합니다."

    async def test_sql_not_loaded_triggers_ingestion(self):
        """데이터 없음 → run_ingestion → run_sql."""
        product = {"id": 2, "name": "JBL Tune 510BT", "price": 79000, "rating": 4.2}
        factory = _make_session_factory(rows=[product], check_row=False)
        session = factory.return_value.__aenter__.return_value
        check_res = MagicMock(); check_res.fetchone.return_value = None
        sql_res = MagicMock(); sql_res.fetchall.return_value = [MagicMock(_mapping=product)]
        session.execute = AsyncMock(side_effect=[check_res, sql_res])

        deps = GraphDeps(session_factory=factory, llm=_make_llm())

        with patch(
            "pipeline.kaggle_load.KaggleLoadFilter.process", new_callable=AsyncMock
        ) as mock_load:
            mock_load.return_value = []
            result = await build_graph(deps).ainvoke({"query": "이어폰 목록"})

        assert result["data_loaded"] is False
        assert "sql_rows" in result

    async def test_llm_with_category_queries_db(self):
        """llm 인텐트라도 csv_filename 있으면 DB 조회 후 generate_response."""
        product = {"id": 1, "name": "Sony WH-1000XM5", "price": 249000, "rating": 4.4}
        deps = _make_deps(rows=[product], check_row=True, llm_text="Sony WH-1000XM5 추천")

        result = await build_graph(deps).ainvoke({"query": "이어폰 추천해줘"})

        assert result["intent"] == "llm"
        assert result["csv_filename"] == "Headphones.csv"
        assert result["data_loaded"] is True
        assert len(result["sql_rows"]) == 1       # DB 조회 실행됨
        assert result["response"] == "Sony WH-1000XM5 추천"

    async def test_no_category_skips_db(self):
        """csv_filename 없으면 DB 조회 없이 generate_response 직행."""
        deps = _make_deps(llm_text="일반 응답")

        result = await build_graph(deps).ainvoke({"query": "커머스 에이전트 사용법"})

        assert result["csv_filename"] is None
        assert "data_loaded" not in result
        assert "sql_rows" not in result

    async def test_run_sql_error_propagates(self):
        factory = _make_session_factory()
        session = factory.return_value.__aenter__.return_value
        check_res = MagicMock(); check_res.fetchone.return_value = MagicMock()
        session.execute = AsyncMock(side_effect=[check_res, RuntimeError("DB 오류")])

        deps = GraphDeps(session_factory=factory, llm=_make_llm())
        result = await build_graph(deps).ainvoke({"query": "스피커 조회"})

        assert result["error"] == "DB 오류"
        assert result["sql_rows"] == []

    async def test_unknown_category_skips_db(self):
        """csv_filename=None → check_loaded 없이 generate_response 직행."""
        deps = _make_deps(llm_text="데이터 없음")

        result = await build_graph(deps).ainvoke({"query": "브랜드별 가격 조회"})

        assert result["csv_filename"] is None
        assert "data_loaded" not in result   # check_loaded 미실행
        assert "sql_rows" not in result      # run_sql 미실행


# ---------------------------------------------------------------------------
# ReAct 경로
# ---------------------------------------------------------------------------

def _make_react_session_factory(check_loaded: bool = True) -> MagicMock:
    """check_db_loaded + query_products 에 각각 응답하는 세션 팩토리."""
    check_result = MagicMock()
    check_result.fetchone.return_value = MagicMock() if check_loaded else None

    sql_result = MagicMock()
    product = {"name": "Sony WH-1000XM5", "brand": "Sony",
               "price": 15625, "rating": 4.4, "review_count": 1200}
    sql_result.fetchall.return_value = [MagicMock(_mapping=product)]

    session = AsyncMock()
    # side_effect: 첫 execute = check_loaded 확인, 두 번째 = SQL 조회
    session.execute = AsyncMock(side_effect=[check_result, sql_result])

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = ctx
    return factory


def _make_react_llm(tool_calls_sequence: list[list[dict]], final_text: str) -> AsyncMock:
    """
    tool_calls_sequence 에 따라 순서대로 도구 호출 후 final_text 로 종료하는 LLM mock.

    tool_calls_sequence 예:
      [[{"name": "search_category", "id": "c1", "args": {"keyword": "이어폰"}}],
       [{"name": "check_db_loaded", "id": "c2", "args": {"csv_filename": "Headphones.csv"}}],
       [{"name": "query_products",  "id": "c3", "args": {"csv_filename": "Headphones.csv"}}]]
    → 3번 도구 호출 후 최종 답변
    """
    responses = []
    for tool_calls in tool_calls_sequence:
        ai = MagicMock()
        ai.content = ""
        ai.tool_calls = tool_calls
        responses.append(ai)

    final = MagicMock()
    final.content = final_text
    final.tool_calls = []
    responses.append(final)

    llm = AsyncMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(side_effect=responses)
    return llm


class TestReActPath:
    async def test_react_flag_routes_to_react_reason(self):
        """use_react=True 이면 classify_intent 를 거치지 않는다."""
        llm = _make_react_llm([], "ReAct 응답")
        deps = GraphDeps(
            session_factory=_make_react_session_factory(),
            llm=llm,
        )
        result = await build_graph(deps).ainvoke(
            {"query": "이어폰 5만원 이하", "use_react": True}
        )
        # classify_intent 가 실행됐다면 intent 가 채워지지만, ReAct 경로에선 비어있음
        assert "intent" not in result
        assert result["response"] == "ReAct 응답"
        assert result.get("react_iterations", 0) == 0  # 도구 호출 없음

    async def test_pipeline_flag_stays_on_pipeline(self):
        """use_react=False(기본) 이면 기존 파이프라인 경로를 탄다."""
        product = {"id": 1, "name": "Sony WH-1000XM5", "price": 15625, "rating": 4.4}
        deps = _make_deps(rows=[product], check_row=True, llm_text="파이프라인 응답")

        result = await build_graph(deps).ainvoke({"query": "이어폰 30만원 이하"})

        assert result["intent"] == "sql"          # classify_intent 실행됨
        assert result["response"] == "파이프라인 응답"

    async def test_react_tool_call_loop(self):
        """LLM이 도구를 2번 호출한 뒤 최종 답변을 생성한다."""
        llm = _make_react_llm(
            tool_calls_sequence=[
                [{"name": "search_category", "id": "c1", "args": {"keyword": "이어폰"}}],
                [{"name": "check_db_loaded", "id": "c2",
                  "args": {"csv_filename": "Headphones.csv"}}],
            ],
            final_text="이어폰 추천: Sony WH-1000XM5",
        )
        deps = GraphDeps(
            session_factory=_make_react_session_factory(check_loaded=True),
            llm=llm,
        )
        result = await build_graph(deps).ainvoke(
            {"query": "이어폰 추천해줘", "use_react": True}
        )
        assert result["response"] == "이어폰 추천: Sony WH-1000XM5"
        assert result["react_iterations"] == 2   # react_act 가 2번 실행됨

    async def test_react_message_history_accumulated(self):
        """매 루프마다 메시지가 누적되어 LLM 컨텍스트가 이어진다."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        llm = _make_react_llm(
            tool_calls_sequence=[
                [{"name": "search_category", "id": "c1", "args": {"keyword": "이어폰"}}],
            ],
            final_text="최종 답변",
        )
        deps = GraphDeps(
            session_factory=_make_react_session_factory(),
            llm=llm,
        )
        result = await build_graph(deps).ainvoke(
            {"query": "이어폰", "use_react": True}
        )
        msgs = result["react_messages"]

        # 순서: SystemMessage, HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)
        assert len(msgs) == 5
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)
        # msgs[2]: LLM이 반환한 AIMessage(mock) — tool_calls 있음
        assert getattr(msgs[2], "tool_calls", None)
        # msgs[3]: 도구 실행 결과 ToolMessage
        assert isinstance(msgs[3], ToolMessage)
        assert msgs[3].name == "search_category"
        # msgs[4]: LLM이 반환한 AIMessage(mock) — tool_calls 없음 (최종 답변)
        assert not getattr(msgs[4], "tool_calls", None)

    async def test_react_search_category_tool_pure(self):
        """search_category 도구 — DB 없이 순수 함수로 동작한다."""
        from agent.react_nodes import _exec_search_category
        import json

        result = json.loads(_exec_search_category("이어폰"))
        assert result["csv_filename"] == "Headphones.csv"
        assert result["category"] == "Headphones"

        result_miss = json.loads(_exec_search_category("없는카테고리"))
        assert result_miss["csv_filename"] is None


# ---------------------------------------------------------------------------
# web_search 경로
# ---------------------------------------------------------------------------

class TestWebSearchPath:
    """web_search 인텐트 — Tavily 호출 후 generate_response."""

    def _make_tavily_deps(self, tavily_response: list[dict], llm_text: str = "웹 검색 응답") -> GraphDeps:
        """Tavily 를 AsyncMock 으로 교체한 deps."""
        return GraphDeps(
            session_factory=_make_session_factory(),
            llm=_make_llm(llm_text),
            tavily_api_key="test-key",
        )

    @pytest.mark.parametrize("query", [
        "Sony WH-1000XM5 리뷰",
        "이어폰 후기 알려줘",
        "노트북 사용자 평가",
        "사람들의 의견이 궁금해",
        "실사용 후기 보여줘",
        "이어폰 사용기",
    ])
    async def test_web_search_intent_detected(self, query):
        """web_search 키워드 포함 쿼리는 web_search 인텐트로 분류된다."""
        node = make_classify_intent_node()
        result = await node({"query": query})
        assert result["intent"] == "web_search", f"query={query!r} → intent={result['intent']!r}"

    async def test_web_search_path_returns_response(self):
        """web_search 경로: Tavily 결과 → generate_response."""
        tavily_results = [
            {"title": "Sony 리뷰", "url": "https://example.com", "content": "음질이 훌륭합니다."}
        ]
        with patch("tavily.AsyncTavilyClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance
            mock_instance.search = AsyncMock(return_value={"results": tavily_results})

            deps = GraphDeps(
                session_factory=_make_session_factory(),
                llm=_make_llm("Sony 리뷰 요약 응답"),
                tavily_api_key="test-key",
            )
            result = await build_graph(deps).ainvoke({"query": "Sony WH-1000XM5 리뷰"})

        assert result["intent"] == "web_search"
        assert result["web_results"] == tavily_results
        assert result["response"] == "Sony 리뷰 요약 응답"
        assert result.get("sql_rows") is None   # DB 조회 없음

    async def test_web_search_no_api_key_graceful(self):
        """TAVILY_API_KEY 없으면 web_results=[] 로 graceful degradation."""
        deps = GraphDeps(
            session_factory=_make_session_factory(),
            llm=_make_llm("키 없음 응답"),
            tavily_api_key=None,    # 키 없음
        )
        result = await build_graph(deps).ainvoke({"query": "이어폰 후기"})

        assert result["intent"] == "web_search"
        assert result["web_results"] == []
        assert result["response"] == "키 없음 응답"

    async def test_web_search_tavily_error_graceful(self):
        """Tavily 호출 실패 시 web_results=[], error 필드 설정."""
        with patch("tavily.AsyncTavilyClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance
            mock_instance.search = AsyncMock(side_effect=RuntimeError("Tavily 오류"))

            deps = GraphDeps(
                session_factory=_make_session_factory(),
                llm=_make_llm("오류 후 응답"),
                tavily_api_key="test-key",
            )
            result = await build_graph(deps).ainvoke({"query": "Sony WH-1000XM5 리뷰"})

        assert result["web_results"] == []
        assert "Tavily 오류" in result.get("error", "")

    async def test_generate_response_uses_web_results_over_sql_rows(self):
        """web_results 가 있으면 sql_rows 보다 우선해 컨텍스트로 사용한다."""
        from agent.nodes import make_generate_response_node

        captured_prompt: list[str] = []

        async def mock_invoke(messages):
            captured_prompt.extend(str(m.content) for m in messages)
            msg = MagicMock()
            msg.content = "응답"
            return msg

        llm = MagicMock()
        llm.ainvoke = mock_invoke

        node = make_generate_response_node(llm)
        await node({
            "query": "리뷰 알려줘",
            "sql_rows": [{"name": "product", "brand": "b", "price": 100, "rating": 4.0}],
            "web_results": [{"title": "웹 제목", "url": "http://a.com", "content": "웹 내용"}],
        })

        full_prompt = "\n".join(captured_prompt)
        assert "웹 검색 결과" in full_prompt
        assert "웹 내용" in full_prompt
        # sql_rows 를 컨텍스트로 사용했다면 "상품 데이터:" 섹션이 나타남
        assert "상품 데이터:" not in full_prompt

    async def test_react_search_web_tool_no_key(self):
        """search_web 도구 — API 키 없으면 빈 results 반환."""
        import json
        from agent.react_nodes import _exec_search_web

        result = json.loads(await _exec_search_web(None, "Sony WH-1000XM5 리뷰"))
        assert result["results"] == []

    async def test_react_search_web_tool_with_key(self):
        """search_web 도구 — Tavily 결과를 JSON 으로 직렬화해 반환한다."""
        import json
        from agent.react_nodes import _exec_search_web

        tavily_results = [
            {"title": "리뷰", "url": "https://x.com", "content": "좋습니다"}
        ]
        with patch("tavily.AsyncTavilyClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance
            mock_instance.search = AsyncMock(return_value={"results": tavily_results})

            result = json.loads(await _exec_search_web("test-key", "이어폰 리뷰"))

        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "리뷰"
