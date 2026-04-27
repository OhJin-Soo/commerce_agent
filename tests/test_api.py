"""POST /query 엔드포인트 테스트.

실제 DB·LLM·Kaggle 없이 단독 실행 가능:
- app.state.graphs 를 Mock 으로 교체해 FastAPI 레이어만 검증한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import create_app

DEFAULT_MODEL = "llama3.1:8b"
WEB_MODEL = "gemma4:26b"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_app(graph_result: dict, model: str = DEFAULT_MODEL):
    """graphs[model].ainvoke() 가 지정된 결과를 반환하는 테스트용 FastAPI 앱."""
    app = create_app()

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=graph_result)
    app.state.graphs = {model: mock_graph}
    app.state.pipeline_model = DEFAULT_MODEL
    app.state.web_search_model = WEB_MODEL

    return app


def _make_policy_app(
    pipeline_result: dict | None = None,
    web_result: dict | None = None,
):
    app = create_app()
    mock_pipeline = MagicMock()
    mock_pipeline.ainvoke = AsyncMock(return_value=pipeline_result or {"response": "pipeline", "sql_rows": []})
    mock_web = MagicMock()
    mock_web.ainvoke = AsyncMock(return_value=web_result or {"response": "web", "sql_rows": [], "react_iterations": 1})
    app.state.graphs = {
        DEFAULT_MODEL: mock_pipeline,
        WEB_MODEL: mock_web,
    }
    app.state.pipeline_model = DEFAULT_MODEL
    app.state.web_search_model = WEB_MODEL
    return app


async def _post(app, query: str, **extra) -> tuple[int, dict]:
    payload = {"query": query, **extra}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/query", json=payload)
    return resp.status_code, resp.json()


# ---------------------------------------------------------------------------
# 정상 응답 테스트
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    async def test_sql_intent_response(self):
        app = _make_app({
            "intent": "sql",
            "category": "Electronics",
            "sql_rows": [
                {"name": "Sony WH-1000XM5", "price": 249000, "rating": 4.4}
            ],
            "response": "Sony WH-1000XM5 를 추천합니다.",
        })

        status, body = await _post(app, "이어폰 30만원 이하")

        assert status == 200
        assert body["intent"] == "sql"
        assert body["category"] == "Electronics"
        assert len(body["sql_rows"]) == 1
        assert body["sql_rows"][0]["name"] == "Sony WH-1000XM5"
        assert body["response"] == "Sony WH-1000XM5 를 추천합니다."
        assert body["error"] is None

    async def test_llm_intent_response(self):
        app = _make_app({
            "intent": "llm",
            "category": "Electronics",
            "response": "이어폰 추천: Sony WH-1000XM5",
        })

        status, body = await _post(app, "이어폰 추천해줘")

        assert status == 200
        assert body["intent"] == "llm"
        assert body["response"] == "이어폰 추천: Sony WH-1000XM5"
        assert body["sql_rows"] == []   # llm 경로 → sql_rows 없음

    async def test_soft_error_still_returns_200(self):
        """노드 내부 오류는 error 필드로 전달되며 HTTP 200 을 반환한다."""
        app = _make_app({
            "intent": "sql",
            "sql_rows": [],
            "response": "",
            "error": "DB 연결 실패",
        })

        status, body = await _post(app, "노트북 조회")

        assert status == 200
        assert body["error"] == "DB 연결 실패"
        assert body["sql_rows"] == []

    async def test_model_echoed_in_response(self):
        """응답의 model 은 실제 자동 선택된 모델을 반환한다."""
        app = _make_policy_app()
        status, body = await _post(app, "노트북", model=WEB_MODEL)
        assert status == 200
        assert body["model"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 입력 유효성 검사
# ---------------------------------------------------------------------------

class TestInputValidation:
    async def test_empty_query_returns_422(self):
        app = _make_app({})
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    async def test_missing_query_field_returns_422(self):
        app = _make_app({})
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 그래프 예외 → HTTP 500
# ---------------------------------------------------------------------------

class TestGraphException:
    async def test_graph_exception_returns_500(self):
        app = create_app()
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("내부 오류"))
        app.state.graphs = {DEFAULT_MODEL: mock_graph}
        app.state.pipeline_model = DEFAULT_MODEL
        app.state.web_search_model = WEB_MODEL

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query", json={"query": "이어폰"})

        assert resp.status_code == 500
        assert "내부 오류" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 모델 라우팅
# ---------------------------------------------------------------------------

class TestModelRouting:
    async def test_missing_policy_model_returns_400(self):
        """정책이 선택한 모델 그래프가 없으면 HTTP 400 을 반환한다."""
        app = create_app()
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"response": "ok", "sql_rows": []})
        app.state.graphs = {DEFAULT_MODEL: mock_graph}
        app.state.pipeline_model = DEFAULT_MODEL
        app.state.web_search_model = WEB_MODEL

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/query", json={"query": "이어폰 후기", "model": "unknown-model:latest"}
            )
        assert resp.status_code == 400

    async def test_general_query_routes_to_llama_pipeline(self):
        """일반 DB 조회는 llama pipeline 으로 자동 라우팅된다."""
        app = _make_policy_app(
            pipeline_result={"response": "llama", "sql_rows": [], "intent": "sql"},
            web_result={"response": "gemma", "sql_rows": [], "react_iterations": 1},
        )
        mock_llama = app.state.graphs[DEFAULT_MODEL]
        mock_gemma = app.state.graphs[WEB_MODEL]

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/query", json={"query": "이어폰 5만원 이하", "model": WEB_MODEL, "use_react": True}
            )

        assert resp.status_code == 200
        assert resp.json()["response"] == "llama"
        assert resp.json()["model"] == DEFAULT_MODEL
        args, _ = mock_llama.ainvoke.await_args
        assert args == ({"query": "이어폰 5만원 이하", "use_react": False},)
        mock_gemma.ainvoke.assert_not_awaited()

    async def test_review_query_routes_to_gemma_react(self):
        """리뷰/후기 수요는 gemma ReAct 경로로 자동 라우팅된다."""
        app = _make_policy_app(
            pipeline_result={"response": "llama", "sql_rows": []},
            web_result={"response": "gemma", "sql_rows": [], "react_iterations": 3},
        )
        mock_llama = app.state.graphs[DEFAULT_MODEL]
        mock_gemma = app.state.graphs[WEB_MODEL]

        status, body = await _post(app, "Sony WH-1000XM5 리뷰", model=DEFAULT_MODEL)

        assert status == 200
        assert body["response"] == "gemma"
        assert body["model"] == WEB_MODEL
        assert body["react_steps"] == 3
        args, _ = mock_gemma.ainvoke.await_args
        assert args == ({"query": "Sony WH-1000XM5 리뷰", "use_react": True},)
        mock_llama.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# graph.ainvoke 호출 인자 검증
# ---------------------------------------------------------------------------

class TestGraphInvocation:
    async def test_query_forwarded_to_graph(self):
        app = _make_policy_app(pipeline_result={"intent": "sql", "response": "ok", "sql_rows": []})
        mock_graph = app.state.graphs[DEFAULT_MODEL]

        await _post(app, "스마트폰 최저가")

        args, kwargs = mock_graph.ainvoke.await_args
        assert args == ({"query": "스마트폰 최저가", "use_react": False},)
        assert "config" in kwargs
