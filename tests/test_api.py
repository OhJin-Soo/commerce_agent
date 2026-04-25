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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_app(graph_result: dict, model: str = DEFAULT_MODEL):
    """graphs[model].ainvoke() 가 지정된 결과를 반환하는 테스트용 FastAPI 앱."""
    app = create_app()

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=graph_result)
    app.state.graphs = {model: mock_graph}

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
        """요청의 model 이 응답의 model 필드로 그대로 반환된다."""
        app = _make_app({"intent": "sql", "response": "ok", "sql_rows": []})
        status, body = await _post(app, "노트북", model=DEFAULT_MODEL)
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
    async def test_unknown_model_returns_400(self):
        """등록되지 않은 모델을 요청하면 HTTP 400 을 반환한다."""
        app = _make_app({}, model=DEFAULT_MODEL)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/query", json={"query": "이어폰", "model": "unknown-model:latest"}
            )
        assert resp.status_code == 400

    async def test_second_model_routed_correctly(self):
        """두 번째 모델로 요청하면 해당 그래프가 호출된다."""
        app = create_app()
        mock_llama = MagicMock()
        mock_llama.ainvoke = AsyncMock(return_value={"response": "llama", "sql_rows": []})
        mock_deepseek = MagicMock()
        mock_deepseek.ainvoke = AsyncMock(return_value={"response": "deepseek", "sql_rows": []})
        app.state.graphs = {
            "llama3.1:8b": mock_llama,
            "deepseek-r1:8b": mock_deepseek,
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/query", json={"query": "이어폰", "model": "deepseek-r1:8b"}
            )

        assert resp.status_code == 200
        assert resp.json()["response"] == "deepseek"
        mock_deepseek.ainvoke.assert_awaited_once()
        mock_llama.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# graph.ainvoke 호출 인자 검증
# ---------------------------------------------------------------------------

class TestGraphInvocation:
    async def test_query_forwarded_to_graph(self):
        app = _make_app({"intent": "sql", "response": "ok", "sql_rows": []})
        mock_graph = app.state.graphs[DEFAULT_MODEL]

        await _post(app, "스마트폰 최저가")

        args, kwargs = mock_graph.ainvoke.await_args
        assert args == ({"query": "스마트폰 최저가", "use_react": False},)
        assert "config" in kwargs
