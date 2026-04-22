"""POST /query 엔드포인트 테스트.

실제 DB·LLM·Kaggle 없이 단독 실행 가능:
- app.state.graph 를 Mock 으로 교체해 FastAPI 레이어만 검증한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.app import create_app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_app(graph_result: dict):
    """graph.ainvoke() 가 지정된 결과를 반환하는 테스트용 FastAPI 앱."""
    app = create_app()

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=graph_result)
    app.state.graph = mock_graph

    # lifespan 이 graph 를 덮어쓰지 않도록 이미 설정된 state 를 유지
    # (lifespan 은 실제 구동 시에만 실행되므로 TestClient 에서는 별도 처리 불필요)
    return app


async def _post(app, query: str) -> tuple[int, dict]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/query", json={"query": query})
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
        app.state.graph = mock_graph

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/query", json={"query": "이어폰"})

        assert resp.status_code == 500
        assert "내부 오류" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# graph.ainvoke 호출 인자 검증
# ---------------------------------------------------------------------------

class TestGraphInvocation:
    async def test_query_forwarded_to_graph(self):
        app = _make_app({"intent": "sql", "response": "ok", "sql_rows": []})
        mock_graph = app.state.graph

        await _post(app, "스마트폰 최저가")

        mock_graph.ainvoke.assert_awaited_once_with(
            {"query": "스마트폰 최저가", "use_react": False}
        )
