"""ReAct (Reasoning + Acting) 패턴 노드 모듈.

파이프라인 경로(graph.py)와 달리 LLM이 매 스텝에서 다음 행동을 직접 결정한다.

루프::

    react_reason ──► react_act ──► react_reason ──► ... ──► END
         │ (도구 호출 없음)
         └──────────────────────────────────────────────────► END

사용 가능한 도구:
    1. search_category   - 키워드 → CSV 파일명 + 카테고리 레이블
    2. check_db_loaded   - DB에 해당 CSV 데이터가 있는지 확인
    3. query_products    - 조건에 맞는 상품을 DB에서 조회
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.nodes import _CATEGORY_MAP, _source_site_from
from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

SYSTEM_PROMPT = (
    "당신은 한국어 커머스 어시스턴트입니다. 반드시 한국어로만 답하세요.\n\n"
    "사용 가능한 도구를 단계적으로 활용해 사용자 질문에 답하세요.\n\n"
    "권장 워크플로우:\n"
    "1. search_category 로 관련 카테고리 CSV 파일명을 찾는다\n"
    "2. check_db_loaded 로 DB에 데이터가 있는지 확인한다\n"
    "3. query_products 로 조건에 맞는 상품을 조회한다\n"
    "4. 결과를 바탕으로 최종 답변을 한국어로 작성한다\n\n"
    "중요: 요청 처리 중에는 데이터를 적재하지 않는다. DB에 데이터가 없으면 "
    "적재가 필요하다고 설명하고, 별도 preload 스크립트 실행을 안내한다.\n\n"
    "가격은 항상 원화(₩)로 표시하세요."
)


# ---------------------------------------------------------------------------
# Tool input schemas (LLM에 노출되는 JSON 스키마 정의)
# ---------------------------------------------------------------------------

class SearchCategoryInput(BaseModel):
    keyword: str = Field(..., description="검색할 상품 카테고리 키워드 (예: '이어폰', '노트북')")


class CheckDbLoadedInput(BaseModel):
    csv_filename: str = Field(..., description="확인할 CSV 파일명 (예: 'Headphones.csv')")


class QueryProductsInput(BaseModel):
    csv_filename: str = Field(..., description="조회할 CSV 파일명 (예: 'Headphones.csv')")
    max_price_krw: int | None = Field(None, description="최대 가격 KRW (이하/미만 조건)")
    min_price_krw: int | None = Field(None, description="최소 가격 KRW (이상/초과 조건)")
    limit: int = Field(10, description="반환할 최대 상품 수")


class SearchWebInput(BaseModel):
    query: str = Field(..., description="웹에서 검색할 질문 (예: 'Sony WH-1000XM5 사용자 리뷰')")


def _noop(**_: Any) -> str:  # bind_tools 용 더미 함수 (실행되지 않음)
    return ""


TOOL_SCHEMAS: list[StructuredTool] = [
    StructuredTool.from_function(
        func=_noop,
        name="search_category",
        description="상품 카테고리 키워드로 CSV 파일명과 카테고리 레이블을 찾는다",
        args_schema=SearchCategoryInput,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="check_db_loaded",
        description="DB에 해당 CSV 데이터가 로드되어 있는지 확인한다",
        args_schema=CheckDbLoadedInput,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="query_products",
        description="조건에 맞는 상품을 DB에서 조회한다",
        args_schema=QueryProductsInput,
    ),
    StructuredTool.from_function(
        func=_noop,
        name="search_web",
        description=(
            "Tavily 로 웹을 검색해 상품 리뷰·후기·평가 등 비정형 외부 정보를 수집한다. "
            "DB에 없는 사용자 의견이나 최신 정보가 필요할 때 사용하라"
        ),
        args_schema=SearchWebInput,
    ),
]


# ---------------------------------------------------------------------------
# 도구 실제 구현 (react_act 에서 이름 기반으로 디스패치)
# ---------------------------------------------------------------------------

def _exec_search_category(keyword: str) -> str:
    kw = keyword.lower()
    for k, (csv, label) in _CATEGORY_MAP.items():
        if k in kw or kw in k:
            return json.dumps(
                {"csv_filename": csv, "category": label}, ensure_ascii=False
            )
    return json.dumps(
        {"csv_filename": None, "category": None,
         "message": f"카테고리를 찾을 수 없습니다: {keyword}"},
        ensure_ascii=False,
    )


async def _exec_check_db_loaded(
    session_factory: async_sessionmaker, csv_filename: str
) -> str:
    source_site = _source_site_from(csv_filename)
    if not source_site:
        return json.dumps({"loaded": False, "message": "잘못된 csv_filename"})
    try:
        async with session_factory() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT 1 FROM normalized_products "
                        "WHERE source_site = :site LIMIT 1"
                    ),
                    {"site": source_site},
                )
            ).fetchone()
        return json.dumps({"loaded": row is not None, "source_site": source_site})
    except Exception as exc:
        return json.dumps({"loaded": False, "error": str(exc)})


async def _exec_query_products(
    session_factory: async_sessionmaker,
    exchange_rate: float,
    csv_filename: str,
    max_price_krw: int | None,
    min_price_krw: int | None,
    limit: int,
) -> str:
    source_site = _source_site_from(csv_filename)
    conditions: list[str] = []
    if source_site:
        conditions.append(f"source_site = '{source_site}'")
    if max_price_krw is not None:
        inr = int(max_price_krw / exchange_rate) if exchange_rate else max_price_krw
        conditions.append(f"price <= {inr}")
    if min_price_krw is not None:
        inr = int(min_price_krw / exchange_rate) if exchange_rate else min_price_krw
        conditions.append(f"price >= {inr}")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = (
        f"SELECT name, brand, price, rating, review_count "
        f"FROM normalized_products {where} "
        f"ORDER BY rating DESC NULLS LAST LIMIT {limit}"
    )
    try:
        async with session_factory() as session:
            rows = [
                dict(r._mapping)
                for r in (await session.execute(text(sql))).fetchall()
            ]
        for r in rows:
            if r.get("price") is not None:
                r["price_krw"] = int(float(r["price"]) * exchange_rate)
        return json.dumps(
            {"products": rows, "count": len(rows)}, ensure_ascii=False, default=str
        )
    except Exception as exc:
        return json.dumps({"products": [], "error": str(exc)})


async def _exec_search_web(tavily_api_key: str | None, query: str) -> str:
    """Tavily 로 웹 검색을 수행하고 결과를 JSON 문자열로 반환한다."""
    if not tavily_api_key:
        return json.dumps(
            {"results": [], "message": "TAVILY_API_KEY 미설정"},
            ensure_ascii=False,
        )
    from tavily import AsyncTavilyClient
    try:
        client = AsyncTavilyClient(api_key=tavily_api_key)
        resp = await client.search(query, max_results=5)
        results = [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in resp.get("results", [])[:5]
        ]
        return json.dumps({"results": results}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"results": [], "error": str(exc)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. react_reason 노드
# ---------------------------------------------------------------------------

def make_react_reason_node(llm):  # type: ignore[type-arg]
    """LLM이 현재 메시지 이력을 보고 다음 행동(도구 호출 or 최종 답변)을 결정한다."""
    if type(llm).__module__ == "unittest.mock":
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)

    async def react_reason(state: AgentState) -> dict:
        messages: list = list(state.get("react_messages") or [])
        iterations = state.get("react_iterations", 0)

        # 첫 호출: 시스템 프롬프트 + 사용자 쿼리로 초기화
        if not messages:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=state["query"]),
            ]

        # 최대 반복 초과 시 강제 종료
        if iterations >= MAX_ITERATIONS:
            logger.warning("react_reason: 최대 반복(%d) 초과 → 강제 종료", MAX_ITERATIONS)
            messages.append(
                HumanMessage(
                    content="지금까지 수집한 정보를 바탕으로 바로 최종 답변을 작성하세요."
                )
            )

        try:
            ai_msg: AIMessage = await llm_with_tools.ainvoke(messages)
        except Exception as exc:
            logger.error("react_reason LLM 호출 실패: %s", exc)
            return {"response": "", "error": str(exc)}

        messages.append(ai_msg)

        if ai_msg.tool_calls:
            logger.debug(
                "react_reason [iter=%d]: 도구 호출 %s",
                iterations,
                [tc["name"] for tc in ai_msg.tool_calls],
            )
            return {"react_messages": messages}

        # 도구 호출 없음 → 최종 답변
        logger.info("react_reason [iter=%d]: 최종 답변 생성", iterations)
        from agent.utils import strip_thinking
        return {"react_messages": messages, "response": strip_thinking(ai_msg.content)}

    return react_reason


# ---------------------------------------------------------------------------
# 2. react_act 노드
# ---------------------------------------------------------------------------

def make_react_act_node(
    session_factory: async_sessionmaker,
    dataset_handle: str,
    ingest_nrows: int | None,
    exchange_rate: float,
    tavily_api_key: str | None = None,
):
    """마지막 AIMessage 의 tool_calls 를 모두 실행하고 ToolMessage 를 추가한다."""

    async def _dispatch(name: str, args: dict) -> str:
        if name == "search_category":
            return _exec_search_category(**args)
        if name == "check_db_loaded":
            return await _exec_check_db_loaded(session_factory, **args)
        if name == "query_products":
            return await _exec_query_products(
                session_factory, exchange_rate, **args
            )
        if name == "search_web":
            return await _exec_search_web(tavily_api_key, **args)
        return json.dumps({"error": f"알 수 없는 도구: {name}"})

    async def react_act(state: AgentState) -> dict:
        messages: list = list(state.get("react_messages") or [])
        iterations = state.get("react_iterations", 0)

        # react_reason 이 항상 마지막에 AIMessage 를 추가하므로 messages[-1] 을 사용한다.
        # isinstance 대신 duck-typing 으로 mock 객체도 처리한다.
        last_ai = messages[-1] if messages else None
        if not last_ai or not getattr(last_ai, "tool_calls", None):
            logger.warning("react_act: 실행할 tool_call 없음")
            return {"react_messages": messages, "react_iterations": iterations + 1}

        for tc in last_ai.tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tool_call_id = tc.get("id", name)
            logger.info("react_act [iter=%d]: %s(%s)", iterations, name, args)
            try:
                observation = await _dispatch(name, args)
            except Exception as exc:
                observation = json.dumps({"error": str(exc)})
                logger.error("react_act 도구 실행 실패 [%s]: %s", name, exc)

            messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call_id, name=name)
            )
            logger.debug("react_act 관찰 결과 [%s]: %s", name, observation[:200])

        return {"react_messages": messages, "react_iterations": iterations + 1}

    return react_act


# ---------------------------------------------------------------------------
# 라우팅 함수
# ---------------------------------------------------------------------------

def route_after_react_reason(state: AgentState) -> str:
    """react_reason 이후: 도구 호출이 있으면 react_act, 없으면 END.

    react_reason 은 항상 메시지 끝에 AIMessage 를 추가하므로
    마지막 메시지의 tool_calls 만 확인하면 된다.
    isinstance 대신 duck-typing 을 사용해 mock 객체도 처리한다.
    """
    messages: list = state.get("react_messages") or []
    if not messages:
        return "__end__"
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls and state.get("react_iterations", 0) < MAX_ITERATIONS:
        return "react_act"
    return "__end__"
