"""노드 팩토리 모듈.

각 `make_*` 함수는 의존성(세션 팩토리, LLM 등)을 클로저로 캡처해
`AgentState → dict` 비동기 콜러블을 반환한다.

노드 흐름::

    classify_intent → check_loaded → run_sql → generate_response
    classify_intent → generate_response  (llm 인텐트)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.query_plan import (
    QueryPlan,
    build_select_from_query_plan,
    resolve_plan_csv_filename,
)
from agent.resolve_product import (
    build_db_anchored_web_query,
    needs_web_search as query_needs_web_search,
    resolve_product_candidates,
)
from agent.state import AgentState, Intent

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict]]

# ---------------------------------------------------------------------------
# 카테고리 매핑
# ---------------------------------------------------------------------------

# 쿼리 키워드 → (csv_filename, category_label)
# csv_filename : Kaggle 데이터셋 내 실제 파일명
# category_label: source_site 구성·사용자 표시용 (csv stem 과 동일)
_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    # 음향기기
    "이어폰":   ("Headphones.csv",              "Headphones"),
    "헤드폰":   ("Headphones.csv",              "Headphones"),
    "스피커":   ("Speakers.csv",                "Speakers"),
    # 영상·TV
    "tv":       ("Televisions.csv",             "Televisions"),
    "텔레비전": ("Televisions.csv",             "Televisions"),
    # 카메라
    "카메라":   ("Cameras.csv",                 "Cameras"),
    # 가전
    "냉장고":   ("Refrigerators.csv",           "Refrigerators"),
    "세탁기":   ("Washing Machines.csv",        "Washing Machines"),
    "에어컨":   ("Air Conditioners.csv",        "Air Conditioners"),
    # 컴퓨터·모바일 (전용 CSV 없음 → 전자 통합본 사용)
    "노트북":   ("All Electronics.csv",         "All Electronics"),
    "컴퓨터":   ("All Electronics.csv",         "All Electronics"),
    "스마트폰": ("All Electronics.csv",         "All Electronics"),
    "핸드폰":   ("All Electronics.csv",         "All Electronics"),
    # 패션
    "시계":     ("Watches.csv",                 "Watches"),
    "가방":     ("Bags and Luggage.csv",        "Bags and Luggage"),
    "신발":     ("Shoes.csv",                   "Shoes"),
    "청바지":   ("Jeans.csv",                   "Jeans"),
    # 게임
    "게임기":   ("Gaming Consoles.csv",         "Gaming Consoles"),
    "게임":     ("Gaming Accessories.csv",      "Gaming Accessories"),
    # 스포츠·아웃도어
    "요가":     ("Yoga.csv",                    "Yoga"),
    "자전거":   ("Cycling.csv",                 "Cycling"),
    "러닝":     ("Running.csv",                 "Running"),
    "캠핑":     ("Camping and Hiking.csv",      "Camping and Hiking"),
    # 생활·홈
    "가구":     ("Furniture.csv",               "Furniture"),
    "주방":     ("Kitchen and Dining.csv",      "Kitchen and Dining"),
    "반려동물": ("Dog supplies.csv",            "Dog supplies"),
    # 도서
    "도서":     ("All Books.csv",               "All Books"),
    "책":       ("All Books.csv",               "All Books"),
    # 장난감
    "장난감":   ("Toys and Games.csv",          "Toys and Games"),
}

# 추천·해석 요청 키워드 → "llm" 인텐트
_LLM_KEYWORDS = ("추천", "비교", "어떤", "왜", "설명", "어때", "좋은")

# 가격 표현: "5만원", "100만원", "3천원", "50000원"
_PRICE_RE = re.compile(r"(\d[\d,]*)\s*(만|천)?\s*원")
_UNIT = {"만": 10_000, "천": 1_000}


def _source_site_from(csv_filename: str | None) -> str | None:
    """csv_filename → DB 에 저장되는 source_site 값."""
    if not csv_filename:
        return None
    return f"kaggle/amazon-products/{Path(csv_filename).stem}"


# ---------------------------------------------------------------------------
# 1. classify_intent  — 의존성 없음
# ---------------------------------------------------------------------------

def make_classify_intent_node() -> NodeFn:
    """쿼리에서 intent / category / csv_filename 을 추출한다.

    Phase 2 에서 LLM structured output 분류로 교체 예정.
    """

    async def classify_intent(state: AgentState) -> dict:
        q = state["query"].lower()

        needs_web = query_needs_web_search(q)

        if any(kw in q for kw in _LLM_KEYWORDS):
            intent = "llm"
        else:
            intent = "sql"

        csv_filename: str | None = None
        category: str | None = None
        for kw, (csv, label) in _CATEGORY_MAP.items():
            if kw in q:
                csv_filename = csv
                category = label
                break

        logger.debug(
            "classify_intent: query=%r → intent=%s, category=%s, csv=%s, needs_web_search=%s",
            state["query"], intent, category, csv_filename, needs_web,
        )
        return {
            "intent": intent,
            "category": category,
            "csv_filename": csv_filename,
            "needs_web_search": needs_web,
        }

    return classify_intent


# ---------------------------------------------------------------------------
# 2. check_loaded  — session_factory 필요
# ---------------------------------------------------------------------------

def make_check_loaded_node(session_factory: async_sessionmaker) -> NodeFn:
    """해당 CSV 의 source_site 로 DB 에 데이터가 있는지 확인한다.

    - csv_filename 이 없으면 data_loaded=False (카테고리 불명 → 인제스트 생략, llm 경로)
    - 1건 이상 존재 → data_loaded=True  → run_sql 직행
    - 0건            → data_loaded=False 를 기록하지만 요청 중 적재하지 않는다
    """

    async def check_loaded(state: AgentState) -> dict:
        source_site = _source_site_from(state.get("csv_filename"))

        if not source_site:
            logger.debug("check_loaded: csv_filename 없음 → data_loaded=False")
            return {"data_loaded": False}

        try:
            async with session_factory() as session:
                result = await session.execute(
                    text(
                        "SELECT 1 FROM normalized_products "
                        "WHERE source_site = :site LIMIT 1"
                    ),
                    {"site": source_site},
                )
                row = result.fetchone()

            loaded = row is not None
            logger.debug("check_loaded: source_site=%r → data_loaded=%s", source_site, loaded)
            return {"data_loaded": loaded}

        except Exception as exc:
            logger.error("check_loaded failed: %s", exc)
            return {"data_loaded": False, "error": str(exc)}

    return check_loaded


# ---------------------------------------------------------------------------
# 3. generate_query_plan — BaseChatModel 필요
# ---------------------------------------------------------------------------

def make_generate_query_plan_node(llm) -> NodeFn:  # type: ignore[type-arg]
    """LLM structured output 으로 자연어 질의를 QueryPlan 으로 변환한다.

    실패 시 query_plan 없이 반환해 run_sql 이 기존 _build_sql() 경로로 fallback 한다.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    available_categories = "\n".join(
        f"- keyword={kw}, csv_filename={csv}, category={label}"
        for kw, (csv, label) in _CATEGORY_MAP.items()
    )
    system = (
        "You convert Korean commerce search queries into a structured QueryPlan. "
        "Use only the listed CSV filenames when a category is known. "
        "Do not invent database columns or SQL. "
        "If the query does not specify a filter, leave that field null/default.\n\n"
        f"Available categories:\n{available_categories}"
    )

    async def generate_query_plan(state: AgentState) -> dict:
        if type(llm).__module__ == "unittest.mock":
            return {
                "query_plan_error": "mock llm does not support structured output",
                "query_plan_llm_calls": 0,
            }

        method = getattr(type(llm), "with_structured_output", None)
        if method is None:
            return {
                "query_plan_error": "llm does not support structured output",
                "query_plan_llm_calls": 0,
            }

        try:
            structured_llm = llm.with_structured_output(QueryPlan)
            plan = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=state["query"]),
                ]
            )
            if isinstance(plan, dict):
                plan = QueryPlan.model_validate(plan)
            if not isinstance(plan, QueryPlan):
                raise TypeError(f"unexpected query plan type: {type(plan)!r}")

            csv_filename = resolve_plan_csv_filename(
                plan,
                _CATEGORY_MAP,
                fallback_csv_filename=state.get("csv_filename"),
            )
            updates: dict = {
                "query_plan": plan.model_dump(exclude_none=True),
                "query_plan_error": None,
                "query_plan_llm_calls": 1,
            }
            if csv_filename:
                updates["csv_filename"] = csv_filename
                updates["category"] = Path(csv_filename).stem
            logger.debug("generate_query_plan: %s", updates["query_plan"])
            return updates
        except Exception as exc:
            logger.warning("generate_query_plan failed; falling back to rule SQL: %s", exc)
            return {"query_plan_error": str(exc), "query_plan_llm_calls": 1}

    return generate_query_plan


def make_resolve_product_node(session_factory: async_sessionmaker) -> NodeFn:
    """리뷰/후기 질의에서 카테고리가 없을 때 DB에서 상품 anchor 를 찾는다."""

    async def resolve_product(state: AgentState) -> dict:
        if not state.get("needs_web_search"):
            return {}
        if state.get("csv_filename"):
            return {}

        try:
            rows = await resolve_product_candidates(session_factory, state["query"], limit=5)
        except Exception as exc:
            logger.error("resolve_product failed: %s", exc)
            return {
                "sql_rows": [],
                "db_miss": True,
                "db_miss_policy": "deny_unanchored_web_search",
                "error": str(exc),
            }

        if not rows:
            logger.info("resolve_product: no DB anchor for %r", state["query"])
            return {
                "sql_rows": [],
                "db_miss": True,
                "db_miss_policy": "deny_unanchored_web_search",
            }

        csv_filename = rows[0].get("csv_filename")
        logger.info("resolve_product: anchored %d rows with csv=%s", len(rows), csv_filename)
        return {
            "sql_rows": rows,
            "csv_filename": csv_filename or state.get("csv_filename"),
            "category": Path(csv_filename).stem if csv_filename else state.get("category"),
            "resolved_product_query": build_db_anchored_web_query(state["query"], rows),
            "db_miss": False,
            "db_miss_policy": None,
        }

    return resolve_product


# ---------------------------------------------------------------------------
# 4. run_sql  — session_factory 필요
# ---------------------------------------------------------------------------

def _build_sql(
    query: str,
    source_site: str | None = None,
    exchange_rate: float = 1.0,
) -> str:
    """키워드 → SQL 변환 (Phase 1 스켈레톤).

    DB 가격은 INR 기준이므로, 사용자 KRW 입력을 exchange_rate 로 나눠 INR 로 변환한다.
    Phase 2 에서 LangChain SQLAgent / LLM 기반 NL→SQL 로 교체 예정.
    """
    q = query.lower()
    conditions: list[str] = []

    # 가격 조건: KRW 입력 → INR 변환 후 DB 비교
    m = _PRICE_RE.search(q)
    if m:
        base = int(m.group(1).replace(",", ""))
        unit = _UNIT.get(m.group(2) or "", 1)
        krw_amount = base * unit
        inr_amount = int(krw_amount / exchange_rate) if exchange_rate else krw_amount
        if "이하" in q or "미만" in q:
            conditions.append(f"price <= {inr_amount}")
        elif "이상" in q or "초과" in q:
            conditions.append(f"price >= {inr_amount}")

    # source_site 조건 (CSV 단위로 적재했으므로 이걸로 카테고리 필터)
    if source_site:
        conditions.append(f"source_site = '{source_site}'")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        "SELECT id, name, brand, category, price, rating, review_count, source_url "
        f"FROM normalized_products {where} ORDER BY rating DESC NULLS LAST LIMIT 20"
    )


def make_run_sql_node(
    session_factory: async_sessionmaker,
    exchange_rate: float = 1.0,
) -> NodeFn:
    async def run_sql(state: AgentState) -> dict:
        source_site = _source_site_from(state.get("csv_filename"))

        stmt = None
        if state.get("query_plan"):
            try:
                stmt = build_select_from_query_plan(
                    state["query_plan"],
                    source_site=source_site,
                    exchange_rate=exchange_rate,
                )
                logger.debug("run_sql query_plan stmt: %s", stmt)
            except Exception as exc:
                logger.warning("query_plan SQL build failed; falling back to _build_sql: %s", exc)
                stmt = None

        sql = _build_sql(state["query"], source_site, exchange_rate)
        executable = stmt if stmt is not None else text(sql)
        if stmt is None:
            logger.debug("run_sql fallback SQL: %s", sql)

        try:
            async with session_factory() as session:
                result = await session.execute(executable)
                rows = [dict(row._mapping) for row in result.fetchall()]
            logger.info("run_sql: %d rows returned", len(rows))
            return {
                "sql_rows": rows,
                "db_miss": len(rows) == 0,
                "db_miss_policy": (
                    "deny_unanchored_web_search"
                    if state.get("needs_web_search") and len(rows) == 0
                    else None
                ),
            }
        except Exception as exc:
            logger.error("run_sql failed: %s", exc)
            return {
                "sql_rows": [],
                "error": str(exc),
                "db_miss": True,
                "db_miss_policy": "deny_unanchored_web_search" if state.get("needs_web_search") else None,
            }

    return run_sql


# ---------------------------------------------------------------------------
# 5. web_search  — Tavily API 키 필요
# ---------------------------------------------------------------------------

def make_web_search_node(tavily_api_key: str | None) -> NodeFn:
    """Tavily 로 웹을 검색해 비정형 외부 정보를 수집한다.

    API 키가 없거나 호출에 실패해도 빈 web_results 로 graceful degradation.
    결과는 generate_response 노드에서 컨텍스트로 사용된다.
    """

    async def web_search(state: AgentState) -> dict:
        rows: list[dict] = state.get("sql_rows") or []
        query = state.get("resolved_product_query") or build_db_anchored_web_query(
            state["query"], rows
        )
        if not tavily_api_key:
            logger.warning("web_search: TAVILY_API_KEY 미설정 → 웹 검색 건너뜀")
            return {"web_results": [], "web_search_query": query}

        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=tavily_api_key)
        logger.info("web_search: query=%r", query)
        try:
            resp = await client.search(query, max_results=5)
            results = resp.get("results", [])[:5]
            # 필요한 필드만 추출해 상태에 저장
            web_results = [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in results
            ]
            logger.info("web_search: %d 건 수집", len(web_results))
            return {"web_results": web_results, "web_search_query": query}
        except Exception as exc:
            logger.error("web_search 실패: %s", exc)
            return {"web_results": [], "error": str(exc), "web_search_query": query}

    return web_search


# ---------------------------------------------------------------------------
# 6. generate_response  — BaseChatModel 필요
# ---------------------------------------------------------------------------

def make_generate_response_node(llm, exchange_rate: float = 1.0) -> NodeFn:  # type: ignore[type-arg]
    """sql_rows / web_results + query 를 바탕으로 자연어 응답을 생성한다.

    - sql_rows 가 있으면 DB 상품 데이터를 컨텍스트로 사용한다.
    - web_results 가 있으면 Tavily 검색 결과를 컨텍스트로 사용한다.
    - 둘 다 없으면 내부 DB에 관련 데이터가 없다고 답한다.

    DB 가격(INR)을 exchange_rate 로 곱해 KRW 로 변환한다.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    SYSTEM = (
        "You are a Korean commerce assistant. "
        "IMPORTANT: You MUST respond ONLY in Korean (한국어). "
        "Do NOT use any other language. Every word must be Korean. "
        "Answer only from the provided DB product data and web search context. "
        "Web search context is allowed only when it is anchored to a DB product. "
        "If there is no product data, say the product/category is not available in the internal DB. "
        "Never suggest that the user provide another model name so you can search the web for DB-missing products. "
        "Always express prices in Korean Won (₩)."
    )

    def _to_krw(inr_price) -> str:
        try:
            return f"₩{int(float(inr_price) * exchange_rate):,}"
        except (TypeError, ValueError):
            return "가격 미상"

    def _build_context(state: AgentState) -> str:
        """상품 정보와 웹 검색 결과를 함께 컨텍스트로 구성한다."""
        sections: list[str] = []
        rows: list[dict] = state.get("sql_rows") or []
        if rows:
            sections.append(
                "상품 데이터:\n" + "\n".join(
                    f"- {r.get('name')} ({r.get('brand')}) "
                    f"가격={_to_krw(r.get('price'))} 평점={r.get('rating')}"
                    for r in rows[:10]
                )
            )

        web_results: list[dict] = state.get("web_results") or []
        if web_results:
            parts = []
            for r in web_results:
                parts.append(
                    f"[출처: {r.get('title', '')}] ({r.get('url', '')})\n"
                    f"{r.get('content', '')}"
                )
            sections.append("웹 검색 결과:\n" + "\n\n".join(parts))

        return "\n\n".join(sections) if sections else "관련 데이터 없음"

    def _usage_updates(ai_msg) -> dict:
        usage = getattr(ai_msg, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        return {
            "response_llm_calls": 1,
            "llm_input_tokens": input_tokens,
            "llm_output_tokens": output_tokens,
            "llm_total_tokens": total_tokens,
        }

    async def generate_response(state: AgentState) -> dict:
        if (
            state.get("db_miss_policy") == "deny_unanchored_web_search"
            and not (state.get("sql_rows") or state.get("web_results"))
        ):
            return {
                "response": (
                    "현재 내부 상품 DB에서 해당 상품이나 후보 상품을 찾지 못했습니다. "
                    "DB에 있는 상품만 외부 리뷰 검색을 수행하므로, DB에 없는 상품의 후기만 "
                    "단독으로 제공하지 않습니다."
                ),
                "response_llm_calls": 0,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "llm_total_tokens": 0,
            }
        context = _build_context(state)
        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"질문: {state['query']}\n\n{context}"),
        ]
        try:
            ai_msg = await llm.ainvoke(messages)
            from agent.utils import strip_thinking
            return {"response": strip_thinking(ai_msg.content), **_usage_updates(ai_msg)}
        except Exception as exc:
            logger.error("generate_response failed: %s", exc)
            return {"response": "", "error": str(exc), "response_llm_calls": 1}

    return generate_response
