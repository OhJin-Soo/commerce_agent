"""노드 팩토리 모듈.

각 `make_*` 함수는 의존성(세션 팩토리, LLM 등)을 클로저로 캡처해
`AgentState → dict` 비동기 콜러블을 반환한다.

graph.py 는 이 팩토리들을 조합해 StateGraph 에 노드를 등록하기만 하면 된다.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Awaitable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.state import AgentState, Intent
from pipeline.ingestion import IngestionPipeline
from pipeline.kaggle_load import KaggleDatasetConfig

logger = logging.getLogger(__name__)

# 노드 함수 타입 alias
NodeFn = Callable[[AgentState], Awaitable[dict]]


# ---------------------------------------------------------------------------
# 1. intent_router  — 의존성 없음
# ---------------------------------------------------------------------------

# 추천·해석 요청 키워드 → "llm"
_LLM_KEYWORDS = ("추천", "비교", "어떤", "왜", "설명", "어때", "좋은")
# 데이터 적재 요청 키워드 → "ingest"
_INGEST_KEYWORDS = ("ingest", "load", "적재", "가져와", "다운로드", "없어")


def make_intent_router() -> NodeFn:
    """키워드 기반 인텐트 분류기.

    Phase 2에서 LLM 기반 intent 분류로 교체 예정.
    """

    async def intent_router(state: AgentState) -> dict:
        q = state["query"].lower()

        if any(kw in q for kw in _LLM_KEYWORDS):
            intent: Intent = "llm"
        elif any(kw in q for kw in _INGEST_KEYWORDS):
            intent = "ingest"
        else:
            intent = "sql"

        logger.debug("intent_router: query=%r → intent=%s", state["query"], intent)
        return {"intent": intent}

    return intent_router


# ---------------------------------------------------------------------------
# 2. sql_query  — session_factory 필요
# ---------------------------------------------------------------------------

# "5만원", "100만원", "3천원", "50000원" 모두 매칭
_PRICE_RE = re.compile(r"(\d[\d,]*)\s*(만|천)?\s*원")
_UNIT = {"만": 10_000, "천": 1_000}


def _build_sql(query: str) -> str:
    """간단한 키워드 → SQL 변환 (Phase 1 스켈레톤).

    Phase 2에서 LangChain SQLAgent 또는 LLM 기반 NL→SQL 로 교체 예정.
    """
    q = query.lower()
    conditions: list[str] = []

    # 가격 조건 (한국어 단위 만/천 포함)
    m = _PRICE_RE.search(q)
    if m:
        base = int(m.group(1).replace(",", ""))
        unit = _UNIT.get(m.group(2) or "", 1)
        amount = base * unit
        if "이하" in q or "미만" in q:
            conditions.append(f"price <= {amount}")
        elif "이상" in q or "초과" in q:
            conditions.append(f"price >= {amount}")

    # 카테고리 키워드 (확장 가능)
    for keyword, col_value in (
        ("이어폰", "Electronics"),
        ("노트북", "Computers"),
        ("스마트폰", "Cell Phones"),
    ):
        if keyword in q:
            conditions.append(f"category = '{col_value}'")
            break

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        f"SELECT id, name, brand, category, price, rating, review_count "
        f"FROM normalized_products {where} ORDER BY rating DESC NULLS LAST LIMIT 20"
    )


def make_sql_node(session_factory: async_sessionmaker) -> NodeFn:
    async def sql_query(state: AgentState) -> dict:
        sql = _build_sql(state["query"])
        logger.debug("sql_query: %s", sql)
        try:
            async with session_factory() as session:
                result = await session.execute(text(sql))
                rows = [dict(row._mapping) for row in result.fetchall()]
            logger.info("sql_query: %d rows returned", len(rows))
            return {"sql_rows": rows}
        except Exception as exc:
            logger.error("sql_query failed: %s", exc)
            return {"sql_rows": [], "error": str(exc)}

    return sql_query


# ---------------------------------------------------------------------------
# 3. kaggle_ingest  — session_factory + KaggleDatasetConfig 필요
# ---------------------------------------------------------------------------

def make_kaggle_ingest_node(
    session_factory: async_sessionmaker,
    config: KaggleDatasetConfig,
    source_site: str = "kaggle/amazon-products",
) -> NodeFn:
    """Kaggle 데이터셋을 내려받아 normalized_products 에 upsert 한다.

    완료 후 graph.py 에서 sql_query 노드로 라우팅된다.
    """

    async def kaggle_ingest(state: AgentState) -> dict:
        logger.info("kaggle_ingest: starting ingestion handle=%s", config.handle)
        try:
            async with session_factory() as session:
                pipeline = IngestionPipeline(
                    source_site=source_site,
                    session=session,
                )
                products = await pipeline.run(config)
            logger.info("kaggle_ingest: upserted %d products", len(products))
            return {}
        except Exception as exc:
            logger.error("kaggle_ingest failed: %s", exc)
            return {"error": str(exc)}

    return kaggle_ingest


# ---------------------------------------------------------------------------
# 4. llm_respond  — BaseChatModel 필요
# ---------------------------------------------------------------------------

def make_llm_respond_node(llm) -> NodeFn:  # type: ignore[type-arg]
    """sql_rows 또는 query 만으로 자연어 응답을 생성한다.

    `llm` 은 `langchain_core.language_models.BaseChatModel` 호환 객체.
    Phase 2에서 HuggingFace 서빙 엔드포인트로 교체 예정.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    SYSTEM = (
        "당신은 커머스 어시스턴트입니다. "
        "상품 데이터를 기반으로 사용자 질문에 한국어로 간결하게 답변하세요."
    )

    async def llm_respond(state: AgentState) -> dict:
        rows = state.get("sql_rows", [])
        context = (
            "\n".join(
                f"- {r.get('name')} ({r.get('brand')}) "
                f"가격={r.get('price')}원 평점={r.get('rating')}"
                for r in rows[:10]
            )
            if rows
            else "관련 상품 데이터 없음"
        )
        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"질문: {state['query']}\n\n상품 데이터:\n{context}"),
        ]
        try:
            ai_msg = await llm.ainvoke(messages)
            return {"response": ai_msg.content}
        except Exception as exc:
            logger.error("llm_respond failed: %s", exc)
            return {"response": "", "error": str(exc)}

    return llm_respond
