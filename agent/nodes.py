"""노드 팩토리 모듈.

각 `make_*` 함수는 의존성(세션 팩토리, LLM 등)을 클로저로 캡처해
`AgentState → dict` 비동기 콜러블을 반환한다.

graph.py 는 이 팩토리들을 조합해 StateGraph 에 노드를 등록하기만 하면 된다.

노드 흐름::

    classify_intent → check_loaded → run_ingestion? → run_sql → generate_response
    classify_intent → generate_response  (llm 인텐트)
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

NodeFn = Callable[[AgentState], Awaitable[dict]]

# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------

# 카테고리 키워드 → DB category 값 매핑
_CATEGORY_MAP: dict[str, str] = {
    "이어폰": "Electronics",
    "헤드폰": "Electronics",
    "스피커": "Electronics",
    "노트북": "Computers",
    "컴퓨터": "Computers",
    "스마트폰": "Cell Phones",
    "핸드폰": "Cell Phones",
    "태블릿": "Tablets",
}

# 추천·해석 요청 키워드 → "llm" 인텐트
_LLM_KEYWORDS = ("추천", "비교", "어떤", "왜", "설명", "어때", "좋은")

# 가격 표현: "5만원", "100만원", "3천원", "50000원"
_PRICE_RE = re.compile(r"(\d[\d,]*)\s*(만|천)?\s*원")
_UNIT = {"만": 10_000, "천": 1_000}


# ---------------------------------------------------------------------------
# 1. classify_intent  — 의존성 없음
# ---------------------------------------------------------------------------

def make_classify_intent_node() -> NodeFn:
    """쿼리에서 intent(sql/llm)와 category 를 추출한다.

    Phase 2에서 LLM 기반 structured output 분류로 교체 예정.
    """

    async def classify_intent(state: AgentState) -> dict:
        q = state["query"].lower()

        intent: Intent = "llm" if any(kw in q for kw in _LLM_KEYWORDS) else "sql"

        category: str | None = next(
            (cat for kw, cat in _CATEGORY_MAP.items() if kw in q),
            None,
        )

        logger.debug(
            "classify_intent: query=%r → intent=%s, category=%s",
            state["query"], intent, category,
        )
        return {"intent": intent, "category": category}

    return classify_intent


# ---------------------------------------------------------------------------
# 2. check_loaded  — session_factory 필요
# ---------------------------------------------------------------------------

def make_check_loaded_node(session_factory: async_sessionmaker) -> NodeFn:
    """DB 에 해당 카테고리 데이터가 적재돼 있는지 확인한다.

    - category 가 None 이면 data_loaded=False (→ 전체 적재 시도)
    - 1건 이상 존재하면 data_loaded=True  → run_sql 로 직행
    - 0건이면 data_loaded=False → run_ingestion 경유
    """

    async def check_loaded(state: AgentState) -> dict:
        category = state.get("category")

        try:
            async with session_factory() as session:
                if category:
                    result = await session.execute(
                        text(
                            "SELECT 1 FROM normalized_products "
                            "WHERE category = :cat LIMIT 1"
                        ),
                        {"cat": category},
                    )
                else:
                    result = await session.execute(
                        text("SELECT 1 FROM normalized_products LIMIT 1")
                    )
                row = result.fetchone()

            loaded = row is not None
            logger.debug("check_loaded: category=%r → data_loaded=%s", category, loaded)
            return {"data_loaded": loaded}

        except Exception as exc:
            logger.error("check_loaded failed: %s", exc)
            # DB 확인 자체가 실패하면 적재를 시도하지 않고 오류로 처리
            return {"data_loaded": False, "error": str(exc)}

    return check_loaded


# ---------------------------------------------------------------------------
# 3. run_ingestion  — session_factory + KaggleDatasetConfig 필요
# ---------------------------------------------------------------------------

def make_run_ingestion_node(
    session_factory: async_sessionmaker,
    config: KaggleDatasetConfig,
    source_site: str = "kaggle/amazon-products",
) -> NodeFn:
    """Kaggle 데이터셋을 내려받아 normalized_products 에 upsert 한다.

    완료 후 graph.py 에서 run_sql 노드로 라우팅된다.
    """

    async def run_ingestion(state: AgentState) -> dict:
        logger.info("run_ingestion: handle=%s", config.handle)
        try:
            async with session_factory() as session:
                pipeline = IngestionPipeline(
                    source_site=source_site,
                    session=session,
                )
                products = await pipeline.run(config)
            logger.info("run_ingestion: upserted %d products", len(products))
            return {}
        except Exception as exc:
            logger.error("run_ingestion failed: %s", exc)
            return {"error": str(exc)}

    return run_ingestion


# ---------------------------------------------------------------------------
# 4. run_sql  — session_factory 필요
# ---------------------------------------------------------------------------

def _build_sql(query: str, category: str | None = None) -> str:
    """키워드 → SQL 변환 (Phase 1 스켈레톤).

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

    # 카테고리 조건 (classify_intent 가 추출한 값 우선 사용)
    if category:
        conditions.append(f"category = '{category}'")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        "SELECT id, name, brand, category, price, rating, review_count "
        f"FROM normalized_products {where} ORDER BY rating DESC NULLS LAST LIMIT 20"
    )


def make_run_sql_node(session_factory: async_sessionmaker) -> NodeFn:
    async def run_sql(state: AgentState) -> dict:
        sql = _build_sql(state["query"], state.get("category"))
        logger.debug("run_sql: %s", sql)
        try:
            async with session_factory() as session:
                result = await session.execute(text(sql))
                rows = [dict(row._mapping) for row in result.fetchall()]
            logger.info("run_sql: %d rows returned", len(rows))
            return {"sql_rows": rows}
        except Exception as exc:
            logger.error("run_sql failed: %s", exc)
            return {"sql_rows": [], "error": str(exc)}

    return run_sql


# ---------------------------------------------------------------------------
# 5. generate_response  — BaseChatModel 필요
# ---------------------------------------------------------------------------

def make_generate_response_node(llm) -> NodeFn:  # type: ignore[type-arg]
    """sql_rows(있으면) + query 를 바탕으로 자연어 응답을 생성한다.

    - sql 인텐트: DB 결과를 요약·정렬해 사용자에게 전달
    - llm 인텐트: 상품 데이터 없이 LLM 이 직접 답변

    `llm` 은 `langchain_core.language_models.BaseChatModel` 호환 객체.
    Phase 2에서 HuggingFace 서빙 엔드포인트로 교체 예정.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    SYSTEM = (
        "당신은 커머스 어시스턴트입니다. "
        "상품 데이터를 기반으로 사용자 질문에 한국어로 간결하게 답변하세요."
    )

    async def generate_response(state: AgentState) -> dict:
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
            logger.error("generate_response failed: %s", exc)
            return {"response": "", "error": str(exc)}

    return generate_response
