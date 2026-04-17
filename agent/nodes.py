"""노드 팩토리 모듈.

각 `make_*` 함수는 의존성(세션 팩토리, LLM 등)을 클로저로 캡처해
`AgentState → dict` 비동기 콜러블을 반환한다.

노드 흐름::

    classify_intent → check_loaded → run_ingestion? → run_sql → generate_response
    classify_intent → generate_response  (llm 인텐트)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from agent.state import AgentState, Intent
from pipeline.ingestion import IngestionPipeline
from pipeline.kaggle_load import KaggleDatasetConfig

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

        intent: Intent = "llm" if any(kw in q for kw in _LLM_KEYWORDS) else "sql"

        csv_filename: str | None = None
        category: str | None = None
        for kw, (csv, label) in _CATEGORY_MAP.items():
            if kw in q:
                csv_filename = csv
                category = label
                break

        logger.debug(
            "classify_intent: query=%r → intent=%s, category=%s, csv=%s",
            state["query"], intent, category, csv_filename,
        )
        return {"intent": intent, "category": category, "csv_filename": csv_filename}

    return classify_intent


# ---------------------------------------------------------------------------
# 2. check_loaded  — session_factory 필요
# ---------------------------------------------------------------------------

def make_check_loaded_node(session_factory: async_sessionmaker) -> NodeFn:
    """해당 CSV 의 source_site 로 DB 에 데이터가 있는지 확인한다.

    - csv_filename 이 없으면 data_loaded=False (카테고리 불명 → 인제스트 생략, llm 경로)
    - 1건 이상 존재 → data_loaded=True  → run_sql 직행
    - 0건            → data_loaded=False → run_ingestion 경유
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
# 3. run_ingestion  — session_factory + dataset_handle 필요
# ---------------------------------------------------------------------------

def make_run_ingestion_node(
    session_factory: async_sessionmaker,
    dataset_handle: str,
    ingest_nrows: int | None = None,
) -> NodeFn:
    """state["csv_filename"] 에 해당하는 CSV 한 파일만 내려받아 upsert 한다.

    source_site 를 csv_filename stem 으로 설정해 check_loaded 와 run_sql 이
    같은 기준으로 데이터를 조회할 수 있도록 한다.
    """

    async def run_ingestion(state: AgentState) -> dict:
        csv_filename = state.get("csv_filename")
        source_site = _source_site_from(csv_filename) or "kaggle/amazon-products/unknown"

        config = KaggleDatasetConfig(
            handle=dataset_handle,
            filename=csv_filename,   # None 이면 KaggleLoadFilter 가 첫 번째 CSV 사용
            nrows=ingest_nrows,
        )
        logger.info("run_ingestion: file=%s  source_site=%s", csv_filename, source_site)
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

def _build_sql(query: str, source_site: str | None = None) -> str:
    """키워드 → SQL 변환 (Phase 1 스켈레톤).

    Phase 2 에서 LangChain SQLAgent / LLM 기반 NL→SQL 로 교체 예정.
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

    # source_site 조건 (CSV 단위로 적재했으므로 이걸로 카테고리 필터)
    if source_site:
        conditions.append(f"source_site = '{source_site}'")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return (
        "SELECT id, name, brand, category, price, rating, review_count "
        f"FROM normalized_products {where} ORDER BY rating DESC NULLS LAST LIMIT 20"
    )


def make_run_sql_node(session_factory: async_sessionmaker) -> NodeFn:
    async def run_sql(state: AgentState) -> dict:
        source_site = _source_site_from(state.get("csv_filename"))
        sql = _build_sql(state["query"], source_site)
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

    `llm` 은 `langchain_core.language_models.BaseChatModel` 호환 객체.
    Phase 2 에서 HuggingFace 서빙 엔드포인트로 교체 예정.
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
