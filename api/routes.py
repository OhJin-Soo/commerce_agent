from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_graph
from api.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="상품 검색 / 추천 쿼리",
    description=(
        "자연어 쿼리를 받아 LangGraph 에이전트를 실행한다.\n\n"
        "- 구조적 질문(`이어폰 5만원 이하`) → DB SQL 조회\n"
        "- 추천·해석 질문(`이어폰 추천해줘`) → LLM 응답\n"
        "- 데이터 미적재 시 Kaggle 자동 인제스트 후 SQL 조회"
    ),
)
async def query_endpoint(
    body: QueryRequest,
    graph=Depends(get_graph),
) -> QueryResponse:
    logger.info("POST /query  query=%r", body.query)
    try:
        result: dict = await graph.ainvoke({"query": body.query})
    except Exception as exc:
        logger.exception("graph.ainvoke failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        response=result.get("response", ""),
        intent=result.get("intent", ""),
        category=result.get("category"),
        sql_rows=result.get("sql_rows", []),
        error=result.get("error"),
    )
