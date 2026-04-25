from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from api.deps import get_exchange_rate
from api.models import QueryRequest, QueryResponse
from observability.langsmith import build_run_config

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_graph(request: Request, model: str):
    """app.state.graphs[model] 또는 app.state.graph 를 반환한다."""
    graphs: dict = getattr(request.app.state, "graphs", None) or {}
    if graphs:
        if model not in graphs:
            available = list(graphs.keys())
            raise HTTPException(
                status_code=400,
                detail=f"모델 '{model}'을 찾을 수 없습니다. 사용 가능한 모델: {available}",
            )
        return graphs[model]
    # 하위 호환: 단일 그래프 모드 (테스트 및 레거시 배포)
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="그래프가 초기화되지 않았습니다.")
    return graph


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="상품 검색 / 추천 쿼리",
    description=(
        "자연어 쿼리를 받아 LangGraph 에이전트를 실행한다.\n\n"
        "**파이프라인 경로** (`use_react=false`, 기본)\n"
        "- 구조적 질문(`이어폰 5만원 이하`) → DB SQL 조회\n"
        "- 추천·해석 질문(`이어폰 추천해줘`) → LLM 응답\n"
        "- 데이터 미적재 시 Kaggle 자동 인제스트 후 SQL 조회\n\n"
        "**ReAct 경로** (`use_react=true`)\n"
        "- LLM이 Thought→Act→Observe 루프로 도구를 직접 선택·실행\n"
        "- 응답의 `react_steps` 필드에서 도구 호출 횟수 확인 가능\n\n"
        "**모델 선택** (`model` 필드)\n"
        "- `llama3.1:8b` (기본), `deepseek-r1:8b`, `gemma4:26b` 등 서버에 로드된 모델 지정"
    ),
)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    graph = _resolve_graph(request, body.model)
    logger.info(
        "POST /query  query=%r  use_react=%s  model=%s",
        body.query, body.use_react, body.model,
    )
    try:
        run_config = build_run_config(
            "api.query",
            tags=["api", "query", "react" if body.use_react else "pipeline", body.model],
            metadata={
                "model": body.model,
                "use_react": body.use_react,
                "query_length": len(body.query),
            },
        )
        result: dict = await graph.ainvoke(
            {"query": body.query, "use_react": body.use_react},
            config=run_config,
        )
    except Exception as exc:
        logger.exception("graph.ainvoke failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        response=result.get("response", ""),
        intent=result.get("intent", ""),
        category=result.get("category"),
        sql_rows=result.get("sql_rows", []),
        error=result.get("error"),
        react_steps=result.get("react_iterations", 0),
        model=body.model,
    )


@router.get("/rate", summary="INR→KRW 환율 조회")
async def rate_endpoint(rate: float = Depends(get_exchange_rate)) -> dict:
    """앱 시작 시 조회한 실시간 INR→KRW 환율을 반환한다."""
    return {"inr_to_krw": rate}
