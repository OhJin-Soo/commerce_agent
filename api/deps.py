from fastapi import HTTPException, Request


def get_graph(request: Request, model: str = "llama3.1:8b"):
    """app.state.graphs[model] 에 저장된 컴파일된 StateGraph 를 반환한다.

    단일 그래프(app.state.graph)가 있으면 하위 호환성을 위해 그것을 반환한다.
    """
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


def get_exchange_rate(request: Request) -> float:
    """앱 시작 시 조회한 INR→KRW 환율을 반환한다."""
    return request.app.state.exchange_rate
