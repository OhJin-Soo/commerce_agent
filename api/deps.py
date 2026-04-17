from fastapi import Request


def get_graph(request: Request):
    """app.state.graph 에 저장된 컴파일된 StateGraph 를 반환한다."""
    return request.app.state.graph


def get_exchange_rate(request: Request) -> float:
    """앱 시작 시 조회한 INR→KRW 환율을 반환한다."""
    return request.app.state.exchange_rate
