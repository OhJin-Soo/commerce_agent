from fastapi import Request


def get_graph(request: Request):
    """app.state.graph 에 저장된 컴파일된 StateGraph 를 반환한다."""
    return request.app.state.graph
