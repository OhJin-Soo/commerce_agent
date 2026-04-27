from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["이어폰 5만원 이하"])
    use_react: bool = Field(
        False,
        description=(
            "Legacy field. The API now routes automatically: normal DB queries use pipeline, "
            "review/web-search queries use ReAct."
        ),
    )
    model: str = Field(
        "llama3.1:8b",
        description=(
            "Legacy field. The API returns the actually selected model in the response."
        ),
    )


class QueryResponse(BaseModel):
    response: str = ""
    intent: str = ""
    category: str | None = None
    sql_rows: list[dict] = Field(default_factory=list)
    error: str | None = None
    # ReAct 경로 전용: LLM이 도구를 몇 번 호출했는지 (파이프라인 경로에서는 0)
    react_steps: int = 0
    model: str = ""
