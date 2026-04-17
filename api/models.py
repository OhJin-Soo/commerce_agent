from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["이어폰 5만원 이하"])


class QueryResponse(BaseModel):
    response: str = ""
    intent: str = ""
    category: str | None = None
    sql_rows: list[dict] = Field(default_factory=list)
    error: str | None = None
