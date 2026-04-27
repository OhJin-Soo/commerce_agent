"""Tavily web search 평가용 골든셋."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebSearchGoldenCase:
    query: str
    min_results: int = 1


WEB_SEARCH_GOLDEN_SET: list[WebSearchGoldenCase] = [
    WebSearchGoldenCase(query="Sony WH-1000XM5 리뷰"),
    WebSearchGoldenCase(query="이어폰 후기 알려줘"),
    WebSearchGoldenCase(query="노트북 사용자 평가"),
    WebSearchGoldenCase(query="실사용 후기가 좋은 스피커"),
]
