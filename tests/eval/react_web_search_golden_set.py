"""ReAct search_web 전용 평가용 골든셋."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReactWebSearchGoldenCase:
    query: str
    min_results: int = 1


REACT_WEB_SEARCH_GOLDEN_SET: list[ReactWebSearchGoldenCase] = [
    ReactWebSearchGoldenCase(query="Sony WH-1000XM5 리뷰"),
    ReactWebSearchGoldenCase(query="이어폰 후기 알려줘"),
    ReactWebSearchGoldenCase(query="노트북 사용자 평가"),
    ReactWebSearchGoldenCase(query="실사용 후기가 좋은 스피커"),
]
