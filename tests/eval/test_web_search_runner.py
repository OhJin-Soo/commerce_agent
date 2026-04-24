from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.eval.web_search_golden_set import WebSearchGoldenCase
from tests.eval.web_search_runner import run_web_search_eval


def _make_node(states: list[dict]) -> MagicMock:
    async def node(_state: dict):
        return states.pop(0)

    return MagicMock(side_effect=node)


class TestWebSearchRunner:
    async def test_success_summary(self):
        web_node = _make_node([
            {
                "web_results": [
                    {"title": "Sony WH-1000XM5 리뷰", "url": "https://x.com", "content": "좋음"}
                ]
            }
        ])
        response_node = _make_node([
            {
                "response": "Sony WH-1000XM5 리뷰를 보면 전반적으로 좋습니다.",
                "response_llm_calls": 1,
                "llm_total_tokens": 42,
            }
        ])
        summary = await run_web_search_eval(
            web_node,
            response_node,
            cases=[WebSearchGoldenCase("Sony WH-1000XM5 리뷰", min_results=1)],
            model_name="test-model",
        )
        assert summary.model_name == "test-model"
        assert summary.search_success_rate == pytest.approx(1.0)
        assert summary.avg_source_count == pytest.approx(1.0)
        assert summary.avg_llm_calls == pytest.approx(1.0)
        assert summary.avg_total_tokens == pytest.approx(42.0)

    async def test_empty_results_fails_min_results(self):
        web_node = _make_node([{"web_results": [], "error": "missing key"}])
        response_node = _make_node([{"response": "", "response_llm_calls": 1}])
        summary = await run_web_search_eval(
            web_node,
            response_node,
            cases=[WebSearchGoldenCase("이어폰 후기", min_results=1)],
        )
        assert summary.search_success_rate == pytest.approx(0.0)
