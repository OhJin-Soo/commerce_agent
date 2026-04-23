from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.eval.query_plan_golden_set import QueryPlanGoldenCase
from tests.eval.query_plan_runner import run_query_plan_eval


def _make_graph(states: list[dict]) -> MagicMock:
    async def ainvoke(_state: dict):
        return states.pop(0)

    graph = MagicMock()
    graph.ainvoke = ainvoke
    return graph


class TestQueryPlanRunner:
    async def test_perfect_plan_scores_100_percent(self):
        expected = {
            "csv_filename": "Headphones.csv",
            "category": "Headphones",
            "max_price_krw": 50_000,
            "sort": "rating_desc",
        }
        graph = _make_graph([
            {
                "query_plan": expected,
                "query_plan_error": None,
                "query_plan_llm_calls": 1,
                "llm_total_tokens": 123,
            }
        ])

        summary = await run_query_plan_eval(
            graph,
            cases=[QueryPlanGoldenCase("이어폰 5만원 이하", expected)],
            model_name="test-model",
        )

        assert summary.model_name == "test-model"
        assert summary.n == 1
        assert summary.plan_accuracy == pytest.approx(1.0)
        assert summary.fallback_rate == pytest.approx(0.0)
        assert summary.avg_llm_calls == pytest.approx(1.0)
        assert summary.avg_total_tokens == pytest.approx(123.0)

    async def test_fallback_counts_as_fallback(self):
        expected = {"csv_filename": "Headphones.csv", "max_price_krw": 50_000}
        graph = _make_graph([
            {
                "query_plan": None,
                "query_plan_error": "structured output failed",
                "query_plan_llm_calls": 1,
            }
        ])

        summary = await run_query_plan_eval(
            graph,
            cases=[QueryPlanGoldenCase("이어폰 5만원 이하", expected)],
        )

        assert summary.plan_accuracy == pytest.approx(0.0)
        assert summary.fallback_rate == pytest.approx(1.0)
        assert summary.cases[0].used_fallback is True
