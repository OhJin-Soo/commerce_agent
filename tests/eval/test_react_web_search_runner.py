from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.eval.react_web_search_golden_set import ReactWebSearchGoldenCase
from tests.eval.react_web_search_runner import run_react_web_search_eval


def _tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=f"id_{name}", name=name)


def _make_graph(state: dict) -> MagicMock:
    async def ainvoke(_state: dict):
        return state

    graph = MagicMock()
    graph.ainvoke = ainvoke
    return graph


class TestReactWebSearchRunner:
    async def test_summary_counts_search_web_success(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "search_web", "args": {"query": "Sony WH-1000XM5 리뷰"}, "id": "1", "type": "tool_call"},
        ])
        graph = _make_graph(
            {
                "response": "Sony WH-1000XM5 리뷰를 보면 착용감이 좋습니다.",
                "react_messages": [
                    ai,
                    _tool_msg("search_web", '{"results":[{"title":"Sony WH-1000XM5 리뷰","url":"https://example.com/review","content":"좋음"}]}'),
                ],
                "react_iterations": 1,
                "llm_total_tokens": 50,
            }
        )
        summary = await run_react_web_search_eval(
            graph,
            cases=[ReactWebSearchGoldenCase("Sony WH-1000XM5 리뷰")],
            model_name="test-model",
        )
        assert summary.model_name == "test-model"
        assert summary.search_web_recall == pytest.approx(1.0)
        assert summary.search_success_rate == pytest.approx(1.0)
        assert summary.avg_source_count == pytest.approx(1.0)
        assert summary.grounding_rate == pytest.approx(1.0)

    async def test_tool_unsupported_is_split_out(self):
        graph = _make_graph(
            {
                "response": "",
                "error": "registry.ollama.ai/library/deepseek-r1:8b does not support tools (status code: 400)",
                "react_messages": [],
                "react_iterations": 0,
            }
        )
        summary = await run_react_web_search_eval(
            graph,
            cases=[ReactWebSearchGoldenCase("이어폰 후기 알려줘")],
        )
        assert summary.tool_unsupported_rate == pytest.approx(1.0)
        assert summary.infra_failure_rate == pytest.approx(0.0)
