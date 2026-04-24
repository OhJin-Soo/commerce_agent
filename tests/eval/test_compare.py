"""compare.py 및 신규 metrics 단위 테스트 — DB·LLM 불필요."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.eval.compare import (
    CompareCase,
    ModelCompareSummary,
    PathMetrics,
    PathResult,
    _classify_error_status,
    _extract_search_web_queries,
    _extract_tool_args,
    _compute_metrics,
    _extract_react_csv,
    _extract_tool_sequence,
    _was_ingest_unnecessary,
    run_compare_eval,
    run_model_compare_eval,
    run_path_eval,
)
from tests.eval.metrics import category_hit, grounding_rate, tool_sequence_metrics, web_grounding_rate
from tests.eval.react_golden_set import REACT_GOLDEN_SET, ReactGoldenCase


# ---------------------------------------------------------------------------
# grounding_rate
# ---------------------------------------------------------------------------

class TestGroundingRate:
    def test_all_mentioned(self):
        rows = [{"name": "Sony WH-1000XM5"}, {"name": "JBL Tune 510BT"}]
        response = "Sony WH-1000XM5 와 JBL Tune 510BT 를 추천합니다."
        assert grounding_rate(response, rows) == pytest.approx(1.0)

    def test_none_mentioned(self):
        rows = [{"name": "Sony WH-1000XM5"}]
        response = "이어폰은 음질이 중요합니다."
        assert grounding_rate(response, rows) == pytest.approx(0.0)

    def test_partial(self):
        rows = [{"name": "Sony WH-1000XM5"}, {"name": "JBL Tune 510BT"}]
        response = "Sony WH-1000XM5 를 추천합니다."
        assert grounding_rate(response, rows) == pytest.approx(0.5)

    def test_empty_rows_returns_1(self):
        """sql_rows 없음 → 지표 미적용 → 1.0 (중립값)."""
        assert grounding_rate("아무 응답", []) == pytest.approx(1.0)

    def test_empty_response_returns_0(self):
        rows = [{"name": "Sony WH-1000XM5"}]
        assert grounding_rate("", rows) == pytest.approx(0.0)

    def test_case_insensitive(self):
        rows = [{"name": "sony wh-1000xm5"}]
        response = "SONY WH-1000XM5 추천"
        assert grounding_rate(response, rows) == pytest.approx(1.0)

    def test_max_10_rows(self):
        """sql_rows 가 10개 초과해도 10개까지만 비교한다."""
        rows = [{"name": f"Product{i}"} for i in range(15)]
        # Product0~9 만 언급 (10개), Product10~14 는 미언급
        response = " ".join(f"Product{i}" for i in range(10))
        assert grounding_rate(response, rows) == pytest.approx(1.0)


class TestWebGroundingRate:
    def test_title_overlap_counts_as_grounded(self):
        results = [{"title": "Sony WH-1000XM5 리뷰", "url": "https://example.com/review"}]
        response = "Sony WH-1000XM5는 착용감과 노이즈 캔슬링 평가가 좋습니다."
        assert web_grounding_rate(response, results) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# category_hit
# ---------------------------------------------------------------------------

class TestCategoryHit:
    def test_same_csv(self):
        assert category_hit("Headphones.csv", "Headphones.csv") is True

    def test_different_csv(self):
        assert category_hit("Headphones.csv", "Speakers.csv") is False

    def test_both_none(self):
        assert category_hit(None, None) is True

    def test_expected_none_actual_set(self):
        assert category_hit(None, "Headphones.csv") is False

    def test_expected_set_actual_none(self):
        assert category_hit("Headphones.csv", None) is False


# ---------------------------------------------------------------------------
# tool_sequence_metrics
# ---------------------------------------------------------------------------

class TestToolSequenceMetrics:
    def test_perfect_required_only(self):
        required = ["search_category", "check_db_loaded", "query_products"]
        actual   = ["search_category", "check_db_loaded", "query_products"]
        m = tool_sequence_metrics(required, [], actual)
        assert m["tool_recall"]    == pytest.approx(1.0)
        assert m["tool_precision"] == pytest.approx(1.0)

    def test_unexpected_ingest_penalized(self):
        """요청 경로에서 ingest_data 를 호출하면 precision 이 낮아져야 한다."""
        required = ["search_category", "check_db_loaded", "query_products"]
        optional = []
        actual   = ["search_category", "check_db_loaded", "ingest_data", "query_products"]
        m = tool_sequence_metrics(required, optional, actual)
        assert m["tool_recall"]    == pytest.approx(1.0)
        assert m["tool_precision"] == pytest.approx(3 / 4)

    def test_missing_required(self):
        """query_products 를 빠뜨리면 recall < 1."""
        required = ["search_category", "check_db_loaded", "query_products"]
        actual   = ["search_category", "check_db_loaded"]
        m = tool_sequence_metrics(required, [], actual)
        assert m["tool_recall"] == pytest.approx(2 / 3)
        assert m["tool_precision"] == pytest.approx(1.0)

    def test_unexpected_tool(self):
        """allowed 에 없는 도구를 호출하면 precision < 1."""
        required = ["search_category", "query_products"]
        actual   = ["search_category", "query_products", "unknown_tool"]
        m = tool_sequence_metrics(required, [], actual)
        assert m["tool_precision"] == pytest.approx(2 / 3)
        assert m["tool_recall"]    == pytest.approx(1.0)

    def test_empty_actual(self):
        """도구를 하나도 호출하지 않으면 recall=0, precision=1 (분모=0 방어)."""
        m = tool_sequence_metrics(["search_category"], [], [])
        assert m["tool_recall"]    == pytest.approx(0.0)
        assert m["tool_precision"] == pytest.approx(1.0)  # 분모 0 → 기본값

    def test_empty_required(self):
        """required 가 비어 있으면 recall=1.0."""
        m = tool_sequence_metrics([], [], ["some_tool"])
        assert m["tool_recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# react_messages 파싱 유틸
# ---------------------------------------------------------------------------

def _tool_msg(name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=f"id_{name}", name=name)


class TestExtractToolSequence:
    def test_extracts_tool_names_in_order(self):
        msgs = [
            _tool_msg("search_category", '{"csv_filename": "Headphones.csv"}'),
            _tool_msg("check_db_loaded", '{"loaded": false}'),
            _tool_msg("ingest_data",     '{"upserted": 100}'),
            _tool_msg("query_products",  '{"products": []}'),
        ]
        assert _extract_tool_sequence(msgs) == [
            "search_category", "check_db_loaded", "ingest_data", "query_products"
        ]

    def test_skips_non_tool_messages(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="query"),
            _tool_msg("search_category", "{}"),
        ]
        assert _extract_tool_sequence(msgs) == ["search_category"]

    def test_empty_messages(self):
        assert _extract_tool_sequence([]) == []


class TestExtractToolArgs:
    def test_extracts_last_tool_args_from_ai_message(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "query_products", "args": {"csv_filename": "Headphones.csv", "limit": 10}, "id": "1", "type": "tool_call"},
        ])
        assert _extract_tool_args([ai], "query_products") == {"csv_filename": "Headphones.csv", "limit": 10}


class TestExtractSearchWebQueries:
    def test_extracts_search_web_queries(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "search_web", "args": {"query": "Sony WH-1000XM5 리뷰"}, "id": "1", "type": "tool_call"},
        ])
        assert _extract_search_web_queries([ai]) == ["Sony WH-1000XM5 리뷰"]


class TestExtractReactCsv:
    def test_extracts_csv_from_search_category(self):
        msgs = [
            _tool_msg("search_category", '{"csv_filename": "Speakers.csv", "category": "Speakers"}'),
        ]
        assert _extract_react_csv(msgs) == "Speakers.csv"

    def test_returns_none_if_not_found(self):
        msgs = [_tool_msg("check_db_loaded", '{"loaded": false}')]
        assert _extract_react_csv(msgs) is None

    def test_returns_none_if_no_csv_filename_in_content(self):
        msgs = [_tool_msg("search_category", '{"category": null}')]
        assert _extract_react_csv(msgs) is None

    def test_handles_invalid_json(self):
        msgs = [_tool_msg("search_category", "NOT JSON")]
        assert _extract_react_csv(msgs) is None


class TestWasIngestUnnecessary:
    def test_unnecessary_when_loaded_true_then_ingest(self):
        msgs = [
            _tool_msg("check_db_loaded", '{"loaded": true}'),
            _tool_msg("ingest_data",     '{"upserted": 50}'),
        ]
        assert _was_ingest_unnecessary(msgs) is True

    def test_necessary_when_loaded_false_then_ingest(self):
        msgs = [
            _tool_msg("check_db_loaded", '{"loaded": false}'),
            _tool_msg("ingest_data",     '{"upserted": 100}'),
        ]
        assert _was_ingest_unnecessary(msgs) is False

    def test_no_ingest_called(self):
        msgs = [
            _tool_msg("check_db_loaded", '{"loaded": true}'),
            _tool_msg("query_products",  '{"products": []}'),
        ]
        assert _was_ingest_unnecessary(msgs) is False

    def test_empty_messages(self):
        assert _was_ingest_unnecessary([]) is False


class TestClassifyErrorStatus:
    def test_tool_unsupported(self):
        assert _classify_error_status("model does not support tools", []) == "tool_unsupported"

    def test_infra_failure(self):
        assert _classify_error_status("TAVILY_API_KEY missing", []) == "infra_failure"


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def _case(self) -> ReactGoldenCase:
        return ReactGoldenCase(
            query="이어폰 5만원 이하",
            csv_filename="Headphones.csv",
            required_tools=["search_category", "check_db_loaded", "query_products"],
            optional_tools=[],
            reference_sql="SELECT 1",
            expected_query_products_args={"csv_filename": "Headphones.csv", "max_price_krw": 50000, "limit": 10},
        )

    def test_pipeline_category_hit(self):
        result = PathResult(csv_filename="Headphones.csv", response="테스트", sql_rows=[])
        m = _compute_metrics(result, [], self._case(), is_react=False)
        assert m.category_hit is True

    def test_pipeline_category_miss(self):
        result = PathResult(csv_filename="Speakers.csv", response="", sql_rows=[])
        m = _compute_metrics(result, [], self._case(), is_react=False)
        assert m.category_hit is False

    def test_react_tool_recall(self):
        msgs = [
            _tool_msg("search_category", '{"csv_filename":"Headphones.csv"}'),
            _tool_msg("check_db_loaded", '{"loaded":true}'),
            # query_products 빠짐
        ]
        result = PathResult(
            csv_filename="Headphones.csv",
            response="",
            sql_rows=[],
            tool_sequence=["search_category", "check_db_loaded"],
            react_messages=msgs,
        )
        m = _compute_metrics(result, [], self._case(), is_react=True)
        assert m.tool_recall == pytest.approx(2 / 3)

    def test_react_tool_argument_accuracy(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "query_products", "args": {"csv_filename": "Headphones.csv", "max_price_krw": 50000, "limit": 10}, "id": "1", "type": "tool_call"},
        ])
        result = PathResult(
            csv_filename="Headphones.csv",
            response="",
            sql_rows=[],
            tool_sequence=["search_category", "check_db_loaded", "query_products"],
            react_messages=[ai],
            tool_arguments={"query_products": {"csv_filename": "Headphones.csv", "max_price_krw": 50000, "limit": 10}},
        )
        m = _compute_metrics(result, [], self._case(), is_react=True)
        assert m.tool_argument_accuracy == pytest.approx(1.0)

    def test_react_error_status_flags(self):
        result = PathResult(
            csv_filename=None,
            response="",
            sql_rows=[],
            error_status="tool_unsupported",
            tool_sequence=[],
            react_messages=[],
        )
        m = _compute_metrics(result, [], self._case(), is_react=True)
        assert m.tool_unsupported is True
        assert m.infra_failure is False

    def test_react_unnecessary_ingest_detected(self):
        msgs = [
            _tool_msg("check_db_loaded", '{"loaded": true}'),
            _tool_msg("ingest_data",     '{"upserted": 50}'),
            _tool_msg("query_products",  '{"products": []}'),
        ]
        result = PathResult(
            csv_filename="Headphones.csv",
            response="",
            sql_rows=[],
            tool_sequence=["check_db_loaded", "ingest_data", "query_products"],
            react_messages=msgs,
        )
        m = _compute_metrics(result, [], self._case(), is_react=True)
        assert m.unnecessary_ingest is True

    def test_ex_f1_with_ref_rows(self):
        ref  = [{"id": 1}, {"id": 2}]
        rows = [{"id": 1}, {"id": 2}]
        result = PathResult(sql_rows=rows, response="", csv_filename="Headphones.csv")
        m = _compute_metrics(result, ref, self._case(), is_react=False)
        assert m.execution_accuracy == pytest.approx(1.0)
        assert m.result_f1          == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run_compare_eval (그래프 mock)
# ---------------------------------------------------------------------------

def _make_tool_msgs(csv: str, loaded: bool) -> list:
    return [
        _tool_msg("search_category", json.dumps({"csv_filename": csv})),
        _tool_msg("check_db_loaded", json.dumps({"loaded": loaded})),
        _tool_msg("query_products",  json.dumps({"products": []})),
    ]


def _make_graph(
    pipeline_state: dict,
    react_state: dict,
) -> MagicMock:
    """pipeline / react 호출을 번갈아 반환하는 mock 그래프."""
    async def ainvoke(state: dict, **_):
        if state.get("use_react"):
            return react_state
        return pipeline_state

    graph = MagicMock()
    graph.ainvoke = ainvoke
    return graph


class TestRunCompareEval:
    async def test_basic_run_returns_summary(self):
        tool_msgs = _make_tool_msgs("Headphones.csv", loaded=True)
        graph = _make_graph(
            pipeline_state={
                "sql_rows": [],
                "response": "파이프라인 응답",
                "csv_filename": "Headphones.csv",
            },
            react_state={
                "sql_rows": [],
                "response": "ReAct 응답",
                "react_messages": tool_msgs,
                "react_iterations": 3,
            },
        )
        cases = REACT_GOLDEN_SET[:2]
        summary = await run_compare_eval(graph, cases, session_factory=None)

        assert summary.n == 2
        assert summary.has_db_eval is False
        assert len(summary.cases) == 2

    async def test_category_hit_rate_pipeline(self):
        """파이프라인이 올바른 CSV 를 반환하면 category_hit_rate=1.0."""
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Headphones.csv", True),
                "react_iterations": 3,
            },
        )
        summary = await run_compare_eval(graph, REACT_GOLDEN_SET[:1], session_factory=None)
        assert summary.pipeline.category_hit_rate == pytest.approx(1.0)

    async def test_react_llm_calls_counted(self):
        """react_iterations=3 이면 llm_calls=4 (act 3번 + 최종 reason 1번)."""
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Headphones.csv", True),
                "react_iterations": 3,
            },
        )
        summary = await run_compare_eval(graph, REACT_GOLDEN_SET[:1], session_factory=None)
        assert summary.react.avg_llm_calls == pytest.approx(4.0)
        assert summary.pipeline.avg_llm_calls == pytest.approx(1.0)

    async def test_pipeline_llm_calls_include_query_plan(self):
        graph = _make_graph(
            pipeline_state={
                "sql_rows": [],
                "response": "",
                "csv_filename": "Headphones.csv",
                "query_plan": {"csv_filename": "Headphones.csv"},
                "query_plan_llm_calls": 1,
                "response_llm_calls": 1,
            },
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Headphones.csv", True),
                "react_iterations": 1,
            },
        )
        summary = await run_compare_eval(graph, REACT_GOLDEN_SET[:1], session_factory=None)
        assert summary.pipeline.avg_llm_calls == pytest.approx(2.0)

    async def test_tokens_and_cost_are_aggregated(self):
        graph = _make_graph(
            pipeline_state={
                "sql_rows": [],
                "response": "",
                "csv_filename": "Headphones.csv",
                "response_llm_calls": 1,
                "llm_input_tokens": 100,
                "llm_output_tokens": 50,
                "llm_total_tokens": 150,
            },
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Headphones.csv", True),
                "react_iterations": 1,
                "llm_input_tokens": 20,
                "llm_output_tokens": 10,
                "llm_total_tokens": 30,
            },
        )
        summary = await run_compare_eval(
            graph,
            REACT_GOLDEN_SET[:1],
            session_factory=None,
            input_cost_per_1k=0.1,
            output_cost_per_1k=0.2,
        )
        assert summary.pipeline.avg_total_tokens == pytest.approx(150.0)
        assert summary.pipeline.total_estimated_cost == pytest.approx(0.02)

    async def test_tool_recall_all_required(self):
        """required 3개 도구를 모두 호출하면 tool_recall=1.0."""
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Headphones.csv", True),
                "react_iterations": 3,
            },
        )
        summary = await run_compare_eval(graph, REACT_GOLDEN_SET[:1], session_factory=None)
        assert summary.react.tool_recall == pytest.approx(1.0)

    async def test_tool_precision_with_unexpected_tool(self):
        """unknown_tool 을 추가로 호출하면 precision < 1.0."""
        extra_msgs = _make_tool_msgs("Headphones.csv", True) + [
            _tool_msg("unknown_tool", '{}')
        ]
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": extra_msgs,
                "react_iterations": 4,
            },
        )
        summary = await run_compare_eval(graph, REACT_GOLDEN_SET[:1], session_factory=None)
        assert summary.react.tool_precision < 1.0

    async def test_unnecessary_ingest_rate(self):
        """loaded=True 인데 ingest 를 호출하면 unnecessary_ingest_rate=1.0."""
        msgs_with_unnecessary = [
            _tool_msg("search_category", '{"csv_filename":"Headphones.csv"}'),
            _tool_msg("check_db_loaded", '{"loaded": true}'),
            _tool_msg("ingest_data",     '{"upserted": 50}'),   # 불필요
            _tool_msg("query_products",  '{"products": []}'),
        ]
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": msgs_with_unnecessary,
                "react_iterations": 4,
            },
        )
        summary = await run_compare_eval(graph, REACT_GOLDEN_SET[:1], session_factory=None)
        assert summary.react.unnecessary_ingest_rate == pytest.approx(1.0)

    async def test_summary_str_contains_key_labels(self):
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Headphones.csv", True),
                "react_iterations": 3,
            },
        )
        summary = await run_compare_eval(
            graph, REACT_GOLDEN_SET[:1], session_factory=None, model_name="test-model"
        )
        out = str(summary)
        assert "test-model"            in out
        assert "category_hit_rate"     in out
        assert "grounding_rate"        in out
        assert "avg_llm_calls"         in out
        assert "tool_recall"           in out
        assert "unnecessary_ingest"    in out

    async def test_run_path_eval_pipeline_only(self):
        graph = _make_graph(
            pipeline_state={"sql_rows": [], "response": "", "csv_filename": "Headphones.csv"},
            react_state={
                "sql_rows": [], "response": "",
                "react_messages": _make_tool_msgs("Speakers.csv", True),
                "react_iterations": 3,
            },
        )
        summary = await run_path_eval(
            graph,
            REACT_GOLDEN_SET[:1],
            use_react=False,
            session_factory=None,
        )
        assert summary.pipeline.category_hit_rate == pytest.approx(1.0)
        assert summary.react.category_hit_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_model_compare_eval
# ---------------------------------------------------------------------------

def _make_multi_model_graphs(csv: str = "Headphones.csv"):
    """두 모델에 대한 mock 그래프 딕셔너리를 반환한다."""
    results = {}
    for model in ("llama3.1:8b", "deepseek-r1:8b"):
        pipeline_state = {"sql_rows": [], "response": f"{model} 파이프라인", "csv_filename": csv}
        react_state = {
            "sql_rows": [], "response": f"{model} ReAct",
            "react_messages": _make_tool_msgs(csv, loaded=True),
            "react_iterations": 2,
        }
        ainvoke = AsyncMock(side_effect=lambda state, p=pipeline_state, r=react_state: (
            r if state.get("use_react") else p
        ))
        g = MagicMock()
        g.ainvoke = ainvoke
        results[model] = g
    return results


class TestRunModelCompareEval:
    async def test_returns_model_compare_summary(self):
        graphs = _make_multi_model_graphs()
        result = await run_model_compare_eval(graphs, REACT_GOLDEN_SET[:1], use_react=False)
        assert isinstance(result, ModelCompareSummary)
        assert set(result.summaries.keys()) == {"llama3.1:8b", "deepseek-r1:8b"}

    async def test_all_models_evaluated(self):
        graphs = _make_multi_model_graphs()
        result = await run_model_compare_eval(graphs, REACT_GOLDEN_SET[:2], use_react=False)
        for name, summary in result.summaries.items():
            assert summary.n == 2, f"{name}: n={summary.n} (expected 2)"

    async def test_best_model_returned(self):
        """best_model 은 지정 지표가 가장 높은 모델을 반환한다."""
        graphs = _make_multi_model_graphs()
        result = await run_model_compare_eval(graphs, REACT_GOLDEN_SET[:1], use_react=False)
        best = result.best_model("grounding_rate")
        assert best in {"llama3.1:8b", "deepseek-r1:8b"}

    async def test_best_model_empty_summaries(self):
        result = ModelCompareSummary(summaries={}, use_react=False)
        assert result.best_model() is None

    async def test_use_react_flag_propagated(self):
        graphs = _make_multi_model_graphs()
        result = await run_model_compare_eval(graphs, REACT_GOLDEN_SET[:1], use_react=True)
        assert result.use_react is True

    async def test_str_contains_model_names(self):
        graphs = _make_multi_model_graphs()
        result = await run_model_compare_eval(graphs, REACT_GOLDEN_SET[:1], use_react=False)
        out = str(result)
        assert "llama3.1:8b"    in out
        assert "deepseek-r1:8b" in out
        assert "grounding"      in out


# ---------------------------------------------------------------------------
# react_golden_set 정합성
# ---------------------------------------------------------------------------

class TestReactGoldenSet:
    def test_all_cases_have_required_tools(self):
        for c in REACT_GOLDEN_SET:
            assert "search_category" in c.required_tools,  f"{c.query}: search_category 누락"
            assert "check_db_loaded" in c.required_tools,  f"{c.query}: check_db_loaded 누락"
            assert "query_products"  in c.required_tools,  f"{c.query}: query_products 누락"

    def test_ingest_not_allowed_in_request_path(self):
        for c in REACT_GOLDEN_SET:
            assert "ingest_data" not in c.required_tools, (
                f"{c.query}: ingest_data 는 request path 에서 호출하면 안 됩니다"
            )
            assert "ingest_data" not in c.optional_tools, (
                f"{c.query}: ingest_data 는 optional_tools 에도 없어야 합니다"
            )

    def test_all_cases_have_csv_filename(self):
        for c in REACT_GOLDEN_SET:
            assert c.csv_filename is not None, f"{c.query}: csv_filename 이 None 입니다"

    def test_reference_sql_parseable(self):
        import sqlglot
        for c in REACT_GOLDEN_SET:
            try:
                sqlglot.parse_one(c.reference_sql, dialect="postgres")
            except Exception as e:
                pytest.fail(f"{c.query}: reference_sql 파싱 실패 — {e}")
