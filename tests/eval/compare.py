"""ReAct vs 파이프라인 비교 평가 실행기.

사용 예::

    import asyncio
    from db.session import AsyncSessionLocal
    from agent.graph import build_graph, GraphDeps
    from langchain_ollama import ChatOllama
    from tests.eval.compare import run_compare_eval
    from tests.eval.react_golden_set import REACT_GOLDEN_SET

    async def main():
        deps = GraphDeps(
            session_factory=AsyncSessionLocal,
            llm=ChatOllama(model="llama3.1:8b"),
        )
        graph = build_graph(deps)

        summary = await run_compare_eval(
            graph=graph,
            cases=REACT_GOLDEN_SET,
            session_factory=AsyncSessionLocal,  # None 이면 EX/F1 생략
            model_name="llama3.1:8b",
        )
        print(summary)

    asyncio.run(main())

측정 지표::

    공통 (두 경로 모두)
        category_hit        - 올바른 CSV/카테고리를 선택했는가
        execution_accuracy  - sql_rows ID 집합 완전 일치 (EX)  ← DB 필요
        result_f1           - sql_rows ID 집합 F1              ← DB 필요
        grounding_rate      - LLM 응답이 sql_rows 상품을 반영했는가
        answer_faithfulness - 응답의 가격/평점 반영 정확도
        precision@k / ndcg@k - top-k retrieval/ranking 품질
        latency_ms          - 경로 전체 실행 시간
        p95_latency_ms      - 경로 p95 지연 시간
        llm_calls           - LLM 호출 횟수
        token/cost          - usage metadata 기반 토큰/비용
        fallback_rate       - structured QueryPlan 실패 후 rule fallback 비율

    ReAct 전용
        tool_recall         - required 도구를 빠짐없이 호출했는가
        tool_precision      - 불필요한 도구를 호출하지 않았는가
        unnecessary_ingest  - loaded=True 인데 ingest_data 를 호출했는가
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from tests.eval.metrics import (
    answer_faithfulness,
    category_hit,
    fallback_rate,
    execution_accuracy,
    grounding_rate,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    query_plan_accuracy,
    react_tool_argument_accuracy,
    result_f1,
    tool_sequence_metrics,
)
from tests.eval.react_golden_set import ReactGoldenCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 내부 유틸 — react_messages 파싱
# ---------------------------------------------------------------------------

def _extract_tool_sequence(react_messages: list) -> list[str]:
    """react_messages 에서 실행된 도구 이름을 호출 순서대로 추출한다.

    ToolMessage 판별: tool_call_id 속성과 name 속성이 모두 있는 메시지.
    """
    result = []
    for m in react_messages:
        name = getattr(m, "name", None)
        if name and getattr(m, "tool_call_id", None) is not None:
            result.append(name)
    return result


def _extract_tool_args(react_messages: list, tool_name: str) -> dict | None:
    """AIMessage.tool_calls 에서 특정 도구의 마지막 args 를 추출한다."""
    for m in reversed(react_messages):
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in reversed(tool_calls):
            if tc.get("name") == tool_name:
                args = tc.get("args")
                return args if isinstance(args, dict) else None
    return None


def _extract_search_web_queries(react_messages: list) -> list[str]:
    queries: list[str] = []
    for m in react_messages:
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            if tc.get("name") != "search_web":
                continue
            args = tc.get("args")
            if isinstance(args, dict):
                query = args.get("query")
                if isinstance(query, str) and query.strip():
                    queries.append(query.strip())
    return queries


def _extract_search_web_results(react_messages: list) -> list[dict]:
    results: list[dict] = []
    for m in react_messages:
        if getattr(m, "name", None) != "search_web":
            continue
        try:
            data = json.loads(getattr(m, "content", "{}"))
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
        items = data.get("results")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    results.append(item)
    return results


def _classify_error_status(error: str | None, react_messages: list | None = None) -> str | None:
    """에러를 tool 지원/인프라/일반 실행 실패로 분류한다."""
    haystacks = []
    if error:
        haystacks.append(error)
    for message in react_messages or []:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            lower = content.lower()
            if any(marker in lower for marker in ('"error"', '"message"', "error", "failed", "exception", "tavily_api_key")):
                haystacks.append(content)
    text_blob = " ".join(haystacks).lower()
    if not text_blob.strip():
        return None
    if "does not support tools" in text_blob or "tool support" in text_blob:
        return "tool_unsupported"
    infra_markers = (
        "tavily_api_key",
        "connection refused",
        "timeout",
        "timed out",
        "name or service not known",
        "failed to connect",
        "could not connect",
        "temporary failure",
        "network",
        "dns",
        "api key",
        "authentication",
        "permission denied",
    )
    if any(marker in text_blob for marker in infra_markers):
        return "infra_failure"
    return "execution_failure"


def _extract_react_csv(react_messages: list) -> str | None:
    """search_category 도구의 응답에서 csv_filename 을 추출한다."""
    for m in react_messages:
        if getattr(m, "name", None) == "search_category":
            try:
                data = json.loads(getattr(m, "content", "{}"))
                return data.get("csv_filename")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
    return None


def _was_ingest_unnecessary(react_messages: list) -> bool:
    """check_db_loaded 가 loaded=True 를 반환한 뒤 ingest_data 를 호출했는가.

    DB 에 이미 데이터가 있는데 불필요한 ingestion 을 수행하면 True.
    """
    db_was_loaded = False
    for m in react_messages:
        name = getattr(m, "name", None)
        if name == "check_db_loaded":
            try:
                data = json.loads(getattr(m, "content", "{}"))
                db_was_loaded = bool(data.get("loaded"))
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        elif name == "ingest_data" and db_was_loaded:
            return True
    return False


# ---------------------------------------------------------------------------
# 결과 자료구조
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    """graph.ainvoke 의 raw 결과 + 경로별 측정값."""
    sql_rows: list[dict] = field(default_factory=list)
    response: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    csv_filename: str | None = None     # 경로가 선택한 카테고리 CSV
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    query_plan: dict | None = None
    query_plan_error: str | None = None
    used_fallback: bool = False
    error_status: str | None = None
    tool_arguments: dict[str, dict] = field(default_factory=dict)
    web_results: list[dict] = field(default_factory=list)
    search_web_queries: list[str] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)  # ReAct 전용
    react_messages: list = field(default_factory=list)      # ReAct 전용 (unnecessary_ingest 판단용)


@dataclass
class PathMetrics:
    """한 경로의 계산된 지표."""
    category_hit: bool = False
    grounding_rate: float = 0.0
    answer_faithfulness: float = 0.0
    execution_accuracy: float = 0.0     # DB 없으면 0.0 (has_db_eval=False 참고)
    result_f1: float = 0.0              # DB 없으면 0.0
    precision_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    query_plan_accuracy: float = 0.0
    tool_argument_accuracy: float = 0.0
    search_web_recall: float = 0.0
    search_web_grounding_rate: float = 0.0
    # ReAct 전용 (파이프라인은 항상 기본값)
    tool_recall: float = 0.0
    tool_precision: float = 0.0
    unnecessary_ingest: bool = False
    tool_unsupported: bool = False
    infra_failure: bool = False


@dataclass
class CompareCase:
    """케이스별 두 경로의 결과와 지표."""
    query: str
    expected_csv: str | None
    pipeline: PathResult
    react: PathResult
    pipeline_metrics: PathMetrics
    react_metrics: PathMetrics


@dataclass
class AggregatedMetrics:
    """케이스 전체 집계."""
    category_hit_rate: float = 0.0
    grounding_rate: float = 0.0
    answer_faithfulness: float = 0.0
    execution_accuracy: float = 0.0
    result_f1: float = 0.0
    precision_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    query_plan_accuracy: float = 0.0
    tool_argument_accuracy: float = 0.0
    search_web_recall: float = 0.0
    search_web_grounding_rate: float = 0.0
    fallback_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_llm_calls: float = 0.0
    avg_total_tokens: float = 0.0
    total_estimated_cost: float = 0.0
    # ReAct 전용 (파이프라인은 항상 0.0)
    tool_recall: float = 0.0
    tool_precision: float = 0.0
    unnecessary_ingest_rate: float = 0.0
    tool_unsupported_rate: float = 0.0
    infra_failure_rate: float = 0.0


@dataclass
class CompareSummary:
    """전체 비교 결과."""
    model_name: str
    n: int
    has_db_eval: bool               # EX/F1 계산 여부 (session_factory 제공 시 True)
    pipeline: AggregatedMetrics
    react: AggregatedMetrics
    cases: list[CompareCase] = field(default_factory=list)

    @property
    def delta_category_hit(self) -> float:
        return self.react.category_hit_rate - self.pipeline.category_hit_rate

    @property
    def delta_grounding(self) -> float:
        return self.react.grounding_rate - self.pipeline.grounding_rate

    @property
    def delta_ex(self) -> float:
        return self.react.execution_accuracy - self.pipeline.execution_accuracy

    @property
    def delta_latency_ms(self) -> float:
        """양수 = ReAct 가 더 느림."""
        return self.react.avg_latency_ms - self.pipeline.avg_latency_ms

    @property
    def delta_llm_calls(self) -> float:
        """양수 = ReAct 가 LLM 을 더 많이 호출함."""
        return self.react.avg_llm_calls - self.pipeline.avg_llm_calls

    def __str__(self) -> str:
        db_note = "" if self.has_db_eval else "  ※ EX/F1: DB 없어 미측정"
        lines = [
            f"=== CompareSummary [{self.model_name}] (n={self.n}){db_note} ===",
            "",
            f"  {'지표':<26} {'pipeline':>10} {'react':>10} {'Δ(react-pipe)':>14}",
            "  " + "-" * 63,
            _fmt_row("category_hit_rate",
                     self.pipeline.category_hit_rate,
                     self.react.category_hit_rate,
                     self.delta_category_hit),
            _fmt_row("grounding_rate",
                     self.pipeline.grounding_rate,
                     self.react.grounding_rate,
                     self.delta_grounding),
            _fmt_row("answer_faithfulness",
                     self.pipeline.answer_faithfulness,
                     self.react.answer_faithfulness,
                     self.react.answer_faithfulness - self.pipeline.answer_faithfulness),
            _fmt_row("execution_accuracy (EX)",
                     self.pipeline.execution_accuracy,
                     self.react.execution_accuracy,
                     self.delta_ex),
            _fmt_row("result_f1",
                     self.pipeline.result_f1,
                     self.react.result_f1,
                     self.react.result_f1 - self.pipeline.result_f1),
            _fmt_row("precision@5",
                     self.pipeline.precision_at_5,
                     self.react.precision_at_5,
                     self.react.precision_at_5 - self.pipeline.precision_at_5),
            _fmt_row("ndcg@5",
                     self.pipeline.ndcg_at_5,
                     self.react.ndcg_at_5,
                     self.react.ndcg_at_5 - self.pipeline.ndcg_at_5),
            "  " + "-" * 63,
            _fmt_row("avg_latency_ms",
                     self.pipeline.avg_latency_ms,
                     self.react.avg_latency_ms,
                     self.delta_latency_ms,
                     fmt=".1f"),
            _fmt_row("p95_latency_ms",
                     self.pipeline.p95_latency_ms,
                     self.react.p95_latency_ms,
                     self.react.p95_latency_ms - self.pipeline.p95_latency_ms,
                     fmt=".1f"),
            _fmt_row("avg_llm_calls",
                     self.pipeline.avg_llm_calls,
                     self.react.avg_llm_calls,
                     self.delta_llm_calls,
                     fmt=".2f"),
            _fmt_row("avg_total_tokens",
                     self.pipeline.avg_total_tokens,
                     self.react.avg_total_tokens,
                     self.react.avg_total_tokens - self.pipeline.avg_total_tokens,
                     fmt=".1f"),
            _fmt_row("total_estimated_cost",
                     self.pipeline.total_estimated_cost,
                     self.react.total_estimated_cost,
                     self.react.total_estimated_cost - self.pipeline.total_estimated_cost,
                     fmt=".4f"),
            _fmt_row("fallback_rate",
                     self.pipeline.fallback_rate,
                     self.react.fallback_rate,
                     self.react.fallback_rate - self.pipeline.fallback_rate),
            _fmt_row("query_plan_accuracy",
                     self.pipeline.query_plan_accuracy,
                     self.react.query_plan_accuracy,
                     self.react.query_plan_accuracy - self.pipeline.query_plan_accuracy),
            _fmt_row("tool_argument_accuracy",
                     self.pipeline.tool_argument_accuracy,
                     self.react.tool_argument_accuracy,
                     self.react.tool_argument_accuracy - self.pipeline.tool_argument_accuracy),
            "  " + "-" * 63,
            "  [ReAct 전용]",
            _fmt_row("  tool_recall",     None, self.react.tool_recall,             None),
            _fmt_row("  tool_precision",  None, self.react.tool_precision,           None),
            _fmt_row("  search_web_recall", None, self.react.search_web_recall, None),
            _fmt_row("  search_web_grounding", None, self.react.search_web_grounding_rate, None),
            _fmt_row("  unnecessary_ingest_rate", None, self.react.unnecessary_ingest_rate, None),
            _fmt_row("  tool_unsupported_rate", None, self.react.tool_unsupported_rate, None),
            _fmt_row("  infra_failure_rate", None, self.react.infra_failure_rate, None),
        ]
        return "\n".join(lines)


def _fmt_row(
    label: str,
    pipe_val,
    react_val,
    delta,
    fmt: str = ".1%",
) -> str:
    p = f"{pipe_val:{fmt}}"  if pipe_val  is not None else "         -"
    r = f"{react_val:{fmt}}" if react_val is not None else "         -"
    d = f"{delta:+{fmt}}"    if delta     is not None else "             -"
    return f"  {label:<26} {p:>10} {r:>10} {d:>14}"


# ---------------------------------------------------------------------------
# 경로별 실행
# ---------------------------------------------------------------------------

def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    return (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k


async def _run_path(
    graph,
    query: str,
    use_react: bool,
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
) -> PathResult:
    """그래프를 한 경로로 실행하고 PathResult 를 반환한다."""
    t0 = time.perf_counter()
    try:
        state: dict = await graph.ainvoke({"query": query, "use_react": use_react})
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("_run_path error [react=%s, query=%r]: %s", use_react, query, exc)
        return PathResult(latency_ms=latency_ms, error=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    input_tokens = int(state.get("llm_input_tokens", 0) or 0)
    output_tokens = int(state.get("llm_output_tokens", 0) or 0)
    total_tokens = int(state.get("llm_total_tokens", input_tokens + output_tokens) or 0)
    estimated_cost = _estimate_cost(
        input_tokens,
        output_tokens,
        input_cost_per_1k,
        output_cost_per_1k,
    )

    if use_react:
        msgs = state.get("react_messages") or []
        iterations = state.get("react_iterations", 0)
        return PathResult(
            sql_rows=state.get("sql_rows", []),
            response=state.get("response", ""),
            latency_ms=latency_ms,
            error=state.get("error"),
            csv_filename=_extract_react_csv(msgs),
            llm_calls=iterations + 1,   # react_act 횟수 + 최종 react_reason 1번
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            error_status=_classify_error_status(state.get("error"), msgs),
            tool_arguments={
                "query_products": _extract_tool_args(msgs, "query_products") or {},
                "search_web": _extract_tool_args(msgs, "search_web") or {},
            },
            web_results=_extract_search_web_results(msgs),
            search_web_queries=_extract_search_web_queries(msgs),
            tool_sequence=_extract_tool_sequence(msgs),
            react_messages=msgs,
        )
    else:
        query_plan_calls = int(state.get("query_plan_llm_calls", 0) or 0)
        response_calls = int(state.get("response_llm_calls", 1) or 1)
        query_plan_error = state.get("query_plan_error")
        return PathResult(
            sql_rows=state.get("sql_rows", []),
            response=state.get("response", ""),
            latency_ms=latency_ms,
            error=state.get("error"),
            csv_filename=state.get("csv_filename"),
            llm_calls=query_plan_calls + response_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            query_plan=state.get("query_plan"),
            query_plan_error=query_plan_error,
            used_fallback=bool(query_plan_error),
        )


async def _fetch_ref_rows(
    session_factory: async_sessionmaker, reference_sql: str
) -> list[dict]:
    try:
        async with session_factory() as session:
            result = await session.execute(text(reference_sql))
            return [dict(r._mapping) for r in result.fetchall()]
    except Exception as exc:
        logger.warning("reference SQL 실행 실패: %s", exc)
        return []


def _compute_metrics(
    path_result: PathResult,
    ref_rows: list[dict],
    case: ReactGoldenCase,
    is_react: bool,
) -> PathMetrics:
    cat_hit = category_hit(case.csv_filename, path_result.csv_filename)
    ground  = grounding_rate(path_result.response, path_result.sql_rows)
    faith   = answer_faithfulness(path_result.response, path_result.sql_rows)
    ex      = execution_accuracy(ref_rows, path_result.sql_rows) if ref_rows else 0.0
    f1      = result_f1(ref_rows, path_result.sql_rows)          if ref_rows else 0.0
    p5      = precision_at_k(ref_rows, path_result.sql_rows, 5)  if ref_rows else 0.0
    ndcg5   = ndcg_at_k(ref_rows, path_result.sql_rows, 5)       if ref_rows else 0.0
    expected_plan = getattr(case, "expected_plan", None)
    plan_acc = query_plan_accuracy(expected_plan, path_result.query_plan) if expected_plan else 0.0
    expected_query_products_args = getattr(case, "expected_query_products_args", None)
    tool_arg_acc = (
        react_tool_argument_accuracy(expected_query_products_args, path_result.tool_arguments.get("query_products"))
        if expected_query_products_args and is_react
        else 0.0
    )
    expected_search_web = bool(getattr(case, "expects_search_web", False))
    search_web_called = "search_web" in path_result.tool_sequence
    search_web_recall = 1.0 if (not expected_search_web or search_web_called) else 0.0
    search_web_ground = 0.0
    if is_react and search_web_called:
        from tests.eval.metrics import web_grounding_rate
        search_web_ground = web_grounding_rate(path_result.response, path_result.web_results)

    if is_react:
        seq = tool_sequence_metrics(
            case.required_tools,
            case.optional_tools,
            path_result.tool_sequence,
        )
        unnecessary = _was_ingest_unnecessary(path_result.react_messages)
        return PathMetrics(
            category_hit=cat_hit,
            grounding_rate=ground,
            answer_faithfulness=faith["attribute_faithfulness"],
            execution_accuracy=ex,
            result_f1=f1,
            precision_at_5=p5,
            ndcg_at_5=ndcg5,
            query_plan_accuracy=plan_acc,
            tool_argument_accuracy=tool_arg_acc,
            search_web_recall=search_web_recall,
            search_web_grounding_rate=search_web_ground,
            tool_recall=seq["tool_recall"],
            tool_precision=seq["tool_precision"],
            unnecessary_ingest=unnecessary,
            tool_unsupported=path_result.error_status == "tool_unsupported",
            infra_failure=path_result.error_status == "infra_failure",
        )

    return PathMetrics(
        category_hit=cat_hit,
        grounding_rate=ground,
        answer_faithfulness=faith["attribute_faithfulness"],
        execution_accuracy=ex,
        result_f1=f1,
        precision_at_5=p5,
        ndcg_at_5=ndcg5,
        query_plan_accuracy=plan_acc,
    )


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def _aggregate(
    compare_cases: list[CompareCase],
) -> tuple[AggregatedMetrics, AggregatedMetrics]:
    """(pipeline_agg, react_agg) 를 반환한다."""
    pm = [c.pipeline_metrics for c in compare_cases]
    rm = [c.react_metrics    for c in compare_cases]
    pr = [c.pipeline         for c in compare_cases]
    rr = [c.react            for c in compare_cases]

    pipe_agg = AggregatedMetrics(
        category_hit_rate   = mean([1.0 if m.category_hit else 0.0 for m in pm]),
        grounding_rate      = mean([m.grounding_rate for m in pm]),
        answer_faithfulness = mean([m.answer_faithfulness for m in pm]),
        execution_accuracy  = mean([m.execution_accuracy for m in pm]),
        result_f1           = mean([m.result_f1 for m in pm]),
        precision_at_5      = mean([m.precision_at_5 for m in pm]),
        ndcg_at_5           = mean([m.ndcg_at_5 for m in pm]),
        query_plan_accuracy = mean([m.query_plan_accuracy for m in pm]),
        tool_argument_accuracy = mean([m.tool_argument_accuracy for m in pm]),
        search_web_recall   = mean([m.search_web_recall for m in pm]),
        search_web_grounding_rate = mean([m.search_web_grounding_rate for m in pm]),
        fallback_rate       = fallback_rate([r.used_fallback for r in pr]),
        avg_latency_ms      = mean([r.latency_ms for r in pr]),
        p95_latency_ms      = percentile([r.latency_ms for r in pr], 95),
        avg_llm_calls       = mean([float(r.llm_calls) for r in pr]),
        avg_total_tokens    = mean([float(r.total_tokens) for r in pr]),
        total_estimated_cost = sum(r.estimated_cost for r in pr),
    )
    react_agg = AggregatedMetrics(
        category_hit_rate        = mean([1.0 if m.category_hit else 0.0 for m in rm]),
        grounding_rate           = mean([m.grounding_rate for m in rm]),
        answer_faithfulness      = mean([m.answer_faithfulness for m in rm]),
        execution_accuracy       = mean([m.execution_accuracy for m in rm]),
        result_f1                = mean([m.result_f1 for m in rm]),
        precision_at_5           = mean([m.precision_at_5 for m in rm]),
        ndcg_at_5                = mean([m.ndcg_at_5 for m in rm]),
        query_plan_accuracy      = mean([m.query_plan_accuracy for m in rm]),
        tool_argument_accuracy   = mean([m.tool_argument_accuracy for m in rm]),
        search_web_recall        = mean([m.search_web_recall for m in rm]),
        search_web_grounding_rate = mean([m.search_web_grounding_rate for m in rm]),
        fallback_rate            = fallback_rate([r.used_fallback for r in rr]),
        avg_latency_ms           = mean([r.latency_ms for r in rr]),
        p95_latency_ms           = percentile([r.latency_ms for r in rr], 95),
        avg_llm_calls            = mean([float(r.llm_calls) for r in rr]),
        avg_total_tokens         = mean([float(r.total_tokens) for r in rr]),
        total_estimated_cost     = sum(r.estimated_cost for r in rr),
        tool_recall              = mean([m.tool_recall for m in rm]),
        tool_precision           = mean([m.tool_precision for m in rm]),
        unnecessary_ingest_rate  = mean([1.0 if m.unnecessary_ingest else 0.0 for m in rm]),
        tool_unsupported_rate    = mean([1.0 if m.tool_unsupported else 0.0 for m in rm]),
        infra_failure_rate       = mean([1.0 if m.infra_failure else 0.0 for m in rm]),
    )
    return pipe_agg, react_agg


# ---------------------------------------------------------------------------
# 메인 실행 함수
# ---------------------------------------------------------------------------

async def run_compare_eval(
    graph,
    cases: list[ReactGoldenCase],
    session_factory: async_sessionmaker | None = None,
    model_name: str = "unknown",
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
) -> CompareSummary:
    """두 경로를 나란히 실행하고 비교 지표를 반환한다.

    Args:
        graph:           build_graph(deps) 로 컴파일된 StateGraph
        cases:           REACT_GOLDEN_SET (또는 그 부분집합)
        session_factory: 제공하면 reference SQL 을 실행해 EX/F1 계산, None 이면 생략
        model_name:      결과 식별용 이름 (예: "llama3.1:8b")
    """
    compare_cases: list[CompareCase] = []

    for case in cases:
        logger.info("compare eval: %r", case.query)

        pipe_res  = await _run_path(
            graph,
            case.query,
            use_react=False,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )
        react_res = await _run_path(
            graph,
            case.query,
            use_react=True,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )

        ref_rows = (
            await _fetch_ref_rows(session_factory, case.reference_sql)
            if session_factory else []
        )

        pipe_metrics  = _compute_metrics(pipe_res,  ref_rows, case, is_react=False)
        react_metrics = _compute_metrics(react_res, ref_rows, case, is_react=True)

        compare_cases.append(CompareCase(
            query=case.query,
            expected_csv=case.csv_filename,
            pipeline=pipe_res,
            react=react_res,
            pipeline_metrics=pipe_metrics,
            react_metrics=react_metrics,
        ))
        logger.debug(
            "  pipeline: cat_hit=%s  grounding=%.2f  llm_calls=%d  latency=%.0fms",
            pipe_metrics.category_hit, pipe_metrics.grounding_rate,
            pipe_res.llm_calls, pipe_res.latency_ms,
        )
        logger.debug(
            "  react:    cat_hit=%s  grounding=%.2f  llm_calls=%d  latency=%.0fms"
            "  tool_recall=%.2f  tool_precision=%.2f  unnecessary_ingest=%s",
            react_metrics.category_hit, react_metrics.grounding_rate,
            react_res.llm_calls, react_res.latency_ms,
            react_metrics.tool_recall, react_metrics.tool_precision,
            react_metrics.unnecessary_ingest,
        )

    pipe_agg, react_agg = _aggregate(compare_cases)

    return CompareSummary(
        model_name=model_name,
        n=len(compare_cases),
        has_db_eval=session_factory is not None,
        pipeline=pipe_agg,
        react=react_agg,
        cases=compare_cases,
    )


async def run_path_eval(
    graph,
    cases: list[ReactGoldenCase],
    use_react: bool,
    session_factory: async_sessionmaker | None = None,
    model_name: str = "unknown",
    input_cost_per_1k: float = 0.0,
    output_cost_per_1k: float = 0.0,
) -> CompareSummary:
    """pipeline 또는 ReAct 중 한 경로만 실행해 CompareSummary 형태로 반환한다."""
    compare_cases: list[CompareCase] = []

    for case in cases:
        path_res = await _run_path(
            graph,
            case.query,
            use_react=use_react,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
        )
        ref_rows = (
            await _fetch_ref_rows(session_factory, case.reference_sql)
            if session_factory else []
        )
        metrics = _compute_metrics(path_res, ref_rows, case, is_react=use_react)

        if use_react:
            compare_cases.append(
                CompareCase(
                    query=case.query,
                    expected_csv=case.csv_filename,
                    pipeline=PathResult(),
                    react=path_res,
                    pipeline_metrics=PathMetrics(),
                    react_metrics=metrics,
                )
            )
        else:
            compare_cases.append(
                CompareCase(
                    query=case.query,
                    expected_csv=case.csv_filename,
                    pipeline=path_res,
                    react=PathResult(),
                    pipeline_metrics=metrics,
                    react_metrics=PathMetrics(),
                )
            )

    pipe_agg, react_agg = _aggregate(compare_cases)
    return CompareSummary(
        model_name=model_name,
        n=len(compare_cases),
        has_db_eval=session_factory is not None,
        pipeline=pipe_agg,
        react=react_agg,
        cases=compare_cases,
    )


# ---------------------------------------------------------------------------
# 모델 비교 전용 자료구조 및 실행 함수
# ---------------------------------------------------------------------------

@dataclass
class ModelCompareSummary:
    """여러 모델의 비교 평가 결과를 묶는 컨테이너.

    summaries: model_name → CompareSummary 매핑
    use_react:  True 면 ReAct 경로, False 면 파이프라인 경로로 평가
    """
    summaries: dict[str, CompareSummary] = field(default_factory=dict)
    use_react: bool = False

    def best_model(self, metric: str = "grounding_rate") -> str | None:
        """지정한 지표가 가장 높은 모델 이름을 반환한다.

        metric: 'grounding_rate' | 'category_hit_rate' | 'execution_accuracy' |
                'result_f1' | 'tool_recall' | 'tool_precision'
        """
        if not self.summaries:
            return None

        def _get(summary: CompareSummary) -> float:
            agg = summary.react if self.use_react else summary.pipeline
            return getattr(agg, metric, 0.0)

        return max(self.summaries, key=lambda name: _get(self.summaries[name]))

    def __str__(self) -> str:
        path_label = "react" if self.use_react else "pipeline"
        lines = [f"=== ModelCompareSummary (path={path_label}) ===", ""]
        for name, summary in self.summaries.items():
            agg = summary.react if self.use_react else summary.pipeline
            db_note = "" if summary.has_db_eval else "  ※ EX/F1 미측정"
            lines.append(f"  [{name}]{db_note}")
            lines.append(
                f"    category_hit={agg.category_hit_rate:.1%}"
                f"  grounding={agg.grounding_rate:.1%}"
                f"  EX={agg.execution_accuracy:.1%}"
                f"  F1={agg.result_f1:.1%}"
                f"  latency={agg.avg_latency_ms:.0f}ms"
                f"  llm_calls={agg.avg_llm_calls:.1f}"
            )
            if self.use_react:
                lines.append(
                    f"    tool_recall={agg.tool_recall:.1%}"
                    f"  tool_precision={agg.tool_precision:.1%}"
                    f"  tool_arg_acc={agg.tool_argument_accuracy:.1%}"
                    f"  search_web_recall={agg.search_web_recall:.1%}"
                    f"  unnecessary_ingest={agg.unnecessary_ingest_rate:.1%}"
                    f"  tool_unsupported={agg.tool_unsupported_rate:.1%}"
                    f"  infra_failure={agg.infra_failure_rate:.1%}"
                )
            lines.append("")
        best = self.best_model()
        if best:
            lines.append(f"  ★ Best (grounding_rate): {best}")
        return "\n".join(lines)


async def run_model_compare_eval(
    graphs: dict,
    cases: list[ReactGoldenCase],
    use_react: bool = False,
    session_factory: async_sessionmaker | None = None,
) -> ModelCompareSummary:
    """여러 모델을 동일한 케이스로 평가하고 ModelCompareSummary 를 반환한다.

    Args:
        graphs:          {model_name: compiled_graph} 딕셔너리
        cases:           REACT_GOLDEN_SET (또는 그 부분집합)
        use_react:       True 면 ReAct 경로로만 평가, False 면 파이프라인 경로로만 평가
        session_factory: 제공하면 reference SQL 을 실행해 EX/F1 계산, None 이면 생략

    각 모델에 대해 run_compare_eval 을 순차 실행하므로, 케이스 수가 많으면
    시간이 오래 걸릴 수 있다.
    """
    summaries: dict[str, CompareSummary] = {}

    for model_name, graph in graphs.items():
        logger.info("model_compare_eval: 모델=%s  use_react=%s", model_name, use_react)
        summary = await run_compare_eval(
            graph=graph,
            cases=cases,
            session_factory=session_factory,
            model_name=model_name,
        )
        summaries[model_name] = summary

    return ModelCompareSummary(summaries=summaries, use_react=use_react)
