"""ReAct search_web 도구 전용 평가기."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from observability.langsmith import build_run_config
from tests.eval.compare import (
    _classify_error_status,
    _extract_search_web_queries,
    _extract_search_web_results,
    _extract_tool_sequence,
)
from tests.eval.metrics import mean, percentile, web_grounding_rate
from tests.eval.react_web_search_golden_set import (
    REACT_WEB_SEARCH_GOLDEN_SET,
    ReactWebSearchGoldenCase,
)

logger = logging.getLogger(__name__)


@dataclass
class ReactWebSearchCaseResult:
    query: str
    search_web_called: bool
    search_query: str | None
    result_count: int
    response: str
    error: str | None = None
    error_status: str | None = None
    grounding_rate: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0


@dataclass
class ReactWebSearchEvalSummary:
    model_name: str
    n: int
    search_web_recall: float
    search_success_rate: float
    avg_source_count: float
    grounding_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_llm_calls: float
    avg_total_tokens: float
    tool_unsupported_rate: float
    infra_failure_rate: float
    cases: list[ReactWebSearchCaseResult] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"=== ReactWebSearchEvalSummary [{self.model_name}] (n={self.n}) ===",
                f"  search_web_recall  : {self.search_web_recall:.1%}",
                f"  search_success_rate: {self.search_success_rate:.1%}",
                f"  avg_source_count   : {self.avg_source_count:.2f}",
                f"  grounding_rate     : {self.grounding_rate:.1%}",
                f"  tool_unsupported   : {self.tool_unsupported_rate:.1%}",
                f"  infra_failure      : {self.infra_failure_rate:.1%}",
                f"  avg_latency_ms     : {self.avg_latency_ms:.1f}",
                f"  p95_latency_ms     : {self.p95_latency_ms:.1f}",
                f"  avg_llm_calls      : {self.avg_llm_calls:.2f}",
                f"  avg_total_tokens   : {self.avg_total_tokens:.1f}",
            ]
        )


async def _run_case(graph, case: ReactWebSearchGoldenCase) -> ReactWebSearchCaseResult:
    t0 = time.perf_counter()
    try:
        run_config = build_run_config(
            "eval.react_web_search",
            tags=["eval", "react", "web-search"],
            metadata={"query": case.query, "use_react": True},
        )
        state: dict = await graph.ainvoke({"query": case.query, "use_react": True}, config=run_config)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return ReactWebSearchCaseResult(
            query=case.query,
            search_web_called=False,
            search_query=None,
            result_count=0,
            response="",
            error=str(exc),
            error_status=_classify_error_status(str(exc), None),
            latency_ms=latency_ms,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    messages = state.get("react_messages") or []
    tool_sequence = _extract_tool_sequence(messages)
    search_queries = _extract_search_web_queries(messages)
    web_results = _extract_search_web_results(messages)
    error = state.get("error")
    error_status = _classify_error_status(error, messages)
    search_web_called = "search_web" in tool_sequence
    response = state.get("response", "")
    search_success = search_web_called and len(web_results) >= case.min_results and error_status is None
    return ReactWebSearchCaseResult(
        query=case.query,
        search_web_called=search_web_called,
        search_query=search_queries[-1] if search_queries else None,
        result_count=len(web_results),
        response=response,
        error=error,
        error_status=error_status,
        grounding_rate=web_grounding_rate(response, web_results) if search_success else 0.0,
        latency_ms=latency_ms,
        llm_calls=int((state.get("react_iterations", 0) or 0) + 1),
        total_tokens=int(state.get("llm_total_tokens", 0) or 0),
    )


async def run_react_web_search_eval(
    graph,
    cases: list[ReactWebSearchGoldenCase] | None = None,
    model_name: str = "unknown",
) -> ReactWebSearchEvalSummary:
    probes = cases or REACT_WEB_SEARCH_GOLDEN_SET
    results = [await _run_case(graph, case) for case in probes]
    return ReactWebSearchEvalSummary(
        model_name=model_name,
        n=len(results),
        search_web_recall=mean([1.0 if r.search_web_called else 0.0 for r in results]),
        search_success_rate=mean([
            1.0 if r.search_web_called and r.result_count >= 1 and r.error_status is None else 0.0
            for r in results
        ]),
        avg_source_count=mean([float(r.result_count) for r in results]),
        grounding_rate=mean([r.grounding_rate for r in results]),
        avg_latency_ms=mean([r.latency_ms for r in results]),
        p95_latency_ms=percentile([r.latency_ms for r in results], 95),
        avg_llm_calls=mean([float(r.llm_calls) for r in results]),
        avg_total_tokens=mean([float(r.total_tokens) for r in results]),
        tool_unsupported_rate=mean([1.0 if r.error_status == "tool_unsupported" else 0.0 for r in results]),
        infra_failure_rate=mean([1.0 if r.error_status == "infra_failure" else 0.0 for r in results]),
        cases=results,
    )
