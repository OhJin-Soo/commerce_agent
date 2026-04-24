"""Web search(Tavily) 평가 실행기."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from tests.eval.metrics import mean, percentile, web_grounding_rate
from tests.eval.web_search_golden_set import WEB_SEARCH_GOLDEN_SET, WebSearchGoldenCase

logger = logging.getLogger(__name__)

WebSearchNode = Callable[[dict], Awaitable[dict]]
ResponseNode = Callable[[dict], Awaitable[dict]]


@dataclass
class WebSearchCaseResult:
    query: str
    min_results: int
    result_count: int
    success: bool
    response: str = ""
    error: str | None = None
    grounding_rate: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0
    web_results: list[dict] = field(default_factory=list)


@dataclass
class WebSearchEvalSummary:
    model_name: str
    n: int
    search_success_rate: float
    avg_source_count: float
    grounding_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_llm_calls: float
    avg_total_tokens: float
    cases: list[WebSearchCaseResult] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"=== WebSearchEvalSummary [{self.model_name}] (n={self.n}) ===",
                f"  search_success_rate: {self.search_success_rate:.1%}",
                f"  avg_source_count   : {self.avg_source_count:.2f}",
                f"  grounding_rate     : {self.grounding_rate:.1%}",
                f"  avg_latency_ms     : {self.avg_latency_ms:.1f}",
                f"  p95_latency_ms     : {self.p95_latency_ms:.1f}",
                f"  avg_llm_calls      : {self.avg_llm_calls:.2f}",
                f"  avg_total_tokens   : {self.avg_total_tokens:.1f}",
            ]
        )


async def _run_case(
    web_search_node: WebSearchNode,
    response_node: ResponseNode,
    case: WebSearchGoldenCase,
) -> WebSearchCaseResult:
    t0 = time.perf_counter()
    search_state = await web_search_node({"query": case.query})
    web_results = search_state.get("web_results") or []
    state_for_response = {"query": case.query, "web_results": web_results}
    response_state = await response_node(state_for_response)
    latency_ms = (time.perf_counter() - t0) * 1000
    success = len(web_results) >= case.min_results and not search_state.get("error")
    response = response_state.get("response", "")
    return WebSearchCaseResult(
        query=case.query,
        min_results=case.min_results,
        result_count=len(web_results),
        success=bool(success),
        response=response,
        error=search_state.get("error") or response_state.get("error"),
        grounding_rate=web_grounding_rate(response, web_results),
        latency_ms=latency_ms,
        llm_calls=int(response_state.get("response_llm_calls", 1) or 1),
        total_tokens=int(response_state.get("llm_total_tokens", 0) or 0),
        web_results=web_results,
    )


async def run_web_search_eval(
    web_search_node: WebSearchNode,
    response_node: ResponseNode,
    cases: list[WebSearchGoldenCase] | None = None,
    model_name: str = "unknown",
) -> WebSearchEvalSummary:
    probes = cases or WEB_SEARCH_GOLDEN_SET
    results = [await _run_case(web_search_node, response_node, case) for case in probes]
    return WebSearchEvalSummary(
        model_name=model_name,
        n=len(results),
        search_success_rate=mean([1.0 if result.success else 0.0 for result in results]),
        avg_source_count=mean([float(result.result_count) for result in results]),
        grounding_rate=mean([result.grounding_rate for result in results]),
        avg_latency_ms=mean([result.latency_ms for result in results]),
        p95_latency_ms=percentile([result.latency_ms for result in results], 95),
        avg_llm_calls=mean([float(result.llm_calls) for result in results]),
        avg_total_tokens=mean([float(result.total_tokens) for result in results]),
        cases=results,
    )
