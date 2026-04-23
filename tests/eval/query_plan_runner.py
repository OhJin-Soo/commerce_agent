"""QueryPlan structured output 평가 실행기."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from tests.eval.metrics import (
    fallback_rate,
    mean,
    percentile,
    query_plan_accuracy,
    query_plan_field_scores,
)
from tests.eval.query_plan_golden_set import QUERY_PLAN_GOLDEN_SET, QueryPlanGoldenCase

logger = logging.getLogger(__name__)


@dataclass
class QueryPlanCaseResult:
    query: str
    expected_plan: dict
    actual_plan: dict | None
    query_plan_error: str | None = None
    used_fallback: bool = False
    accuracy: float = 0.0
    field_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0


@dataclass
class QueryPlanEvalSummary:
    model_name: str
    n: int
    plan_accuracy: float
    field_accuracy: dict[str, float]
    fallback_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_llm_calls: float
    avg_total_tokens: float
    cases: list[QueryPlanCaseResult] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"=== QueryPlanEvalSummary [{self.model_name}] (n={self.n}) ===",
            f"  plan_accuracy   : {self.plan_accuracy:.1%}",
            f"  fallback_rate   : {self.fallback_rate:.1%}",
            f"  avg_latency_ms  : {self.avg_latency_ms:.1f}",
            f"  p95_latency_ms  : {self.p95_latency_ms:.1f}",
            f"  avg_llm_calls   : {self.avg_llm_calls:.2f}",
            f"  avg_total_tokens: {self.avg_total_tokens:.1f}",
            "  field_accuracy:",
        ]
        for field, score in sorted(self.field_accuracy.items()):
            lines.append(f"    {field}: {score:.1%}")
        return "\n".join(lines)


async def _run_case(graph, case: QueryPlanGoldenCase) -> QueryPlanCaseResult:
    t0 = time.perf_counter()
    try:
        state: dict = await graph.ainvoke({"query": case.query, "use_react": False})
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.exception("query plan eval failed: query=%r", case.query)
        return QueryPlanCaseResult(
            query=case.query,
            expected_plan=case.expected_plan,
            actual_plan=None,
            query_plan_error=str(exc),
            used_fallback=True,
            latency_ms=latency_ms,
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    actual = state.get("query_plan")
    field_scores = query_plan_field_scores(case.expected_plan, actual)
    return QueryPlanCaseResult(
        query=case.query,
        expected_plan=case.expected_plan,
        actual_plan=actual,
        query_plan_error=state.get("query_plan_error"),
        used_fallback=bool(state.get("query_plan_error")),
        accuracy=query_plan_accuracy(case.expected_plan, actual),
        field_scores=field_scores,
        latency_ms=latency_ms,
        llm_calls=int(state.get("query_plan_llm_calls", 0) or 0),
        total_tokens=int(state.get("llm_total_tokens", 0) or 0),
    )


async def run_query_plan_eval(
    graph,
    cases: list[QueryPlanGoldenCase] | None = None,
    model_name: str = "unknown",
) -> QueryPlanEvalSummary:
    probes = cases or QUERY_PLAN_GOLDEN_SET
    results = [await _run_case(graph, case) for case in probes]

    field_names = sorted({name for result in results for name in result.field_scores})
    field_accuracy = {
        name: mean([result.field_scores.get(name, 0.0) for result in results])
        for name in field_names
    }
    return QueryPlanEvalSummary(
        model_name=model_name,
        n=len(results),
        plan_accuracy=mean([result.accuracy for result in results]),
        field_accuracy=field_accuracy,
        fallback_rate=fallback_rate([result.used_fallback for result in results]),
        avg_latency_ms=mean([result.latency_ms for result in results]),
        p95_latency_ms=percentile([result.latency_ms for result in results], 95),
        avg_llm_calls=mean([float(result.llm_calls) for result in results]),
        avg_total_tokens=mean([float(result.total_tokens) for result in results]),
        cases=results,
    )
