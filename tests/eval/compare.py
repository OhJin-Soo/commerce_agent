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
        latency_ms          - 경로 전체 실행 시간
        llm_calls           - LLM 호출 횟수

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
    category_hit,
    execution_accuracy,
    grounding_rate,
    mean,
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
    tool_sequence: list[str] = field(default_factory=list)  # ReAct 전용
    react_messages: list = field(default_factory=list)      # ReAct 전용 (unnecessary_ingest 판단용)


@dataclass
class PathMetrics:
    """한 경로의 계산된 지표."""
    category_hit: bool = False
    grounding_rate: float = 0.0
    execution_accuracy: float = 0.0     # DB 없으면 0.0 (has_db_eval=False 참고)
    result_f1: float = 0.0              # DB 없으면 0.0
    # ReAct 전용 (파이프라인은 항상 기본값)
    tool_recall: float = 0.0
    tool_precision: float = 0.0
    unnecessary_ingest: bool = False


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
    execution_accuracy: float = 0.0
    result_f1: float = 0.0
    avg_latency_ms: float = 0.0
    avg_llm_calls: float = 0.0
    # ReAct 전용 (파이프라인은 항상 0.0)
    tool_recall: float = 0.0
    tool_precision: float = 0.0
    unnecessary_ingest_rate: float = 0.0


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
            _fmt_row("execution_accuracy (EX)",
                     self.pipeline.execution_accuracy,
                     self.react.execution_accuracy,
                     self.delta_ex),
            _fmt_row("result_f1",
                     self.pipeline.result_f1,
                     self.react.result_f1,
                     self.react.result_f1 - self.pipeline.result_f1),
            "  " + "-" * 63,
            _fmt_row("avg_latency_ms",
                     self.pipeline.avg_latency_ms,
                     self.react.avg_latency_ms,
                     self.delta_latency_ms,
                     fmt=".1f"),
            _fmt_row("avg_llm_calls",
                     self.pipeline.avg_llm_calls,
                     self.react.avg_llm_calls,
                     self.delta_llm_calls,
                     fmt=".2f"),
            "  " + "-" * 63,
            "  [ReAct 전용]",
            _fmt_row("  tool_recall",     None, self.react.tool_recall,             None),
            _fmt_row("  tool_precision",  None, self.react.tool_precision,           None),
            _fmt_row("  unnecessary_ingest_rate", None, self.react.unnecessary_ingest_rate, None),
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

async def _run_path(graph, query: str, use_react: bool) -> PathResult:
    """그래프를 한 경로로 실행하고 PathResult 를 반환한다."""
    t0 = time.perf_counter()
    try:
        state: dict = await graph.ainvoke({"query": query, "use_react": use_react})
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("_run_path error [react=%s, query=%r]: %s", use_react, query, exc)
        return PathResult(latency_ms=latency_ms, error=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000

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
            tool_sequence=_extract_tool_sequence(msgs),
            react_messages=msgs,
        )
    else:
        return PathResult(
            sql_rows=state.get("sql_rows", []),
            response=state.get("response", ""),
            latency_ms=latency_ms,
            error=state.get("error"),
            csv_filename=state.get("csv_filename"),
            llm_calls=1,    # generate_response 만 LLM 호출
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
    ex      = execution_accuracy(ref_rows, path_result.sql_rows) if ref_rows else 0.0
    f1      = result_f1(ref_rows, path_result.sql_rows)          if ref_rows else 0.0

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
            execution_accuracy=ex,
            result_f1=f1,
            tool_recall=seq["tool_recall"],
            tool_precision=seq["tool_precision"],
            unnecessary_ingest=unnecessary,
        )

    return PathMetrics(
        category_hit=cat_hit,
        grounding_rate=ground,
        execution_accuracy=ex,
        result_f1=f1,
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
        execution_accuracy  = mean([m.execution_accuracy for m in pm]),
        result_f1           = mean([m.result_f1 for m in pm]),
        avg_latency_ms      = mean([r.latency_ms for r in pr]),
        avg_llm_calls       = mean([float(r.llm_calls) for r in pr]),
    )
    react_agg = AggregatedMetrics(
        category_hit_rate        = mean([1.0 if m.category_hit else 0.0 for m in rm]),
        grounding_rate           = mean([m.grounding_rate for m in rm]),
        execution_accuracy       = mean([m.execution_accuracy for m in rm]),
        result_f1                = mean([m.result_f1 for m in rm]),
        avg_latency_ms           = mean([r.latency_ms for r in rr]),
        avg_llm_calls            = mean([float(r.llm_calls) for r in rr]),
        tool_recall              = mean([m.tool_recall for m in rm]),
        tool_precision           = mean([m.tool_precision for m in rm]),
        unnecessary_ingest_rate  = mean([1.0 if m.unnecessary_ingest else 0.0 for m in rm]),
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

        pipe_res  = await _run_path(graph, case.query, use_react=False)
        react_res = await _run_path(graph, case.query, use_react=True)

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
                    f"  unnecessary_ingest={agg.unnecessary_ingest_rate:.1%}"
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
