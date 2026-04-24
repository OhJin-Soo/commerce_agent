from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from db.models import ModelEvalCase, ModelEvalRun
from tests.eval.compare import CompareSummary
from tests.eval.query_plan_runner import QueryPlanEvalSummary
from tests.eval.web_search_runner import WebSearchEvalSummary


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compare_summary_to_records(summary: CompareSummary, path: str) -> tuple[dict, list[dict]]:
    agg = summary.react if path == "react" else summary.pipeline
    run = {
        "model_name": summary.model_name,
        "eval_path": path,
        "probe_count": summary.n,
        "has_db_eval": summary.has_db_eval,
        "category_hit_rate": agg.category_hit_rate,
        "grounding_rate": agg.grounding_rate,
        "answer_faithfulness": agg.answer_faithfulness,
        "execution_accuracy": agg.execution_accuracy,
        "result_f1": agg.result_f1,
        "precision_at_5": agg.precision_at_5,
        "ndcg_at_5": agg.ndcg_at_5,
        "query_plan_accuracy": agg.query_plan_accuracy,
        "fallback_rate": agg.fallback_rate,
        "avg_latency_ms": agg.avg_latency_ms,
        "p95_latency_ms": agg.p95_latency_ms,
        "avg_llm_calls": agg.avg_llm_calls,
        "avg_total_tokens": agg.avg_total_tokens,
        "total_estimated_cost": agg.total_estimated_cost,
        "tool_recall": agg.tool_recall if path == "react" else None,
        "tool_precision": agg.tool_precision if path == "react" else None,
        "unnecessary_ingest_rate": agg.unnecessary_ingest_rate if path == "react" else None,
        "summary_json": _jsonable(summary),
    }

    cases: list[dict] = []
    for case in summary.cases:
        result = case.react if path == "react" else case.pipeline
        metrics = case.react_metrics if path == "react" else case.pipeline_metrics
        cases.append(
            {
                "query": case.query,
                "expected_csv": case.expected_csv,
                "actual_csv": result.csv_filename,
                "response": result.response,
                "error_msg": result.error,
                "category_hit": metrics.category_hit,
                "grounding_rate": metrics.grounding_rate,
                "answer_faithfulness": metrics.answer_faithfulness,
                "execution_accuracy": metrics.execution_accuracy,
                "result_f1": metrics.result_f1,
                "precision_at_5": metrics.precision_at_5,
                "ndcg_at_5": metrics.ndcg_at_5,
                "query_plan_accuracy": metrics.query_plan_accuracy,
                "used_fallback": result.used_fallback,
                "latency_ms": result.latency_ms,
                "llm_calls": result.llm_calls,
                "total_tokens": result.total_tokens,
                "estimated_cost": result.estimated_cost,
                "tool_recall": metrics.tool_recall if path == "react" else None,
                "tool_precision": metrics.tool_precision if path == "react" else None,
                "unnecessary_ingest": metrics.unnecessary_ingest if path == "react" else None,
                "expected_plan": getattr(case, "expected_plan", None),
                "actual_plan": result.query_plan,
                "raw_json": _jsonable(case),
            }
        )
    return run, cases


def query_plan_summary_to_records(summary: QueryPlanEvalSummary) -> tuple[dict, list[dict]]:
    run = {
        "model_name": summary.model_name,
        "eval_path": "query-plan",
        "probe_count": summary.n,
        "has_db_eval": False,
        "query_plan_accuracy": summary.plan_accuracy,
        "fallback_rate": summary.fallback_rate,
        "avg_latency_ms": summary.avg_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "avg_llm_calls": summary.avg_llm_calls,
        "avg_total_tokens": summary.avg_total_tokens,
        "summary_json": _jsonable(summary),
    }
    cases = [
        {
            "query": case.query,
            "expected_csv": case.expected_plan.get("csv_filename"),
            "actual_csv": (case.actual_plan or {}).get("csv_filename"),
            "error_msg": case.query_plan_error,
            "query_plan_accuracy": case.accuracy,
            "used_fallback": case.used_fallback,
            "latency_ms": case.latency_ms,
            "llm_calls": case.llm_calls,
            "total_tokens": case.total_tokens,
            "expected_plan": case.expected_plan,
            "actual_plan": case.actual_plan,
            "raw_json": _jsonable(case),
        }
        for case in summary.cases
    ]
    return run, cases


def web_search_summary_to_records(summary: WebSearchEvalSummary) -> tuple[dict, list[dict]]:
    run = {
        "model_name": summary.model_name,
        "eval_path": "web-search",
        "probe_count": summary.n,
        "has_db_eval": False,
        "search_success_rate": summary.search_success_rate,
        "avg_source_count": summary.avg_source_count,
        "grounding_rate": summary.grounding_rate,
        "avg_latency_ms": summary.avg_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "avg_llm_calls": summary.avg_llm_calls,
        "avg_total_tokens": summary.avg_total_tokens,
        "status": "skipped" if summary.skipped else "completed",
        "skip_reason": summary.skip_reason,
        "summary_json": _jsonable(summary),
    }
    cases = [
        {
            "query": case.query,
            "response": case.response,
            "error_msg": case.error,
            "grounding_rate": case.grounding_rate,
            "latency_ms": case.latency_ms,
            "llm_calls": case.llm_calls,
            "total_tokens": case.total_tokens,
            "raw_json": _jsonable(case),
        }
        for case in summary.cases
    ]
    return run, cases


async def persist_model_eval(
    session_factory: async_sessionmaker,
    run_record: dict,
    case_records: list[dict],
) -> int:
    run = ModelEvalRun(
        **{k: v for k, v in run_record.items() if hasattr(ModelEvalRun, k)}
    )
    run.cases = [
        ModelEvalCase(**{k: v for k, v in case.items() if hasattr(ModelEvalCase, k)})
        for case in case_records
    ]
    async with session_factory() as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return int(run.id)


def export_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_csv(path: Path, run_records: list[dict], case_records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run in run_records:
        rows.append({"record_type": "run", **_flatten_for_csv(run)})
    for case in case_records:
        rows.append({"record_type": "case", **_flatten_for_csv(case)})

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_for_csv(row: dict) -> dict:
    flattened: dict = {}
    for key, value in row.items():
        if key in {"summary_json", "raw_json"}:
            continue
        if isinstance(value, (dict, list)):
            flattened[key] = json.dumps(_jsonable(value), ensure_ascii=False)
        else:
            flattened[key] = value
    return flattened
