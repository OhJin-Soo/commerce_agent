"""모델 평가 CLI.

Examples:
    uv run python eval_models.py --models llama3.1:8b,deepseek-r1:8b --path pipeline
    uv run python eval_models.py --models llama3.1:8b --path react --no-db-eval
    uv run python eval_models.py --models llama3.1:8b --path query-plan --no-save
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from agent import GraphDeps, build_graph
from db.currency import fetch_inr_to_krw
from db.session import AsyncSessionLocal
from tests.eval.compare import run_path_eval
from tests.eval.model_report import (
    compare_summary_to_records,
    export_csv,
    export_json,
    persist_model_eval,
    query_plan_summary_to_records,
)
from tests.eval.query_plan_runner import run_query_plan_eval
from tests.eval.react_golden_set import REACT_GOLDEN_SET


def _parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def _default_report_paths(output_dir: Path, path: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"model_eval_{path}_{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".csv")


async def _build_graph(model: str):
    dataset_handle = os.getenv("KAGGLE_DATASET_HANDLE", "lokeshparab/amazon-products-dataset")
    ingest_nrows = int(os.getenv("KAGGLE_NROWS", "50000"))
    tavily_api_key = os.getenv("TAVILY_API_KEY") or None
    exchange_rate = await fetch_inr_to_krw()
    deps = GraphDeps(
        session_factory=AsyncSessionLocal,
        llm=ChatOllama(model=model),
        dataset_handle=dataset_handle,
        ingest_nrows=ingest_nrows,
        exchange_rate=exchange_rate,
        tavily_api_key=tavily_api_key,
    )
    return build_graph(deps)


async def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate commerce-agent models.")
    parser.add_argument(
        "--models",
        default=os.getenv("OLLAMA_MODELS", os.getenv("OLLAMA_MODEL", "llama3.1:8b")),
        help="Comma-separated Ollama model names.",
    )
    parser.add_argument(
        "--path",
        choices=["pipeline", "react", "query-plan"],
        default="pipeline",
        help="Evaluation path to run.",
    )
    parser.add_argument(
        "--no-db-eval",
        action="store_true",
        help="Skip DB-backed EX/F1/precision@k/NDCG metrics for pipeline/react.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist results into model_eval_runs/model_eval_cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_reports",
        help="Directory for default JSON/CSV reports.",
    )
    parser.add_argument("--json", dest="json_path", help="JSON report path.")
    parser.add_argument("--csv", dest="csv_path", help="CSV report path.")
    parser.add_argument(
        "--input-cost-per-1k",
        type=float,
        default=0.0,
        help="Input token cost per 1K tokens.",
    )
    parser.add_argument(
        "--output-cost-per-1k",
        type=float,
        default=0.0,
        help="Output token cost per 1K tokens.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = _parse_models(args.models)
    json_path, csv_path = _default_report_paths(Path(args.output_dir), args.path)
    if args.json_path:
        json_path = Path(args.json_path)
    if args.csv_path:
        csv_path = Path(args.csv_path)

    payload: dict = {
        "path": args.path,
        "models": models,
        "runs": [],
    }
    run_records: list[dict] = []
    case_records: list[dict] = []

    for model in models:
        logging.info("Evaluating model=%s path=%s", model, args.path)
        graph = await _build_graph(model)

        if args.path == "query-plan":
            summary = await run_query_plan_eval(graph, model_name=model)
            run_record, cases = query_plan_summary_to_records(summary)
        else:
            summary = await run_path_eval(
                graph=graph,
                cases=REACT_GOLDEN_SET,
                use_react=args.path == "react",
                session_factory=None if args.no_db_eval else AsyncSessionLocal,
                model_name=model,
                input_cost_per_1k=args.input_cost_per_1k,
                output_cost_per_1k=args.output_cost_per_1k,
            )
            run_record, cases = compare_summary_to_records(summary, args.path)

        if not args.no_save:
            run_id = await persist_model_eval(AsyncSessionLocal, run_record, cases)
            run_record = {**run_record, "id": run_id}
            logging.info("Persisted model eval run id=%s model=%s path=%s", run_id, model, args.path)

        run_records.append(run_record)
        case_records.extend({**case, "model_name": model, "eval_path": args.path} for case in cases)
        payload["runs"].append(
            {
                "model_name": model,
                "eval_path": args.path,
                "run": run_record,
                "cases": cases,
            }
        )
        print(summary)
        print()

    export_json(json_path, payload)
    export_csv(csv_path, run_records, case_records)
    print(f"JSON report: {json_path}")
    print(f"CSV report : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
