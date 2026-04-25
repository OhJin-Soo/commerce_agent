"""모델 평가 CLI.

Examples:
    uv run python eval_models.py --models llama3.1:8b,deepseek-r1:8b,gemma4:26b --path pipeline
    uv run python eval_models.py --models llama3.1:8b,deepseek-r1:8b,gemma4:26b --path all
    uv run python eval_models.py --models llama3.1:8b,deepseek-r1:8b,gemma4:26b --paths pipeline,query-plan
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
from uuid import uuid4

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from agent import GraphDeps, build_graph
from agent.nodes import make_generate_query_plan_node, make_generate_response_node, make_web_search_node
from db.currency import fetch_inr_to_krw
from db.session import AsyncSessionLocal
from tests.eval.compare import run_path_eval
from tests.eval.model_report import (
    compare_summary_to_records,
    export_csv,
    export_json,
    persist_model_evals,
    query_plan_summary_to_records,
    react_web_search_summary_to_records,
    web_search_summary_to_records,
)
from tests.eval.query_plan_runner import run_query_plan_eval
from tests.eval.react_golden_set import REACT_GOLDEN_SET
from tests.eval.react_web_search_runner import run_react_web_search_eval
from tests.eval.web_search_runner import make_skipped_web_search_eval
from tests.eval.web_search_runner import run_web_search_eval


def _parse_models(raw: str) -> list[str]:
    models = [part.strip() for part in raw.split(",") if part.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


_ALL_PATHS = ["pipeline", "react", "query-plan", "web-search", "react-web-search"]


def _parse_paths(path: str, paths: str | None) -> list[str]:
    raw_values = paths if paths is not None else path
    if raw_values == "all":
        return list(_ALL_PATHS)

    result = [part.strip() for part in raw_values.split(",") if part.strip()]
    if "all" in result:
        return list(_ALL_PATHS)

    invalid = sorted(set(result) - set(_ALL_PATHS))
    if invalid:
        raise ValueError(f"invalid path(s): {invalid}. Allowed: {_ALL_PATHS} or all")
    if not result:
        raise ValueError("at least one path is required")
    return result


def _path_label(paths: list[str]) -> str:
    return "all" if paths == _ALL_PATHS else "_".join(paths)


def _default_report_paths(output_dir: Path, path_label: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"model_eval_{path_label}_{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".csv")


async def _build_graph(model: str):
    dataset_handle = os.getenv("KAGGLE_DATASET_HANDLE", "lokeshparab/amazon-products-dataset")
    ingest_nrows = int(os.getenv("KAGGLE_NROWS", "50000"))
    tavily_api_key = os.getenv("TAVILY_API_KEY") or None
    exchange_rate = await fetch_inr_to_krw()
    llm = ChatOllama(model=model)
    deps = GraphDeps(
        session_factory=AsyncSessionLocal,
        llm=llm,
        dataset_handle=dataset_handle,
        ingest_nrows=ingest_nrows,
        exchange_rate=exchange_rate,
        tavily_api_key=tavily_api_key,
    )
    return {
        "graph": build_graph(deps),
        "query_plan_node": make_generate_query_plan_node(llm),
        "web_search_node": make_web_search_node(tavily_api_key),
        "response_node": make_generate_response_node(llm, exchange_rate=exchange_rate),
        "tavily_enabled": bool(tavily_api_key),
    }


async def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate commerce-agent models.")
    parser.add_argument(
        "--models",
        default=os.getenv("OLLAMA_MODELS", os.getenv("OLLAMA_MODEL", "llama3.1:8b,deepseek-r1:8b,gemma4:26b")),
        help="Comma-separated Ollama model names.",
    )
    parser.add_argument(
        "--path",
        choices=["pipeline", "react", "query-plan", "web-search", "react-web-search", "all"],
        default="pipeline",
        help="Evaluation path to run. Use 'all' for every path.",
    )
    parser.add_argument(
        "--paths",
        help="Comma-separated paths to run, e.g. pipeline,react,query-plan. Overrides --path.",
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
    paths = _parse_paths(args.path, args.paths)
    path_label = _path_label(paths)
    json_path, csv_path = _default_report_paths(Path(args.output_dir), path_label)
    if args.json_path:
        json_path = Path(args.json_path)
    if args.csv_path:
        csv_path = Path(args.csv_path)

    payload: dict = {
        "paths": paths,
        "models": models,
        "runs": [],
    }
    is_batch_run = args.path == "all" or (args.paths is not None and "all" in args.paths.split(","))
    batch_id = str(uuid4()) if is_batch_run else None
    if batch_id:
        payload["batch_id"] = batch_id
    if batch_id:
        logging.info("Starting batch eval batch_id=%s", batch_id)
    run_records: list[dict] = []
    case_records: list[dict] = []
    pending_persist: list[tuple[dict, list[dict]]] = []

    try:
        for model in models:
            logging.info("Building graph for model=%s", model)
            runtime = await _build_graph(model)
            graph = runtime["graph"]

            for eval_path in paths:
                logging.info("Evaluating model=%s path=%s", model, eval_path)

                if eval_path == "query-plan":
                    summary = await run_query_plan_eval(runtime["query_plan_node"], model_name=model)
                    run_record, cases = query_plan_summary_to_records(summary)
                elif eval_path == "web-search":
                    if runtime["tavily_enabled"]:
                        summary = await run_web_search_eval(
                            runtime["web_search_node"],
                            runtime["response_node"],
                            model_name=model,
                        )
                    else:
                        logging.warning("Skipping web-search eval for model=%s: missing TAVILY_API_KEY", model)
                        summary = make_skipped_web_search_eval(model, "missing_tavily_api_key")
                    run_record, cases = web_search_summary_to_records(summary)
                elif eval_path == "react-web-search":
                    summary = await run_react_web_search_eval(
                        graph,
                        model_name=model,
                    )
                    run_record, cases = react_web_search_summary_to_records(summary)
                else:
                    summary = await run_path_eval(
                        graph=graph,
                        cases=REACT_GOLDEN_SET,
                        use_react=eval_path == "react",
                        session_factory=None if args.no_db_eval else AsyncSessionLocal,
                        model_name=model,
                        input_cost_per_1k=args.input_cost_per_1k,
                        output_cost_per_1k=args.output_cost_per_1k,
                    )
                    run_record, cases = compare_summary_to_records(summary, eval_path)

                if batch_id:
                    run_record["batch_id"] = batch_id

                if not args.no_save and run_record.get("status") != "skipped":
                    pending_persist.append((dict(run_record), list(cases)))

                run_records.append(run_record)
                case_records.extend({**case, "model_name": model, "eval_path": eval_path} for case in cases)
                payload["runs"].append(
                    {
                        "model_name": model,
                        "eval_path": eval_path,
                        "run": run_record,
                        "cases": cases,
                    }
                )
                print(summary)
                print()
    except KeyboardInterrupt:
        logging.warning("Evaluation interrupted. No DB rows or report files were written.")
        return 130
    except Exception:
        logging.exception("Evaluation failed. No DB rows or report files were written.")
        return 1

    if not args.no_save and pending_persist:
        run_ids = await persist_model_evals(AsyncSessionLocal, pending_persist)
        run_id_iter = iter(run_ids)
        for entry in payload["runs"]:
            run = entry["run"]
            if run.get("status") == "skipped":
                continue
            run_id = next(run_id_iter)
            run["id"] = run_id
            logging.info(
                "Persisted model eval run id=%s model=%s path=%s",
                run_id,
                entry["model_name"],
                entry["eval_path"],
            )
        run_record_iter = iter(run_ids)
        for run_record in run_records:
            if run_record.get("status") == "skipped":
                continue
            run_record["id"] = next(run_record_iter)

    export_json(json_path, payload)
    export_csv(csv_path, run_records, case_records)
    print(f"JSON report: {json_path}")
    print(f"CSV report : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
