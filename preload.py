"""Kaggle product CSVs 를 DB에 미리 적재하는 별도 스크립트.

앱 요청 경로에서는 ingestion 을 수행하지 않는다. 새 DB 또는 새 카테고리를 준비할 때
이 스크립트를 명시적으로 실행한다.

Usage:
    uv run python preload.py
    uv run python preload.py --csv Headphones.csv --csv Speakers.csv
    uv run python preload.py --nrows 1000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from agent.nodes import _CATEGORY_MAP
from db.session import AsyncSessionLocal
from pipeline.preload import preload_csv_files


def _default_csv_files() -> list[str]:
    return sorted({csv for csv, _ in _CATEGORY_MAP.values()})


def _parse_csv_args(values: list[str] | None) -> list[str]:
    if not values:
        return _default_csv_files()
    csvs: list[str] = []
    for value in values:
        csvs.extend(part.strip() for part in value.split(",") if part.strip())
    return csvs


async def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Preload Kaggle product CSVs into PostgreSQL.")
    parser.add_argument(
        "--dataset",
        default=os.getenv("KAGGLE_DATASET_HANDLE", "lokeshparab/amazon-products-dataset"),
        help="Kaggle dataset handle.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        help="CSV filename to preload. Repeat or pass comma-separated values. Defaults to all mapped categories.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=int(os.getenv("KAGGLE_NROWS", "50000")),
        help="Maximum rows per CSV. Use 0 for full file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any CSV fails.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    csv_files = _parse_csv_args(args.csv)
    nrows = None if args.nrows == 0 else args.nrows
    summary = await preload_csv_files(
        session_factory=AsyncSessionLocal,
        dataset_handle=args.dataset,
        csv_filenames=csv_files,
        nrows=nrows,
    )

    print("\n=== Preload Summary ===")
    for result in summary.results:
        detail = f"{result.csv_filename}: {result.status}"
        if result.upserted:
            detail += f" upserted={result.upserted}"
        if result.error:
            detail += f" error={result.error}"
        print(detail)
    print(f"loaded={summary.loaded_count} failed={summary.failed_count}")

    return 1 if args.strict and summary.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
