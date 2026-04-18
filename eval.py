"""품질 평가 실행 스크립트.

    uv run python eval.py
    uv run python eval.py --model rule-based-v1
    uv run python eval.py --no-save          # DB 기록 생략
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from agent.nodes import _CATEGORY_MAP, _build_sql, _source_site_from
from db.session import AsyncSessionLocal
from tests.eval.runner import run_eval

logging.basicConfig(level=logging.WARNING)


def _make_generate_sql_fn(exchange_rate: float = 16.0):
    """현재 규칙 기반 SQL 생성기를 run_eval 에 주입할 수 있는 형태로 반환."""
    def generate(query: str) -> str:
        q = query.lower()
        for kw, (csv, _) in _CATEGORY_MAP.items():
            if kw in q:
                return _build_sql(query, _source_site_from(csv), exchange_rate=exchange_rate)
        return _build_sql(query, exchange_rate=exchange_rate)
    return generate


async def main(model_name: str, skip_db_record: bool) -> None:
    summary = await run_eval(
        session_factory=AsyncSessionLocal,
        generate_sql_fn=_make_generate_sql_fn(),
        model_name=model_name,
        skip_db_record=skip_db_record,
    )
    print(summary)

    if not skip_db_record:
        print(f"\n✓ eval_runs 에 저장됨 (model={model_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL 품질 평가")
    parser.add_argument("--model", default="rule-based-v1", help="모델/생성기 식별 이름")
    parser.add_argument("--no-save", action="store_true", help="DB 기록 생략")
    args = parser.parse_args()

    asyncio.run(main(model_name=args.model, skip_db_record=args.no_save))
