"""골든셋 평가 실행기.

사용 예::

    import asyncio
    from db.session import AsyncSessionLocal
    from agent.nodes import _build_sql, _source_site_from
    from tests.eval.runner import run_eval

    async def main():
        # Phase 1: 규칙 기반 SQL 생성기
        def rule_based(query: str) -> str:
            from agent.nodes import _build_sql, _CATEGORY_MAP, _source_site_from
            for kw, (csv, _) in _CATEGORY_MAP.items():
                if kw in query.lower():
                    return _build_sql(query, _source_site_from(csv), exchange_rate=16.0)
            return _build_sql(query, exchange_rate=16.0)

        summary = await run_eval(
            session_factory=AsyncSessionLocal,
            generate_sql_fn=rule_based,
            model_name="rule-based-v1",
        )
        print(summary)

    asyncio.run(main())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from db.models import EvalCase, EvalRun
from tests.eval.golden_set import GOLDEN_SET
from tests.eval.metrics import (
    check_schema,
    component_match,
    execution_accuracy,
    mean,
    result_f1,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 케이스별 결과
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    query: str
    reference_sql: str
    generated_sql: str

    # probe 지표
    is_schema_valid: bool = True
    is_executable: bool = True

    # golden set 지표
    execution_accuracy: float = 0.0
    result_f1: float = 0.0
    table_match: float = 0.0
    condition_match: float = 0.0

    error_msg: str | None = None


@dataclass
class EvalSummary:
    model_name: str
    probe_count: int
    # probe 지표
    schema_invalid_rate: float = 0.0
    execution_failure_rate: float = 0.0
    # golden set 지표
    execution_accuracy: float = 0.0
    result_f1: float = 0.0
    table_match_rate: float = 0.0
    condition_match_rate: float = 0.0

    cases: list[CaseResult] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"=== EvalSummary [{self.model_name}] (n={self.probe_count}) ===",
            f"  schema_invalid_rate    : {self.schema_invalid_rate:.1%}",
            f"  execution_failure_rate : {self.execution_failure_rate:.1%}",
            f"  execution_accuracy     : {self.execution_accuracy:.1%}",
            f"  result_f1              : {self.result_f1:.1%}",
            f"  table_match_rate       : {self.table_match_rate:.1%}",
            f"  condition_match_rate   : {self.condition_match_rate:.1%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 핵심 실행 함수
# ---------------------------------------------------------------------------

async def run_eval(
    session_factory: async_sessionmaker,
    generate_sql_fn: Callable[[str], str],   # query → SQL (동기 함수)
    model_name: str = "unknown",
    golden_set: list[tuple[str, str]] | None = None,
    skip_db_record: bool = False,
) -> EvalSummary:
    """골든셋 전체를 평가하고 결과를 DB에 기록한 뒤 EvalSummary를 반환한다.

    Args:
        session_factory:   AsyncSessionLocal
        generate_sql_fn:   (query: str) → SQL 문자열을 반환하는 함수.
                           LLM 기반이면 await 가 필요하므로 별도 래핑 필요.
        model_name:        식별용 이름 (예: "rule-based-v1", "llama3.1:8b")
        golden_set:        None 이면 GOLDEN_SET 사용
        skip_db_record:    True 이면 DB 기록 생략 (단위 테스트용)
    """
    probes = golden_set or GOLDEN_SET
    cases: list[CaseResult] = []

    for query, ref_sql in probes:
        result = CaseResult(query=query, reference_sql=ref_sql, generated_sql="")

        # 1. SQL 생성
        try:
            result.generated_sql = generate_sql_fn(query)
        except Exception as e:
            result.is_schema_valid = False
            result.is_executable = False
            result.error_msg = f"generation error: {e}"
            cases.append(result)
            continue

        # 2. 스키마 검증
        schema_ok, schema_err = check_schema(result.generated_sql)
        result.is_schema_valid = schema_ok
        if not schema_ok:
            result.error_msg = schema_err

        # 3. 실행 가능 여부 (EXPLAIN) + 결과 집합 비교
        async with session_factory() as session:
            # EXPLAIN 으로 실행 가능 여부 확인
            try:
                await session.execute(text(f"EXPLAIN {result.generated_sql}"))
                result.is_executable = True
            except Exception as e:
                result.is_executable = False
                result.error_msg = str(e)

            # 실행 가능할 때만 결과 집합 비교
            if result.is_executable:
                try:
                    ref_rows = [
                        dict(r._mapping)
                        for r in (await session.execute(text(ref_sql))).fetchall()
                    ]
                    gen_rows = [
                        dict(r._mapping)
                        for r in (await session.execute(text(result.generated_sql))).fetchall()
                    ]
                    result.execution_accuracy = execution_accuracy(ref_rows, gen_rows)
                    result.result_f1 = result_f1(ref_rows, gen_rows)
                except Exception as e:
                    result.error_msg = str(e)

        # 4. Component match (SQL 파싱 기반, DB 불필요)
        comp = component_match(ref_sql, result.generated_sql)
        result.table_match = comp["table_match"]
        result.condition_match = comp["condition_match"]

        cases.append(result)
        logger.debug(
            "eval case: query=%r  EX=%.2f  F1=%.2f  table=%.1f  cond=%.1f",
            query,
            result.execution_accuracy,
            result.result_f1,
            result.table_match,
            result.condition_match,
        )

    # 5. 집계
    n = len(cases)
    summary = EvalSummary(
        model_name=model_name,
        probe_count=n,
        schema_invalid_rate=mean([0.0 if c.is_schema_valid else 1.0 for c in cases]),
        execution_failure_rate=mean([0.0 if c.is_executable else 1.0 for c in cases]),
        execution_accuracy=mean([c.execution_accuracy for c in cases]),
        result_f1=mean([c.result_f1 for c in cases]),
        table_match_rate=mean([c.table_match for c in cases]),
        condition_match_rate=mean([c.condition_match for c in cases]),
        cases=cases,
    )

    # 6. DB 기록
    if not skip_db_record:
        await _persist(session_factory, summary)

    return summary


async def _persist(session_factory: async_sessionmaker, summary: EvalSummary) -> None:
    """EvalRun + EvalCase 를 DB에 저장한다."""
    run = EvalRun(
        model_name=summary.model_name,
        probe_count=summary.probe_count,
        schema_invalid_rate=summary.schema_invalid_rate,
        execution_failure_rate=summary.execution_failure_rate,
        execution_accuracy=summary.execution_accuracy,
        result_f1=summary.result_f1,
        table_match_rate=summary.table_match_rate,
        condition_match_rate=summary.condition_match_rate,
        cases=[
            EvalCase(
                query=c.query,
                generated_sql=c.generated_sql,
                is_schema_valid=c.is_schema_valid,
                is_executable=c.is_executable,
                reference_sql=c.reference_sql,
                execution_accuracy=c.execution_accuracy,
                result_f1=c.result_f1,
                table_match=c.table_match,
                condition_match=c.condition_match,
                error_msg=c.error_msg,
            )
            for c in summary.cases
        ],
    )
    async with session_factory() as session:
        session.add(run)
        await session.commit()
    logger.info("eval run persisted: model=%s  EX=%.2f", summary.model_name, summary.execution_accuracy)
