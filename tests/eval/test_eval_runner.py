"""runner.py 단위 테스트 — skip_db_record=True 로 DB 불필요."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.nodes import _CATEGORY_MAP, _build_sql, _source_site_from
from tests.eval.golden_set import GOLDEN_SET
from tests.eval.runner import run_eval


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rule_based_sql(query: str, exchange_rate: float = 16.0) -> str:
    """현재 규칙 기반 _build_sql 을 그대로 사용하는 generate_sql_fn."""
    q = query.lower()
    for kw, (csv, _) in _CATEGORY_MAP.items():
        if kw in q:
            return _build_sql(query, _source_site_from(csv), exchange_rate=exchange_rate)
    return _build_sql(query, exchange_rate=exchange_rate)


def _make_session_factory(ref_rows: list[dict], gen_rows: list[dict]) -> MagicMock:
    """EXPLAIN 성공 + 두 번의 SELECT 실행을 흉내내는 세션 팩토리."""
    explain_result = MagicMock()

    ref_result = MagicMock()
    ref_result.fetchall.return_value = [MagicMock(_mapping=r) for r in ref_rows]

    gen_result = MagicMock()
    gen_result.fetchall.return_value = [MagicMock(_mapping=r) for r in gen_rows]

    session = AsyncMock()
    # 호출 순서: EXPLAIN → ref SELECT → gen SELECT  (케이스마다 반복)
    session.execute = AsyncMock(side_effect=[
        explain_result, ref_result, gen_result
    ] * len(GOLDEN_SET))

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = ctx
    return factory


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------

class TestRunEval:
    @pytest.mark.parametrize("query,ref_sql", GOLDEN_SET[:3])
    async def test_rule_based_generates_valid_schema(self, query, ref_sql):
        """규칙 기반 SQL 은 항상 schema_invalid_rate=0 이어야 한다."""
        from tests.eval.metrics import check_schema
        sql = _rule_based_sql(query)
        ok, err = check_schema(sql)
        assert ok, f"query={query!r}  sql={sql!r}  err={err}"

    async def test_perfect_match_gives_ex_1(self):
        """ref 와 gen 이 동일한 행을 반환하면 EX=1.0."""
        rows = [{"id": i, "name": f"p{i}"} for i in range(5)]
        factory = _make_session_factory(ref_rows=rows, gen_rows=rows)

        summary = await run_eval(
            session_factory=factory,
            generate_sql_fn=_rule_based_sql,
            model_name="test-perfect",
            skip_db_record=True,
        )
        assert summary.execution_accuracy == pytest.approx(1.0)
        assert summary.result_f1 == pytest.approx(1.0)

    async def test_no_match_gives_ex_0(self):
        """ref 와 gen 이 완전히 다른 행을 반환하면 EX=0.0."""
        ref_rows = [{"id": i} for i in range(3)]
        gen_rows = [{"id": i + 100} for i in range(3)]
        factory = _make_session_factory(ref_rows=ref_rows, gen_rows=gen_rows)

        summary = await run_eval(
            session_factory=factory,
            generate_sql_fn=_rule_based_sql,
            model_name="test-no-match",
            skip_db_record=True,
        )
        assert summary.execution_accuracy == pytest.approx(0.0)

    async def test_schema_invalid_rate_for_bad_generator(self):
        """존재하지 않는 테이블을 생성하는 SQL 생성기 → schema_invalid_rate=1.0."""
        def bad_generator(query: str) -> str:
            return "SELECT * FROM products WHERE price < 1000"

        rows = [{"id": 1}]
        factory = _make_session_factory(ref_rows=rows, gen_rows=rows)

        summary = await run_eval(
            session_factory=factory,
            generate_sql_fn=bad_generator,
            model_name="test-bad-schema",
            skip_db_record=True,
        )
        assert summary.schema_invalid_rate == pytest.approx(1.0)

    async def test_execution_failure_rate_when_explain_fails(self):
        """EXPLAIN 이 실패하면 execution_failure_rate=1.0."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("relation does not exist"))

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=session)
        ctx.__aexit__ = AsyncMock(return_value=False)
        factory = MagicMock()
        factory.return_value = ctx

        summary = await run_eval(
            session_factory=factory,
            generate_sql_fn=_rule_based_sql,
            model_name="test-exec-fail",
            skip_db_record=True,
        )
        assert summary.execution_failure_rate == pytest.approx(1.0)
        assert all(c.error_msg is not None for c in summary.cases)

    async def test_summary_str(self):
        rows = [{"id": 1}]
        factory = _make_session_factory(ref_rows=rows, gen_rows=rows)
        summary = await run_eval(
            session_factory=factory,
            generate_sql_fn=_rule_based_sql,
            model_name="test-str",
            skip_db_record=True,
        )
        out = str(summary)
        assert "execution_accuracy" in out
        assert "result_f1" in out
