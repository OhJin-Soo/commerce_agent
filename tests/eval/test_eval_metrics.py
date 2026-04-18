"""metrics.py 단위 테스트 — DB 불필요."""
from __future__ import annotations

import pytest

from tests.eval.metrics import (
    check_schema,
    component_match,
    execution_accuracy,
    result_f1,
)

REF_SQL = (
    "SELECT id, name, brand, category, price, rating, review_count, source_url"
    " FROM normalized_products"
    " WHERE price <= 3125"
    " AND source_site = 'kaggle/amazon-products/Headphones'"
    " ORDER BY rating DESC NULLS LAST LIMIT 20"
)


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------

class TestCheckSchema:
    def test_valid_sql(self):
        ok, err = check_schema(REF_SQL)
        assert ok is True
        assert err is None

    def test_unknown_table(self):
        ok, err = check_schema("SELECT id FROM products WHERE price < 1000")
        assert ok is False
        assert "products" in err

    def test_unknown_column(self):
        ok, err = check_schema(
            "SELECT id, price_krw FROM normalized_products"
        )
        assert ok is False
        assert "price_krw" in err

    def test_unparseable_sql(self):
        ok, err = check_schema("THIS IS NOT SQL !!!")
        assert ok is False
        assert "parse error" in err


# ---------------------------------------------------------------------------
# execution_accuracy
# ---------------------------------------------------------------------------

class TestExecutionAccuracy:
    def test_identical_rows(self):
        rows = [{"id": 1}, {"id": 2}]
        assert execution_accuracy(rows, rows) == 1.0

    def test_different_rows(self):
        assert execution_accuracy([{"id": 1}], [{"id": 2}]) == 0.0

    def test_empty_both(self):
        assert execution_accuracy([], []) == 1.0

    def test_partial_overlap(self):
        ref = [{"id": 1}, {"id": 2}]
        gen = [{"id": 1}, {"id": 3}]
        assert execution_accuracy(ref, gen) == 0.0


# ---------------------------------------------------------------------------
# result_f1
# ---------------------------------------------------------------------------

class TestResultF1:
    def test_perfect_match(self):
        rows = [{"id": i} for i in range(5)]
        assert result_f1(rows, rows) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert result_f1([{"id": 1}], [{"id": 2}]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        ref = [{"id": 1}, {"id": 2}, {"id": 3}]
        gen = [{"id": 1}, {"id": 2}, {"id": 99}]
        # tp=2, fp=1, fn=1 → p=2/3, r=2/3 → F1=2/3
        assert result_f1(ref, gen) == pytest.approx(2 / 3, abs=1e-6)

    def test_empty_ref(self):
        assert result_f1([], [{"id": 1}]) == pytest.approx(0.0)

    def test_empty_gen(self):
        assert result_f1([{"id": 1}], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# component_match
# ---------------------------------------------------------------------------

class TestComponentMatch:
    def test_identical_sql(self):
        comp = component_match(REF_SQL, REF_SQL)
        assert comp["table_match"] == 1.0
        assert comp["condition_match"] == 1.0

    def test_wrong_table(self):
        bad = REF_SQL.replace("normalized_products", "products")
        comp = component_match(REF_SQL, bad)
        assert comp["table_match"] == 0.0

    def test_wrong_condition(self):
        bad = REF_SQL.replace("price <= 3125", "price <= 9999")
        comp = component_match(REF_SQL, bad)
        assert comp["condition_match"] == 0.0

    def test_unparseable_gen(self):
        comp = component_match(REF_SQL, "NOT SQL")
        assert comp["table_match"] == 0.0
        assert comp["condition_match"] == 0.0
