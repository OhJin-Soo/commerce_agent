"""metrics.py 단위 테스트 — DB 불필요."""
from __future__ import annotations

import pytest

from tests.eval.metrics import (
    answer_faithfulness,
    check_schema,
    component_match,
    execution_accuracy,
    fallback_rate,
    ndcg_at_k,
    percentile,
    precision_at_k,
    query_plan_accuracy,
    query_plan_field_scores,
    recall_at_k,
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
# top-k retrieval metrics
# ---------------------------------------------------------------------------

class TestTopKMetrics:
    def test_precision_recall_at_k(self):
        ref = [{"id": 1}, {"id": 2}, {"id": 3}]
        gen = [{"id": 1}, {"id": 99}, {"id": 2}]
        assert precision_at_k(ref, gen, 2) == pytest.approx(0.5)
        assert recall_at_k(ref, gen, 2) == pytest.approx(1 / 3)

    def test_ndcg_at_k_perfect(self):
        ref = [{"id": 1}, {"id": 2}]
        gen = [{"id": 1}, {"id": 2}]
        assert ndcg_at_k(ref, gen, 2) == pytest.approx(1.0)

    def test_ndcg_at_k_partial(self):
        ref = [{"id": 1}, {"id": 2}]
        gen = [{"id": 99}, {"id": 1}]
        assert 0.0 < ndcg_at_k(ref, gen, 2) < 1.0


# ---------------------------------------------------------------------------
# QueryPlan metrics
# ---------------------------------------------------------------------------

class TestQueryPlanMetrics:
    def test_field_scores_and_accuracy(self):
        expected = {
            "csv_filename": "Headphones.csv",
            "max_price_krw": 50000,
            "brand_exclude": ["Apple"],
        }
        actual = {
            "csv_filename": "headphones.csv",
            "max_price_krw": 40000,
            "brand_exclude": ["apple"],
        }
        scores = query_plan_field_scores(expected, actual)
        assert scores["csv_filename"] == pytest.approx(1.0)
        assert scores["max_price_krw"] == pytest.approx(0.0)
        assert scores["brand_exclude"] == pytest.approx(1.0)
        assert query_plan_accuracy(expected, actual) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 운영 지표 helpers
# ---------------------------------------------------------------------------

class TestOperationalMetrics:
    def test_fallback_rate(self):
        assert fallback_rate([True, False, True]) == pytest.approx(2 / 3)

    def test_percentile(self):
        assert percentile([10, 20, 30], 95) == pytest.approx(29.0)


# ---------------------------------------------------------------------------
# answer faithfulness
# ---------------------------------------------------------------------------

class TestAnswerFaithfulness:
    def test_price_and_rating_match(self):
        rows = [{"name": "Sony WH-1000XM5", "price": 3125, "rating": 4.5}]
        response = "Sony WH-1000XM5는 ₩50,000이고 평점 4.5입니다."
        scores = answer_faithfulness(response, rows, exchange_rate=16.0)
        assert scores["grounding_rate"] == pytest.approx(1.0)
        assert scores["attribute_faithfulness"] == pytest.approx(1.0)

    def test_price_mismatch_when_price_claim_present(self):
        rows = [{"name": "Sony WH-1000XM5", "price": 3125, "rating": 4.5}]
        response = "Sony WH-1000XM5는 ₩40,000이고 평점 4.5입니다."
        scores = answer_faithfulness(response, rows, exchange_rate=16.0)
        assert scores["price_faithfulness"] == pytest.approx(0.0)


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
