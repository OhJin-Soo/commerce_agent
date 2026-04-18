"""평가 지표 계산 함수 모음.

모든 함수는 순수 함수(side-effect 없음)이며 단독 테스트 가능하다.
"""
from __future__ import annotations

import sqlglot
import sqlglot.expressions as exp

# ---------------------------------------------------------------------------
# 스키마 검증 (골든셋 불필요)
# ---------------------------------------------------------------------------

VALID_SCHEMA: dict[str, set[str]] = {
    "normalized_products": {
        "id", "source_site", "source_product_id", "source_url",
        "name", "brand", "category", "price", "currency",
        "rating", "review_count", "updated_at",
    },
    "product_facts": {
        "id", "product_id", "attribute_name", "attribute_value",
    },
}

_VALID_TABLES = set(VALID_SCHEMA)
_VALID_COLS = {col for cols in VALID_SCHEMA.values() for col in cols}


def check_schema(sql: str) -> tuple[bool, str | None]:
    """생성된 SQL에 존재하지 않는 테이블·컬럼이 있으면 (False, reason) 반환."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as e:
        return False, f"parse error: {e}"

    bad_tables = {t.name.lower() for t in parsed.find_all(exp.Table)} - _VALID_TABLES
    if bad_tables:
        return False, f"unknown tables: {bad_tables}"

    bad_cols = {c.name.lower() for c in parsed.find_all(exp.Column)} - _VALID_COLS
    if bad_cols:
        return False, f"unknown columns: {bad_cols}"

    return True, None


# ---------------------------------------------------------------------------
# 결과 집합 지표 (골든셋 필요)
# ---------------------------------------------------------------------------

def execution_accuracy(ref_rows: list[dict], gen_rows: list[dict]) -> float:
    """결과 집합이 완전히 같으면 1.0, 다르면 0.0.

    행 순서 무관, id 컬럼 기준 집합 비교.
    """
    ref_ids = {r.get("id") for r in ref_rows}
    gen_ids = {r.get("id") for r in gen_rows}
    return 1.0 if ref_ids == gen_ids else 0.0


def result_f1(ref_rows: list[dict], gen_rows: list[dict]) -> float:
    """반환된 행의 id 기준 F1.  부분 일치에 점수를 준다."""
    ref_ids = {r.get("id") for r in ref_rows}
    gen_ids = {r.get("id") for r in gen_rows}

    tp = len(ref_ids & gen_ids)
    fp = len(gen_ids - ref_ids)
    fn = len(ref_ids - gen_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Component Match (진단용)
# ---------------------------------------------------------------------------

def _safe_parse(sql: str):
    try:
        return sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return None


def _f1_sets(ref: set, gen: set) -> float:
    tp = len(ref & gen)
    p  = tp / len(gen) if gen else 0.0
    r  = tp / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def component_match(ref_sql: str, gen_sql: str) -> dict[str, float]:
    """SELECT·WHERE·ORDER BY 절별 일치율을 반환한다.

    Returns:
        {
            "table_match":      1.0 or 0.0,
            "select_col_f1":    0.0‒1.0,
            "condition_match":  1.0 or 0.0,
        }
    """
    ref = _safe_parse(ref_sql)
    gen = _safe_parse(gen_sql)

    if ref is None or gen is None:
        return {"table_match": 0.0, "select_col_f1": 0.0, "condition_match": 0.0}

    ref_tables = {t.name.lower() for t in ref.find_all(exp.Table)}
    gen_tables = {t.name.lower() for t in gen.find_all(exp.Table)}

    ref_cols = {c.name.lower() for c in ref.find_all(exp.Column)}
    gen_cols = {c.name.lower() for c in gen.find_all(exp.Column)}

    # WHERE 절 전체 텍스트를 정규화해 비교
    ref_where = str(ref.find(exp.Where) or "").lower().strip()
    gen_where = str(gen.find(exp.Where) or "").lower().strip()

    return {
        "table_match":     1.0 if ref_tables == gen_tables else 0.0,
        "select_col_f1":   _f1_sets(ref_cols, gen_cols),
        "condition_match": 1.0 if ref_where == gen_where else 0.0,
    }


# ---------------------------------------------------------------------------
# 집계 유틸
# ---------------------------------------------------------------------------

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
