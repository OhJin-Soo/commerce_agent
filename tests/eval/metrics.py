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


# ---------------------------------------------------------------------------
# ReAct vs 파이프라인 비교 전용 지표
# ---------------------------------------------------------------------------

def grounding_rate(response: str, sql_rows: list[dict]) -> float:
    """sql_rows 상품 중 응답에 이름이 언급된 비율.

    sql_rows 가 비어 있으면 1.0 (측정 불가 → 중립값).
    응답이 비어 있으면 0.0 (상품 데이터를 전혀 반영하지 않음).

    sql_rows 의 최대 10개까지만 비교한다 (generate_response 가 10개 제한을 두므로).
    """
    if not sql_rows:
        return 1.0
    if not response:
        return 0.0
    candidates = sql_rows[:10]
    resp_lower = response.lower()
    mentioned = sum(
        1 for row in candidates
        if (name := row.get("name", "")) and name.lower() in resp_lower
    )
    return mentioned / len(candidates)


def category_hit(expected_csv: str | None, actual_csv: str | None) -> bool:
    """두 경로가 동일한 CSV 파일(카테고리)을 선택했는가.

    둘 다 None 이면 True (카테고리 없는 쿼리를 올바르게 처리함).
    """
    return expected_csv == actual_csv


def tool_sequence_metrics(
    required_tools: list[str],
    optional_tools: list[str],
    actual_tools: list[str],
) -> dict[str, float]:
    """ReAct 도구 호출의 정밀도(precision)와 재현율(recall)을 계산한다.

    precision : 실제 호출한 도구 중 예상(required+optional) 도구 비율
                → 낮으면 불필요한 도구를 호출한 것
    recall    : required 도구 중 실제 호출된 비율
                → 낮으면 필수 도구를 빠뜨린 것

    Args:
        required_tools: 반드시 호출해야 할 도구 목록
        optional_tools: 상황에 따라 호출될 수 있는 도구 목록
        actual_tools:   실제로 호출된 도구 목록 (중복 포함 가능)
    """
    required = set(required_tools)
    allowed = required | set(optional_tools)
    actual = set(actual_tools)          # 중복 제거 후 집합 비교

    recall = len(required & actual) / len(required) if required else 1.0
    precision = len(actual & allowed) / len(actual) if actual else 1.0

    return {"tool_recall": recall, "tool_precision": precision}
