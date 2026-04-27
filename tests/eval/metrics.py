"""평가 지표 계산 함수 모음.

모든 함수는 순수 함수(side-effect 없음)이며 단독 테스트 가능하다.
"""
from __future__ import annotations

import math
import re
from urllib.parse import urlparse

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


def precision_at_k(ref_rows: list[dict], gen_rows: list[dict], k: int) -> float:
    """Top-k 결과 중 reference id 집합에 포함된 비율."""
    if k <= 0:
        return 0.0
    top = gen_rows[:k]
    if not top:
        return 0.0
    ref_ids = {r.get("id") for r in ref_rows}
    hits = sum(1 for row in top if row.get("id") in ref_ids)
    return hits / len(top)


def recall_at_k(ref_rows: list[dict], gen_rows: list[dict], k: int) -> float:
    """Reference id 집합 중 top-k 결과가 회수한 비율."""
    ref_ids = {r.get("id") for r in ref_rows}
    if not ref_ids or k <= 0:
        return 0.0
    top_ids = {row.get("id") for row in gen_rows[:k]}
    return len(ref_ids & top_ids) / len(ref_ids)


def ndcg_at_k(ref_rows: list[dict], gen_rows: list[dict], k: int) -> float:
    """Binary relevance 기준 NDCG@k."""
    if k <= 0:
        return 0.0
    ref_ids = {r.get("id") for r in ref_rows}
    if not ref_ids:
        return 0.0

    def _dcg(rows: list[dict]) -> float:
        score = 0.0
        for idx, row in enumerate(rows[:k], start=1):
            rel = 1.0 if row.get("id") in ref_ids else 0.0
            score += rel / math.log2(idx + 1)
        return score

    ideal_hits = min(len(ref_ids), k)
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return _dcg(gen_rows) / ideal if ideal else 0.0


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


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank style percentile with linear interpolation."""
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def fallback_rate(flags: list[bool]) -> float:
    return mean([1.0 if flag else 0.0 for flag in flags])


# ---------------------------------------------------------------------------
# QueryPlan 평가
# ---------------------------------------------------------------------------

_PLAN_FIELDS = (
    "csv_filename",
    "category",
    "max_price_krw",
    "min_price_krw",
    "min_rating",
    "min_review_count",
    "brand_include",
    "brand_exclude",
    "sort",
    "limit",
    "needs_recommendation",
)


def _normalize_plan_value(value):
    if isinstance(value, list):
        return sorted(str(v).strip().lower() for v in value if str(v).strip())
    if isinstance(value, str):
        return value.strip().lower()
    return value


def query_plan_field_scores(expected: dict, actual: dict | None) -> dict[str, float]:
    """Expected QueryPlan dict 와 actual QueryPlan dict 의 field-level accuracy."""
    actual = actual or {}
    scores: dict[str, float] = {}
    for field in _PLAN_FIELDS:
        if field not in expected:
            continue
        scores[field] = (
            1.0
            if _normalize_plan_value(expected.get(field)) == _normalize_plan_value(actual.get(field))
            else 0.0
        )
    return scores


def query_plan_accuracy(expected: dict, actual: dict | None) -> float:
    scores = query_plan_field_scores(expected, actual)
    return mean(list(scores.values()))


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


def answer_faithfulness(response: str, sql_rows: list[dict], exchange_rate: float = 16.0) -> dict[str, float]:
    """응답이 DB 결과의 상품명/가격/평점을 얼마나 충실히 반영했는지 측정한다.

    문자열 기반의 보수적 지표다. 언급되지 않은 속성은 분모에서 제외한다.
    """
    if not sql_rows:
        return {
            "grounding_rate": 1.0,
            "hallucinated_product_rate": 0.0,
            "price_faithfulness": 1.0,
            "rating_faithfulness": 1.0,
            "attribute_faithfulness": 1.0,
        }
    if not response:
        return {
            "grounding_rate": 0.0,
            "hallucinated_product_rate": 0.0,
            "price_faithfulness": 0.0,
            "rating_faithfulness": 0.0,
            "attribute_faithfulness": 0.0,
        }

    resp = response.lower()
    rows = sql_rows[:10]
    mentioned = [row for row in rows if (name := row.get("name")) and str(name).lower() in resp]
    ground = len(mentioned) / len(rows)

    price_checks: list[float] = []
    rating_checks: list[float] = []
    for row in mentioned:
        price = row.get("price")
        if price is not None:
            try:
                krw = int(float(price) * exchange_rate)
                candidates = {
                    f"{krw:,}".lower(),
                    str(krw).lower(),
                    f"₩{krw:,}".lower(),
                }
                if any(candidate in resp for candidate in candidates):
                    price_checks.append(1.0)
                elif "₩" in resp or "원" in resp:
                    price_checks.append(0.0)
            except (TypeError, ValueError):
                pass

        rating = row.get("rating")
        if rating is not None:
            try:
                rating_text = f"{float(rating):.1f}".rstrip("0").rstrip(".")
                if rating_text in resp:
                    rating_checks.append(1.0)
                elif "평점" in resp:
                    rating_checks.append(0.0)
            except (TypeError, ValueError):
                pass

    price_score = mean(price_checks) if price_checks else 1.0
    rating_score = mean(rating_checks) if rating_checks else 1.0
    return {
        "grounding_rate": ground,
        # 상품명 hallucination 은 현재 상품명 추출기가 없으므로 보수적으로 0.0.
        "hallucinated_product_rate": 0.0,
        "price_faithfulness": price_score,
        "rating_faithfulness": rating_score,
        "attribute_faithfulness": mean([price_score, rating_score]),
    }


def web_grounding_rate(response: str, web_results: list[dict]) -> float:
    """응답이 웹 검색 결과의 source title/url을 얼마나 반영했는지 측정한다.

    exact title match 대신 title token overlap 또는 source URL/domain mention을 허용한다.
    """
    if not web_results:
        return 1.0
    if not response:
        return 0.0
    resp = response.lower()
    hits = 0
    candidates = web_results[:5]

    def _tokens(text: str) -> set[str]:
        parts = re.findall(r"[0-9a-zA-Z가-힣]+", text.lower())
        return {
            token for token in parts
            if len(token) >= 2 and token not in {"리뷰", "후기", "사용자", "평가", "좋은", "대한", "the", "and"}
        }

    for row in candidates:
        title = str(row.get("title", "")).strip().lower()
        url = str(row.get("url", "")).strip().lower()
        hostname = urlparse(url).hostname or ""
        hostname = hostname.removeprefix("www.")
        title_tokens = _tokens(title)
        resp_tokens = _tokens(resp)
        overlap = len(title_tokens & resp_tokens) / len(title_tokens) if title_tokens else 0.0
        if (
            (title and title in resp)
            or (hostname and hostname in resp)
            or (url and url in resp)
            or overlap >= 0.4
        ):
            hits += 1
    return hits / len(candidates)


def react_tool_argument_accuracy(expected: dict, actual: dict | None) -> float:
    """ReAct tool args 와 기대 인자의 field-level accuracy."""
    return query_plan_accuracy(expected, actual)


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
