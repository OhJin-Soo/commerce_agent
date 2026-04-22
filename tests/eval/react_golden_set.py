"""ReAct 비교 평가용 골든셋.

각 케이스는 GOLDEN_SET 의 reference_sql 을 재사용하고,
ReAct 전용 필드(expected_tools, optional_tools)를 추가한다.

required_tools : 어떤 상황에서도 반드시 호출해야 하는 도구
optional_tools : 상황(DB 적재 여부)에 따라 호출될 수 있는 도구

정상 도구 순서::

    search_category → check_db_loaded → [ingest_data?] → query_products
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tests.eval.golden_set import GOLDEN_SET

_BASE = (
    "SELECT id, name, brand, category, price, rating, review_count, source_url"
    " FROM normalized_products"
)
_ORDER = "ORDER BY rating DESC NULLS LAST LIMIT 20"


@dataclass
class ReactGoldenCase:
    query: str
    csv_filename: str | None          # 정답 CSV 파일명 (category_hit 판단 기준)
    required_tools: list[str]         # 반드시 호출해야 할 도구
    optional_tools: list[str]         # 호출해도/안 해도 되는 도구
    reference_sql: str                # EX/F1 계산용 기준 SQL


# GOLDEN_SET 과 동일한 쿼리 순서 유지
REACT_GOLDEN_SET: list[ReactGoldenCase] = [
    # ── 음향 ────────────────────────────────────────────────────────────────
    ReactGoldenCase(
        query="이어폰 5만원 이하",
        csv_filename="Headphones.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE price <= 3125"
            " AND source_site = 'kaggle/amazon-products/Headphones'"
            f" {_ORDER}"
        ),
    ),
    ReactGoldenCase(
        query="헤드폰 30만원 이하",
        csv_filename="Headphones.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE price <= 18750"
            " AND source_site = 'kaggle/amazon-products/Headphones'"
            f" {_ORDER}"
        ),
    ),
    ReactGoldenCase(
        query="스피커 목록",
        csv_filename="Speakers.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Speakers'"
            f" {_ORDER}"
        ),
    ),
    # ── 영상·TV ─────────────────────────────────────────────────────────────
    ReactGoldenCase(
        query="tv 50만원 이하",
        csv_filename="Televisions.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE price <= 31250"
            " AND source_site = 'kaggle/amazon-products/Televisions'"
            f" {_ORDER}"
        ),
    ),
    # ── 가전 ────────────────────────────────────────────────────────────────
    ReactGoldenCase(
        query="냉장고 목록",
        csv_filename="Refrigerators.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Refrigerators'"
            f" {_ORDER}"
        ),
    ),
    ReactGoldenCase(
        query="세탁기 50만원 이하",
        csv_filename="Washing Machines.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE price <= 31250"
            " AND source_site = 'kaggle/amazon-products/Washing Machines'"
            f" {_ORDER}"
        ),
    ),
    # ── 컴퓨터 ──────────────────────────────────────────────────────────────
    ReactGoldenCase(
        query="노트북 100만원 이상",
        csv_filename="All Electronics.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE price >= 62500"
            " AND source_site = 'kaggle/amazon-products/All Electronics'"
            f" {_ORDER}"
        ),
    ),
    # ── 패션 ────────────────────────────────────────────────────────────────
    ReactGoldenCase(
        query="시계 10만원 이하",
        csv_filename="Watches.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE price <= 6250"
            " AND source_site = 'kaggle/amazon-products/Watches'"
            f" {_ORDER}"
        ),
    ),
    ReactGoldenCase(
        query="가방 목록",
        csv_filename="Bags and Luggage.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Bags and Luggage'"
            f" {_ORDER}"
        ),
    ),
    # ── 카메라 ──────────────────────────────────────────────────────────────
    ReactGoldenCase(
        query="카메라 목록",
        csv_filename="Cameras.csv",
        required_tools=["search_category", "check_db_loaded", "query_products"],
        optional_tools=["ingest_data"],
        reference_sql=(
            f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Cameras'"
            f" {_ORDER}"
        ),
    ),
]

assert len(REACT_GOLDEN_SET) == len(GOLDEN_SET), (
    "REACT_GOLDEN_SET 과 GOLDEN_SET 의 케이스 수가 다릅니다. 동기화가 필요합니다."
)
