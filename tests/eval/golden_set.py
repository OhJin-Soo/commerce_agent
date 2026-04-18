"""골든셋 정의.

각 항목은 (자연어 쿼리, 레퍼런스 SQL) 쌍이다.
레퍼런스 SQL은 exchange_rate=16.0 기준으로 _build_sql 이 생성하는 올바른 SQL을 수동 검증해
기록한 것이다. Phase 2 에서 LLM SQL 생성기를 도입할 때 이 셋을 기준으로 EX/F1 을 측정한다.

가격 변환 공식:
    INR = KRW // exchange_rate (16.0)
    예) 5만원  → 50_000 // 16 = 3_125 INR
        30만원 → 300_000 // 16 = 18_750 INR
        50만원 → 500_000 // 16 = 31_250 INR
       100만원 → 1_000_000 // 16 = 62_500 INR
"""

_BASE = (
    "SELECT id, name, brand, category, price, rating, review_count, source_url"
    " FROM normalized_products"
)
_ORDER = "ORDER BY rating DESC NULLS LAST LIMIT 20"

# (query, reference_sql)
GOLDEN_SET: list[tuple[str, str]] = [
    # ── 음향 ────────────────────────────────────────────────────────────────
    (
        "이어폰 5만원 이하",
        f"{_BASE} WHERE price <= 3125"
        " AND source_site = 'kaggle/amazon-products/Headphones'"
        f" {_ORDER}",
    ),
    (
        "헤드폰 30만원 이하",
        f"{_BASE} WHERE price <= 18750"
        " AND source_site = 'kaggle/amazon-products/Headphones'"
        f" {_ORDER}",
    ),
    (
        "스피커 목록",
        f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Speakers'"
        f" {_ORDER}",
    ),
    # ── 영상·TV ─────────────────────────────────────────────────────────────
    (
        "tv 50만원 이하",
        f"{_BASE} WHERE price <= 31250"
        " AND source_site = 'kaggle/amazon-products/Televisions'"
        f" {_ORDER}",
    ),
    # ── 가전 ────────────────────────────────────────────────────────────────
    (
        "냉장고 목록",
        f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Refrigerators'"
        f" {_ORDER}",
    ),
    (
        "세탁기 50만원 이하",
        f"{_BASE} WHERE price <= 31250"
        " AND source_site = 'kaggle/amazon-products/Washing Machines'"
        f" {_ORDER}",
    ),
    # ── 컴퓨터 ──────────────────────────────────────────────────────────────
    (
        "노트북 100만원 이상",
        f"{_BASE} WHERE price >= 62500"
        " AND source_site = 'kaggle/amazon-products/All Electronics'"
        f" {_ORDER}",
    ),
    # ── 패션 ────────────────────────────────────────────────────────────────
    (
        "시계 10만원 이하",
        f"{_BASE} WHERE price <= 6250"
        " AND source_site = 'kaggle/amazon-products/Watches'"
        f" {_ORDER}",
    ),
    (
        "가방 목록",
        f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Bags and Luggage'"
        f" {_ORDER}",
    ),
    # ── 가격 조건 없음 (category only) ──────────────────────────────────────
    (
        "카메라 목록",
        f"{_BASE} WHERE source_site = 'kaggle/amazon-products/Cameras'"
        f" {_ORDER}",
    ),
]
