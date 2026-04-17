from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from db.models import NormalizedProduct
from pipeline.base import Filter

logger = logging.getLogger(__name__)

# 가격 문자열에서 숫자만 추출 (통화 기호·쉼표·공백 제거)
_PRICE_STRIP = re.compile(r"[^\d.]")


@dataclass
class ColumnMapping:
    """CSV 컬럼 이름 → NormalizedProduct 필드 매핑 규칙.

    각 필드는 ``str | list[str]`` 형태로 지정한다. 리스트면 앞에서부터 순서대로 시도해
    처음으로 존재하는 컬럼을 사용한다. ``None`` 이면 해당 필드는 채우지 않는다.

    Example::

        ColumnMapping(
            name=["product_name", "title"],
            price="discounted_price",
            source_url_template="https://www.amazon.in/dp/{source_product_id}",
        )
    """

    # --- 필수 ---
    name: str | list[str] = field(
        default_factory=lambda: ["product_name", "name", "title", "item_name"]
    )

    # --- 식별자 ---
    source_product_id: str | list[str] | None = field(
        default_factory=lambda: ["product_id", "id", "asin", "sku", "item_id"]
    )

    # --- URL ---
    # 컬럼에서 읽는다. 없으면 template 으로 생성, 둘 다 없으면 source_site 만 사용
    source_url_col: str | list[str] | None = field(
        default_factory=lambda: ["url", "source_url", "product_url", "link", "product_link"]
    )
    source_url_template: str | None = None  # e.g. "https://example.com/dp/{source_product_id}"

    # --- 상품 속성 ---
    brand: str | list[str] | None = field(
        default_factory=lambda: ["brand", "brand_name", "manufacturer", "seller"]
    )
    category: str | list[str] | None = field(
        default_factory=lambda: ["category", "main_category", "category_name", "department"]
    )

    # --- 가격 ---
    price: str | list[str] | None = field(
        default_factory=lambda: [
            "price",
            "discounted_price",
            "actual_price",
            "selling_price",
            "sale_price",
        ]
    )
    # 컬럼에서 읽거나 literal 고정값 사용
    currency_col: str | None = "currency"
    currency_literal: str = "KRW"  # currency_col 컬럼이 없을 때 사용할 기본값

    # --- 평점 / 리뷰 ---
    rating: str | list[str] | None = field(
        default_factory=lambda: ["rating", "average_rating", "stars", "star_rating"]
    )
    review_count: str | list[str] | None = field(
        default_factory=lambda: [
            "review_count",
            "rating_count",
            "num_reviews",
            "reviews",
            "no_of_ratings",
        ]
    )


class NormalizeFilter(Filter[dict[str, Any], NormalizedProduct]):
    """규칙 기반으로 raw CSV row dict 를 ``NormalizedProduct`` ORM 객체로 변환한다.

    - ``name`` 이 없으면 None 을 반환해 파이프라인에서 drop 한다.
    - 컬럼 매칭은 대소문자 무시(case-insensitive).
    - 가격: 통화 기호·쉼표 제거 후 ``Decimal`` 변환, Numeric(12, 0) 에 맞춰 정수 반올림.
    - 평점: 0.0–5.0 범위로 clamp.

    Usage::

        filter_ = NormalizeFilter(
            source_site="kaggle/amazon-products",
            mapping=ColumnMapping(
                price="discounted_price",
                source_url_template="https://www.amazon.in/dp/{source_product_id}",
            ),
        )
        product = await filter_.process(row)
    """

    def __init__(self, source_site: str, mapping: ColumnMapping | None = None) -> None:
        self._source_site = source_site
        self._mapping = mapping or ColumnMapping()

    async def process(self, item: dict[str, Any]) -> NormalizedProduct | None:
        # 컬럼 조회를 대소문자 무시로 처리하기 위해 lower-key 인덱스를 만든다
        row = _LowerKeyView(item)
        m = self._mapping

        # --- name (필수) ---
        name = _pick_str(row, m.name)
        if not name:
            logger.debug("drop row: missing name  row_keys=%s", list(item.keys())[:6])
            return None

        # --- source_product_id ---
        source_product_id = _pick_str(row, m.source_product_id)

        # --- source_url ---
        source_url = _pick_str(row, m.source_url_col)
        if not source_url and m.source_url_template and source_product_id:
            source_url = m.source_url_template.format(
                source_product_id=source_product_id,
                source_site=self._source_site,
            )
        if not source_url:
            source_url = self._source_site  # 최후 fallback

        # --- brand / category ---
        brand = _pick_str(row, m.brand)
        category = _pick_str(row, m.category)

        # --- price ---
        price = _parse_price(_pick_raw(row, m.price))

        # --- currency ---
        currency = _pick_str(row, m.currency_col) if m.currency_col else None
        currency = (currency or m.currency_literal).strip().upper() or m.currency_literal

        # --- rating ---
        rating = _parse_rating(_pick_raw(row, m.rating))

        # --- review_count ---
        review_count = _parse_int(_pick_raw(row, m.review_count))

        return NormalizedProduct(
            source_site=self._source_site,
            source_product_id=source_product_id,
            source_url=source_url,
            name=name,
            brand=brand,
            category=category,
            price=price,
            currency=currency,
            rating=rating,
            review_count=review_count,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _LowerKeyView:
    """원본 dict 를 변경하지 않고 소문자 키로 값을 조회할 수 있는 뷰."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._index: dict[str, Any] = {k.lower().strip(): v for k, v in data.items()}

    def get(self, key: str) -> Any:
        return self._index.get(key.lower().strip())

    def keys(self) -> list[str]:
        return list(self._index.keys())


def _candidates(spec: str | list[str] | None) -> list[str]:
    if spec is None:
        return []
    return [spec] if isinstance(spec, str) else spec


def _is_missing(val: Any) -> bool:
    """None 또는 pandas/numpy NaN 을 '값 없음'으로 판단한다."""
    if val is None:
        return True
    try:
        return isinstance(val, float) and math.isnan(val)
    except (TypeError, ValueError):
        return False


def _pick_raw(row: _LowerKeyView, spec: str | list[str] | None) -> Any:
    for col in _candidates(spec):
        val = row.get(col)
        if not _is_missing(val):
            return val
    return None


def _pick_str(row: _LowerKeyView, spec: str | list[str] | None) -> str | None:
    val = _pick_raw(row, spec)
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _parse_price(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    cleaned = _PRICE_STRIP.sub("", str(raw))
    if not cleaned:
        return None
    try:
        return Decimal(cleaned).quantize(Decimal("1"))  # Numeric(12, 0) — 정수 반올림
    except InvalidOperation:
        logger.debug("price parse failed: %r", raw)
        return None


def _parse_rating(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    try:
        value = float(str(raw).split()[0])  # "4.5 out of 5" 같은 형식 대응
        clamped = max(0.0, min(5.0, value))
        return Decimal(str(round(clamped, 2)))
    except (ValueError, IndexError):
        logger.debug("rating parse failed: %r", raw)
        return None


def _parse_int(raw: Any) -> int | None:
    if raw is None:
        return None
    cleaned = _PRICE_STRIP.sub("", str(raw))  # 쉼표 제거 ("1,234" → "1234")
    if not cleaned:
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        logger.debug("int parse failed: %r", raw)
        return None
