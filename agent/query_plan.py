from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import Select, or_, select

from db.models import NormalizedProduct


SortMode = Literal[
    "rating_desc",
    "price_asc",
    "price_desc",
    "review_count_desc",
    "relevance",
]


class QueryPlan(BaseModel):
    """LLM이 자연어 상품 질의를 구조화해 반환하는 검색 계획.

    SQL은 이 모델을 직접 실행하지 않고, 아래 SQLAlchemy builder가 허용된 필드만
    안전하게 select 조건으로 변환한다.
    """

    csv_filename: str | None = Field(
        None,
        description="Kaggle CSV filename for the product category, e.g. Headphones.csv.",
    )
    category: str | None = Field(
        None,
        description="Human-readable category label, e.g. Headphones.",
    )
    max_price_krw: int | None = Field(None, ge=0)
    min_price_krw: int | None = Field(None, ge=0)
    min_rating: float | None = Field(None, ge=0.0, le=5.0)
    min_review_count: int | None = Field(None, ge=0)
    brand_include: list[str] = Field(default_factory=list)
    brand_exclude: list[str] = Field(default_factory=list)
    sort: SortMode = "rating_desc"
    limit: int = Field(20, ge=1, le=50)
    needs_recommendation: bool = False

    @field_validator("brand_include", "brand_exclude")
    @classmethod
    def _strip_brand_terms(cls, value: list[str]) -> list[str]:
        return [term.strip() for term in value if term and term.strip()]


def coerce_query_plan(value: QueryPlan | dict) -> QueryPlan:
    if isinstance(value, QueryPlan):
        return value
    try:
        return QueryPlan.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"invalid query_plan: {exc}") from exc


def resolve_plan_csv_filename(
    plan: QueryPlan,
    category_map: dict[str, tuple[str, str]],
    fallback_csv_filename: str | None = None,
) -> str | None:
    """Resolve a QueryPlan category/csv into one of the supported CSV files."""

    allowed_csv = {csv for csv, _ in category_map.values()}
    if plan.csv_filename in allowed_csv:
        return plan.csv_filename

    if plan.category:
        normalized = plan.category.strip().lower()
        for csv, label in category_map.values():
            if normalized in {label.lower(), csv.removesuffix(".csv").lower()}:
                return csv

    return fallback_csv_filename


def build_select_from_query_plan(
    plan_value: QueryPlan | dict,
    source_site: str | None,
    exchange_rate: float,
) -> Select:
    """Build a bounded read-only product query from a structured QueryPlan."""

    plan = coerce_query_plan(plan_value)
    stmt = select(
        NormalizedProduct.id,
        NormalizedProduct.name,
        NormalizedProduct.brand,
        NormalizedProduct.category,
        NormalizedProduct.price,
        NormalizedProduct.rating,
        NormalizedProduct.review_count,
        NormalizedProduct.source_url,
    )

    if source_site:
        stmt = stmt.where(NormalizedProduct.source_site == source_site)

    if plan.max_price_krw is not None:
        max_price_inr = int(plan.max_price_krw / exchange_rate) if exchange_rate else plan.max_price_krw
        stmt = stmt.where(NormalizedProduct.price <= max_price_inr)

    if plan.min_price_krw is not None:
        min_price_inr = int(plan.min_price_krw / exchange_rate) if exchange_rate else plan.min_price_krw
        stmt = stmt.where(NormalizedProduct.price >= min_price_inr)

    if plan.min_rating is not None:
        stmt = stmt.where(NormalizedProduct.rating >= plan.min_rating)

    if plan.min_review_count is not None:
        stmt = stmt.where(NormalizedProduct.review_count >= plan.min_review_count)

    if plan.brand_include:
        stmt = stmt.where(
            or_(*(NormalizedProduct.brand.ilike(f"%{brand}%") for brand in plan.brand_include))
        )

    for brand in plan.brand_exclude:
        stmt = stmt.where(
            or_(NormalizedProduct.brand.is_(None), NormalizedProduct.brand.not_ilike(f"%{brand}%"))
        )

    if plan.sort == "price_asc":
        stmt = stmt.order_by(NormalizedProduct.price.asc().nulls_last())
    elif plan.sort == "price_desc":
        stmt = stmt.order_by(NormalizedProduct.price.desc().nulls_last())
    elif plan.sort == "review_count_desc":
        stmt = stmt.order_by(NormalizedProduct.review_count.desc().nulls_last())
    elif plan.sort == "relevance":
        stmt = stmt.order_by(
            NormalizedProduct.rating.desc().nulls_last(),
            NormalizedProduct.review_count.desc().nulls_last(),
        )
    else:
        stmt = stmt.order_by(NormalizedProduct.rating.desc().nulls_last())

    return stmt.limit(plan.limit)
