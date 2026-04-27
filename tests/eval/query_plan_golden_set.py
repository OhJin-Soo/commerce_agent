"""QueryPlan structured output 평가용 골든셋."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlanGoldenCase:
    query: str
    expected_plan: dict


def _plan(**overrides) -> dict:
    base = {
        "csv_filename": None,
        "category": None,
        "max_price_krw": None,
        "min_price_krw": None,
        "min_rating": None,
        "min_review_count": None,
        "brand_include": [],
        "brand_exclude": [],
        "sort": "rating_desc",
        "limit": 20,
        "needs_recommendation": False,
    }
    base.update(overrides)
    return base


QUERY_PLAN_GOLDEN_SET: list[QueryPlanGoldenCase] = [
    QueryPlanGoldenCase(
        query="이어폰 5만원 이하",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            max_price_krw=50_000,
        ),
    ),
    QueryPlanGoldenCase(
        query="노트북 100만원 이상",
        expected_plan=_plan(
            csv_filename="All Electronics.csv",
            category="All Electronics",
            min_price_krw=1_000_000,
        ),
    ),
    QueryPlanGoldenCase(
        query="리뷰 많은 5만원 이하 이어폰 추천해줘",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            max_price_krw=50_000,
            sort="review_count_desc",
            limit=10,
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="애플 제외하고 평점 4점 이상 이어폰",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            min_rating=4.0,
            brand_exclude=["애플"],
        ),
    ),
    QueryPlanGoldenCase(
        query="소니 이어폰",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            brand_include=["소니"],
        ),
    ),
    QueryPlanGoldenCase(
        query="삼성 스마트폰 50만원 이하",
        expected_plan=_plan(
            csv_filename="All Electronics.csv",
            category="All Electronics",
            max_price_krw=500_000,
            brand_include=["삼성"],
        ),
    ),
    QueryPlanGoldenCase(
        query="스피커 평점 4.5 이상",
        expected_plan=_plan(
            csv_filename="Speakers.csv",
            category="Speakers",
            min_rating=4.5,
        ),
    ),
    QueryPlanGoldenCase(
        query="후기 많은 스피커",
        expected_plan=_plan(
            csv_filename="Speakers.csv",
            category="Speakers",
            sort="review_count_desc",
        ),
    ),
    QueryPlanGoldenCase(
        query="가성비 좋은 이어폰 추천",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="가방 목록",
        expected_plan=_plan(
            csv_filename="Bags and Luggage.csv",
            category="Bags and Luggage",
        ),
    ),
    QueryPlanGoldenCase(
        query="시계 최저가",
        expected_plan=_plan(
            csv_filename="Watches.csv",
            category="Watches",
            sort="price_asc",
        ),
    ),
    QueryPlanGoldenCase(
        query="비싼 시계 보여줘",
        expected_plan=_plan(
            csv_filename="Watches.csv",
            category="Watches",
            sort="price_desc",
        ),
    ),
    QueryPlanGoldenCase(
        query="리뷰 1000개 이상 이어폰",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            min_review_count=1000,
        ),
    ),
    QueryPlanGoldenCase(
        query="소니 제외하고 노트북 추천",
        expected_plan=_plan(
            csv_filename="All Electronics.csv",
            category="All Electronics",
            brand_exclude=["소니"],
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="냉장고 200만원 이하 추천",
        expected_plan=_plan(
            csv_filename="Refrigerators.csv",
            category="Refrigerators",
            max_price_krw=2_000_000,
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="세탁기 50만원 이상 100만원 이하",
        expected_plan=_plan(
            csv_filename="Washing Machines.csv",
            category="Washing Machines",
            min_price_krw=500_000,
            max_price_krw=1_000_000,
        ),
    ),
    QueryPlanGoldenCase(
        query="카메라 추천해줘",
        expected_plan=_plan(
            csv_filename="Cameras.csv",
            category="Cameras",
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="캠핑 용품 추천",
        expected_plan=_plan(
            csv_filename="Camping and Hiking.csv",
            category="Camping and Hiking",
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="주방 가전 보여줘",
        expected_plan=_plan(
            csv_filename="Kitchen and Dining.csv",
            category="Kitchen and Dining",
        ),
    ),
    QueryPlanGoldenCase(
        query="이어폰 10개만 추천해줘",
        expected_plan=_plan(
            csv_filename="Headphones.csv",
            category="Headphones",
            limit=10,
            needs_recommendation=True,
        ),
    ),
    QueryPlanGoldenCase(
        query="브랜드별 가격 비교",
        expected_plan=_plan(),
    ),
    QueryPlanGoldenCase(
        query="가구 추천",
        expected_plan=_plan(
            csv_filename="Furniture.csv",
            category="Furniture",
            needs_recommendation=True,
        ),
    ),
]
