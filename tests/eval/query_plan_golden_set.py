"""QueryPlan structured output 평가용 골든셋."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlanGoldenCase:
    query: str
    expected_plan: dict


QUERY_PLAN_GOLDEN_SET: list[QueryPlanGoldenCase] = [
    QueryPlanGoldenCase(
        query="이어폰 5만원 이하",
        expected_plan={
            "csv_filename": "Headphones.csv",
            "category": "Headphones",
            "max_price_krw": 50_000,
            "min_price_krw": None,
            "min_rating": None,
            "min_review_count": None,
            "brand_include": [],
            "brand_exclude": [],
            "sort": "rating_desc",
            "limit": 20,
            "needs_recommendation": False,
        },
    ),
    QueryPlanGoldenCase(
        query="노트북 100만원 이상",
        expected_plan={
            "csv_filename": "All Electronics.csv",
            "category": "All Electronics",
            "max_price_krw": None,
            "min_price_krw": 1_000_000,
            "min_rating": None,
            "min_review_count": None,
            "brand_include": [],
            "brand_exclude": [],
            "sort": "rating_desc",
            "limit": 20,
            "needs_recommendation": False,
        },
    ),
    QueryPlanGoldenCase(
        query="리뷰 많은 5만원 이하 이어폰 추천해줘",
        expected_plan={
            "csv_filename": "Headphones.csv",
            "category": "Headphones",
            "max_price_krw": 50_000,
            "min_price_krw": None,
            "min_rating": None,
            "min_review_count": None,
            "brand_include": [],
            "brand_exclude": [],
            "sort": "review_count_desc",
            "limit": 10,
            "needs_recommendation": True,
        },
    ),
    QueryPlanGoldenCase(
        query="애플 제외하고 평점 4점 이상 이어폰",
        expected_plan={
            "csv_filename": "Headphones.csv",
            "category": "Headphones",
            "max_price_krw": None,
            "min_price_krw": None,
            "min_rating": 4.0,
            "min_review_count": None,
            "brand_include": [],
            "brand_exclude": ["애플"],
            "sort": "rating_desc",
            "limit": 20,
            "needs_recommendation": False,
        },
    ),
]
