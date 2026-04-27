from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

REVIEW_KEYWORDS = ("리뷰", "후기", "평가", "의견", "평판", "실사용", "사용기")
_NOISE_TOKENS = {
    "알려줘",
    "보여줘",
    "추천해줘",
    "추천",
    "목록",
    "사용자",
    "사람들",
    "좋은",
    "궁금해",
}


def needs_web_search(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in REVIEW_KEYWORDS)


def extract_product_search_terms(query: str) -> list[str]:
    cleaned = query.lower()
    for kw in REVIEW_KEYWORDS:
        cleaned = cleaned.replace(kw, " ")
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", cleaned)
    terms: list[str] = []
    for token in cleaned.split():
        if token in _NOISE_TOKENS or len(token) <= 1:
            continue
        terms.append(token)
    return terms[:5]


def source_site_to_csv_filename(source_site: str | None) -> str | None:
    if not source_site:
        return None
    return f"{Path(source_site).name}.csv"


async def resolve_product_candidates(
    session_factory: async_sessionmaker,
    query: str,
    limit: int = 5,
) -> list[dict]:
    terms = extract_product_search_terms(query)
    if not terms:
        return []

    clauses: list[str] = []
    params: dict[str, object] = {"limit": limit}
    for i, term in enumerate(terms):
        key = f"term_{i}"
        params[key] = f"%{term}%"
        clauses.append(f"(LOWER(name) LIKE :{key} OR LOWER(COALESCE(brand, '')) LIKE :{key})")

    stmt = text(
        "SELECT id, name, brand, category, price, rating, review_count, source_site, source_url "
        "FROM normalized_products "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY rating DESC NULLS LAST, review_count DESC NULLS LAST "
        "LIMIT :limit"
    )
    async with session_factory() as session:
        result = await session.execute(stmt, params)
        rows = [dict(row._mapping) for row in result.fetchall()]

    for row in rows:
        row["csv_filename"] = source_site_to_csv_filename(row.get("source_site"))
    return rows


def build_db_anchored_web_query(query: str, rows: list[dict]) -> str:
    names = [str(r.get("name", "")).strip() for r in rows if r.get("name")]
    brands = [str(r.get("brand", "")).strip() for r in rows if r.get("brand")]
    anchors = [x for x in [*names[:3], *brands[:2]] if x]
    if not anchors:
        return query
    suffix = " 리뷰 후기 평가"
    return f"{' / '.join(dict.fromkeys(anchors))}{suffix}"
