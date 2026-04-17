from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import NormalizedProduct
from pipeline.base import Filter

logger = logging.getLogger(__name__)

# INSERT 시 전달할 컬럼 (id, updated_at, facts 관계 제외)
_INSERT_COLS: tuple[str, ...] = (
    "source_site",
    "source_product_id",
    "source_url",
    "name",
    "brand",
    "category",
    "price",
    "currency",
    "rating",
    "review_count",
)

# ON CONFLICT 충돌 시 갱신할 컬럼 (conflict key인 source_site, source_url 제외)
_UPDATE_COLS: tuple[str, ...] = (
    "source_product_id",
    "name",
    "brand",
    "category",
    "price",
    "currency",
    "rating",
    "review_count",
)


class UpsertFilter(Filter[NormalizedProduct, NormalizedProduct]):
    """``NormalizedProduct`` 를 PostgreSQL 에 upsert 하고 ``id`` / ``updated_at`` 를 채워 반환한다.

    충돌 기준: ``uq_normalized_products_site_url`` (``source_site``, ``source_url``)
    - INSERT 시 모든 컬럼을 삽입
    - 충돌 시 ``_UPDATE_COLS`` 를 덮어쓰고 ``updated_at = NOW()`` 갱신
    - ``RETURNING id, updated_at`` 으로 DB 할당값을 Python 객체에 반영

    트랜잭션 관리는 호출자(caller) 책임이다. 이 필터는 ``flush()`` 만 수행하며
    ``commit()`` 은 호출하지 않는다.

    Usage::

        async with AsyncSessionLocal() as session:
            upsert = UpsertFilter(session)
            pipeline = Pipeline(normalize_filter, upsert)
            products = await pipeline.run(rows)
            await session.commit()
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def process(self, item: NormalizedProduct) -> NormalizedProduct | None:
        values = {col: getattr(item, col) for col in _INSERT_COLS}

        stmt = pg_insert(NormalizedProduct).values(**values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_normalized_products_site_url",
            set_={
                **{col: getattr(stmt.excluded, col) for col in _UPDATE_COLS},
                "updated_at": func.now(),
            },
        ).returning(
            NormalizedProduct.id,
            NormalizedProduct.updated_at,
        )

        result = await self._session.execute(stmt)
        row = result.one()

        item.id = row.id
        item.updated_at = row.updated_at

        logger.debug(
            "upserted product id=%s site=%r name=%r",
            item.id,
            item.source_site,
            item.name,
        )
        return item
