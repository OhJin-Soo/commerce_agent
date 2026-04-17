from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import NormalizedProduct
from pipeline.base import Pipeline
from pipeline.kaggle_load import KaggleDatasetConfig, KaggleLoadFilter
from pipeline.normalize import ColumnMapping, NormalizeFilter
from pipeline.upsert import UpsertFilter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Kaggle CSV → 정규화 → PostgreSQL upsert 를 묶은 end-to-end 파이프라인.

    Usage::

        async with AsyncSessionLocal() as session:
            pipeline = IngestionPipeline(
                source_site="kaggle/amazon-products",
                session=session,
                mapping=ColumnMapping(price="discounted_price"),
            )
            products = await pipeline.run(
                KaggleDatasetConfig(handle="owner/dataset", nrows=1000)
            )
    """

    def __init__(
        self,
        source_site: str,
        session: AsyncSession,
        mapping: ColumnMapping | None = None,
    ) -> None:
        self._loader = KaggleLoadFilter()
        self._pipeline = Pipeline(
            NormalizeFilter(source_site, mapping),
            UpsertFilter(session),
        )
        self._session = session

    async def run(self, config: KaggleDatasetConfig) -> list[NormalizedProduct]:
        rows = await self._loader.process(config)
        if not rows:
            logger.info("No rows loaded from %s", config.handle)
            return []

        logger.info("Loaded %d raw rows from %s", len(rows), config.handle)
        products = await self._pipeline.run(rows)

        if products:
            await self._session.commit()
            logger.info("Committed %d products", len(products))

        return products
