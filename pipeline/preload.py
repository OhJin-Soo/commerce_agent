from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from pipeline.ingestion import IngestionPipeline
from pipeline.kaggle_load import KaggleDatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class PreloadResult:
    csv_filename: str
    source_site: str
    status: str
    upserted: int = 0
    error: str | None = None


@dataclass
class PreloadSummary:
    results: list[PreloadResult] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return sum(1 for r in self.results if r.status in {"already_loaded", "ingested"})

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")


def source_site_from_csv(csv_filename: str) -> str:
    return f"kaggle/amazon-products/{Path(csv_filename).stem}"


async def preload_csv_files(
    session_factory: async_sessionmaker,
    dataset_handle: str,
    csv_filenames: list[str],
    nrows: int | None,
) -> PreloadSummary:
    """Ensure configured Kaggle CSV files are loaded before request handling.

    Existing source_site data is reused. Missing categories are ingested once at
    application startup, so request-time graph execution can remain read-only.
    """

    summary = PreloadSummary()
    for csv_filename in csv_filenames:
        source_site = source_site_from_csv(csv_filename)
        try:
            async with session_factory() as session:
                row = (
                    await session.execute(
                        text(
                            "SELECT 1 FROM normalized_products "
                            "WHERE source_site = :site LIMIT 1"
                        ),
                        {"site": source_site},
                    )
                ).fetchone()
                if row is not None:
                    summary.results.append(
                        PreloadResult(
                            csv_filename=csv_filename,
                            source_site=source_site,
                            status="already_loaded",
                        )
                    )
                    continue

            logger.info("preload: ingesting %s", csv_filename)
            config = KaggleDatasetConfig(
                handle=dataset_handle,
                filename=csv_filename,
                nrows=nrows,
            )
            async with session_factory() as session:
                products = await IngestionPipeline(
                    source_site=source_site,
                    session=session,
                ).run(config)

            summary.results.append(
                PreloadResult(
                    csv_filename=csv_filename,
                    source_site=source_site,
                    status="ingested",
                    upserted=len(products),
                )
            )
        except Exception as exc:
            logger.exception("preload failed: csv=%s", csv_filename)
            summary.results.append(
                PreloadResult(
                    csv_filename=csv_filename,
                    source_site=source_site,
                    status="failed",
                    error=str(exc),
                )
            )

    return summary
