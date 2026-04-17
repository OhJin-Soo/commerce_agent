from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd

from pipeline.base import Filter

logger = logging.getLogger(__name__)


@dataclass
class KaggleDatasetConfig:
    """Kaggle 데이터셋 다운로드 + 로드 설정.

    Attributes:
        handle:         Kaggle 데이터셋 핸들 (e.g. ``"maharshipandya/-spotify-tracks-dataset"``)
        filename:       데이터셋 내 특정 CSV 파일명. None이면 첫 번째 .csv를 자동 선택.
        nrows:          읽을 최대 행 수. None이면 전체 로드. 개발/테스트 시 유용.
        force_download: 캐시가 있어도 재다운로드할지 여부.
        dtype:          pandas read_csv에 전달할 컬럼별 dtype 힌트.
    """

    handle: str
    filename: str | None = None
    nrows: int | None = None
    force_download: bool = False
    dtype: dict[str, Any] = field(default_factory=dict)


class KaggleLoadFilter(Filter[KaggleDatasetConfig, list[dict[str, Any]]]):
    """Kaggle 데이터셋을 다운로드하고 CSV를 pandas로 읽어 row dict 리스트로 반환한다.

    - kagglehub.dataset_download() 와 pd.read_csv() 는 동기 I/O이므로
      asyncio executor 에서 실행해 이벤트 루프를 블로킹하지 않는다.
    - NaN 값은 None 으로 변환해 downstream filter 처리가 일관되게 한다.

    Usage::

        config = KaggleDatasetConfig(
            handle="maharshipandya/-spotify-tracks-dataset",
            nrows=1000,
        )
        rows = await KaggleLoadFilter().process(config)
    """

    async def process(
        self, item: KaggleDatasetConfig
    ) -> list[dict[str, Any]] | None:
        loop = asyncio.get_running_loop()

        # 1) 다운로드 (캐시 hit 시 즉시 반환)
        logger.info("Downloading Kaggle dataset: %s", item.handle)
        dataset_dir: str = await loop.run_in_executor(
            None,
            lambda: kagglehub.dataset_download(
                item.handle,
                force_download=item.force_download,
            ),
        )
        logger.info("Dataset path: %s", dataset_dir)

        # 2) CSV 파일 선택
        csv_path = self._resolve_csv(Path(dataset_dir), item.filename)
        logger.info("Loading CSV: %s (nrows=%s)", csv_path, item.nrows)

        # 3) pandas 로드 (executor)
        df: pd.DataFrame = await loop.run_in_executor(
            None,
            lambda: pd.read_csv(
                csv_path,
                nrows=item.nrows,
                dtype=item.dtype if item.dtype else None,
            ),
        )
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # 4) NaN → None 으로 정규화 후 dict 리스트 반환
        return df.where(pd.notna(df), other=None).to_dict(orient="records")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_csv(dataset_dir: Path, filename: str | None) -> Path:
        if filename:
            path = dataset_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"{filename} not found in dataset directory {dataset_dir}"
                )
            return path

        csv_files = sorted(dataset_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in dataset directory {dataset_dir}"
            )
        if len(csv_files) > 1:
            logger.warning(
                "Multiple CSVs found, using first: %s",
                [p.name for p in csv_files],
            )
        return csv_files[0]
