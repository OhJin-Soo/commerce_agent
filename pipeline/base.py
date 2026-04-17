from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")

logger = logging.getLogger(__name__)


class Filter(ABC, Generic[T, U]):
    """단일 변환 단계 ABC.

    - process()가 값을 반환하면 다음 Filter로 전달
    - process()가 None을 반환하면 해당 아이템을 파이프라인에서 제거(drop)
    """

    @abstractmethod
    async def process(self, item: T) -> U | None: ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class Pipeline(Generic[T, U]):
    """Filter 인스턴스를 순서대로 연결해 데이터를 흘려보내는 파이프라인.

    Usage::

        pipeline = Pipeline(CSVNormalizeFilter(), PriceRangeFilter(), UpsertFilter())
        results = await pipeline.run(rows)
    """

    def __init__(self, *filters: Filter) -> None:
        if not filters:
            raise ValueError("Pipeline requires at least one filter")
        self._filters: list[Filter] = list(filters)

    async def run_one(self, item: T) -> U | None:
        """단일 아이템을 모든 Filter에 통과시킨다. 중간에 drop되면 None 반환."""
        result: Any = item
        for f in self._filters:
            result = await f.process(result)
            if result is None:
                logger.debug("%s dropped item", f)
                return None
        return result

    async def run(self, items: list[T]) -> list[U]:
        """배치 처리. drop된 아이템은 결과에서 제외된다."""
        out: list[U] = []
        for item in items:
            result = await self.run_one(item)
            if result is not None:
                out.append(result)
        return out

    async def stream(self, items: AsyncIterator[T]) -> AsyncIterator[U]:
        """스트리밍 처리. 대용량 CSV 등 메모리를 아껴야 할 때 사용."""
        async for item in items:
            result = await self.run_one(item)
            if result is not None:
                yield result  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self._filters)

    def __repr__(self) -> str:
        chain = " -> ".join(repr(f) for f in self._filters)
        return f"Pipeline({chain})"
