"""End-to-end tests for the ingestion pipeline.

외부 의존성 없이 단독 실행 가능:
- kagglehub.dataset_download  →  unittest.mock.patch 로 임시 CSV 디렉터리 대체
- AsyncSession                →  AsyncMock 으로 대체 (실제 DB 불필요)
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
from itertools import count
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from db.models import NormalizedProduct
from pipeline import ColumnMapping, NormalizeFilter
from pipeline.ingestion import IngestionPipeline
from pipeline.kaggle_load import KaggleDatasetConfig
from pipeline.upsert import UpsertFilter


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _make_session(start_id: int = 1) -> AsyncMock:
    """id 를 순서대로 발급하는 가짜 AsyncSession."""
    session = AsyncMock()
    _counter = count(start_id)

    def _fake_execute(_stmt):
        row = MagicMock()
        row.id = next(_counter)
        row.updated_at = datetime(2026, 4, 17, tzinfo=timezone.utc)
        result = MagicMock()
        result.one.return_value = row
        return result

    session.execute = AsyncMock(side_effect=_fake_execute)
    return session


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


_AMAZON_ROWS = [
    {
        "product_id": "B001",
        "product_name": "Sony WH-1000XM5",
        "brand": "Sony",
        "category": "Electronics",
        "discounted_price": "₹24,990",
        "rating": "4.4",
        "no_of_ratings": "12,345",
        "product_link": "https://amzn.in/dp/B001",
    },
    {
        "product_id": "B002",
        "product_name": "Apple AirPods Pro",
        "brand": "Apple",
        "category": "Electronics",
        "discounted_price": "₹19,900",
        "rating": "4.8",
        "no_of_ratings": "8,900",
        "product_link": "https://amzn.in/dp/B002",
    },
    # drop 대상: product_name 없음
    {
        "product_id": "B003",
        "brand": "Unknown",
        "discounted_price": "₹100",
        "product_link": "https://amzn.in/dp/B003",
    },
]

_MAPPING = ColumnMapping(
    price=["discounted_price", "price"],
    source_url_col="product_link",
)

SOURCE_SITE = "kaggle/amazon-products"


# ---------------------------------------------------------------------------
# NormalizeFilter — 단위 테스트
# ---------------------------------------------------------------------------

class TestNormalizeFilter:
    @pytest.fixture
    def f(self):
        return NormalizeFilter(SOURCE_SITE, _MAPPING)

    async def test_valid_row_fields(self, f):
        result = await f.process(_AMAZON_ROWS[0])

        assert result is not None
        assert result.name == "Sony WH-1000XM5"
        assert result.brand == "Sony"
        assert result.category == "Electronics"
        assert result.price == Decimal("24990")
        assert result.rating == Decimal("4.40")
        assert result.review_count == 12345
        assert result.source_site == SOURCE_SITE
        assert result.source_url == "https://amzn.in/dp/B001"
        assert result.source_product_id == "B001"

    async def test_drops_row_without_name(self, f):
        assert await f.process(_AMAZON_ROWS[2]) is None

    @pytest.mark.parametrize("raw,expected", [
        ("₹24,990",  Decimal("24990")),
        ("$249.99",  Decimal("250")),    # Numeric(12,0) → 정수 반올림
        ("1,234",    Decimal("1234")),
        ("",         None),
        (None,       None),
    ])
    async def test_price_parsing(self, f, raw, expected):
        row = {"product_name": "Item", "discounted_price": raw, "product_link": "http://x.com"}
        result = await f.process(row)
        assert result is not None
        assert result.price == expected

    @pytest.mark.parametrize("raw,expected", [
        ("4.8 out of 5", Decimal("4.80")),  # 문자열 토큰
        ("9.9",          Decimal("5.00")),  # 상한 clamp
        ("-1.0",         Decimal("0.00")),  # 하한 clamp
        ("bad",          None),
    ])
    async def test_rating_parsing(self, f, raw, expected):
        row = {"product_name": "Item", "rating": raw, "product_link": "http://x.com"}
        result = await f.process(row)
        assert result is not None
        assert result.rating == expected

    async def test_source_url_template_fallback(self):
        mapping = ColumnMapping(
            price="price",
            source_url_col=None,
            source_url_template="https://example.com/dp/{source_product_id}",
        )
        f = NormalizeFilter(SOURCE_SITE, mapping)
        row = {"product_name": "Item", "product_id": "SKU-99"}
        result = await f.process(row)
        assert result is not None
        assert result.source_url == "https://example.com/dp/SKU-99"

    async def test_case_insensitive_column_matching(self, f):
        row = {
            "PRODUCT_NAME": "Widget",
            "Brand": "Acme",
            "DISCOUNTED_PRICE": "500",
            "product_link": "http://x.com",
        }
        result = await f.process(row)
        assert result is not None
        assert result.name == "Widget"
        assert result.brand == "Acme"
        assert result.price == Decimal("500")


# ---------------------------------------------------------------------------
# UpsertFilter — 단위 테스트
# ---------------------------------------------------------------------------

class TestUpsertFilter:
    def _product(self, **overrides) -> NormalizedProduct:
        defaults = dict(
            source_site=SOURCE_SITE,
            source_product_id="P001",
            source_url="https://test.com/p/001",
            name="Test Product",
            brand="TestBrand",
            category="Test",
            price=Decimal("10000"),
            currency="KRW",
            rating=Decimal("4.5"),
            review_count=100,
        )
        return NormalizedProduct(**{**defaults, **overrides})

    async def test_returns_product_with_db_assigned_id(self):
        session = _make_session(start_id=42)
        result = await UpsertFilter(session).process(self._product())

        assert result is not None
        assert result.id == 42
        assert result.updated_at == datetime(2026, 4, 17, tzinfo=timezone.utc)

    async def test_execute_called_once_per_item(self):
        session = _make_session()
        await UpsertFilter(session).process(self._product())
        session.execute.assert_awaited_once()

    async def test_sequential_ids_across_calls(self):
        session = _make_session(start_id=1)
        f = UpsertFilter(session)
        products = [
            self._product(source_url=f"https://test.com/{i}", name=f"Prod {i}")
            for i in range(3)
        ]
        results = [await f.process(p) for p in products]

        assert [r.id for r in results] == [1, 2, 3]
        assert session.execute.await_count == 3


# ---------------------------------------------------------------------------
# IngestionPipeline — end-to-end 테스트
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    @pytest.fixture
    def csv_dir(self, tmp_path) -> Path:
        _write_csv(tmp_path / "products.csv", _AMAZON_ROWS)
        return tmp_path

    @pytest.fixture
    def session(self):
        return _make_session(start_id=100)

    @pytest.fixture
    def pipeline(self, session):
        return IngestionPipeline(
            source_site=SOURCE_SITE,
            session=session,
            mapping=_MAPPING,
        )

    @pytest.fixture
    def config(self):
        return KaggleDatasetConfig(handle="owner/amazon-products")

    # -- 정상 흐름 --

    async def test_returns_only_valid_products(self, pipeline, config, csv_dir):
        with patch("kagglehub.dataset_download", return_value=str(csv_dir)):
            products = await pipeline.run(config)

        # 3행 중 name 없는 1행 drop → 2건
        assert len(products) == 2

    async def test_product_names_in_order(self, pipeline, config, csv_dir):
        with patch("kagglehub.dataset_download", return_value=str(csv_dir)):
            products = await pipeline.run(config)

        assert products[0].name == "Sony WH-1000XM5"
        assert products[1].name == "Apple AirPods Pro"

    async def test_db_ids_assigned(self, pipeline, config, csv_dir):
        with patch("kagglehub.dataset_download", return_value=str(csv_dir)):
            products = await pipeline.run(config)

        assert products[0].id == 100
        assert products[1].id == 101

    async def test_price_normalized(self, pipeline, config, csv_dir):
        with patch("kagglehub.dataset_download", return_value=str(csv_dir)):
            products = await pipeline.run(config)

        assert products[0].price == Decimal("24990")
        assert products[1].price == Decimal("19900")

    async def test_source_site_propagated(self, pipeline, config, csv_dir):
        with patch("kagglehub.dataset_download", return_value=str(csv_dir)):
            products = await pipeline.run(config)

        assert all(p.source_site == SOURCE_SITE for p in products)

    async def test_commit_called_after_upsert(self, pipeline, config, csv_dir, session):
        with patch("kagglehub.dataset_download", return_value=str(csv_dir)):
            await pipeline.run(config)

        session.commit.assert_awaited_once()

    # -- 경계 케이스 --

    async def test_empty_csv_returns_empty_list(self, pipeline, tmp_path, session):
        (tmp_path / "empty.csv").write_text("product_name,price\n")
        config = KaggleDatasetConfig(handle="owner/empty")

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            products = await pipeline.run(config)

        assert products == []
        session.commit.assert_not_awaited()

    async def test_all_rows_drop_no_commit(self, tmp_path, session):
        """모든 행이 name 없으면 commit 을 호출하지 않는다."""
        rows = [{"brand": "X", "price": "100"}, {"brand": "Y", "price": "200"}]
        _write_csv(tmp_path / "no_name.csv", rows)

        pipeline = IngestionPipeline(SOURCE_SITE, session, _MAPPING)
        config = KaggleDatasetConfig(handle="owner/no-name")

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            products = await pipeline.run(config)

        assert products == []
        session.commit.assert_not_awaited()

    async def test_nrows_limits_loaded_rows(self, tmp_path, session):
        """nrows=1 이면 1건만 처리된다."""
        _write_csv(tmp_path / "products.csv", _AMAZON_ROWS)
        pipeline = IngestionPipeline(SOURCE_SITE, session, _MAPPING)
        config = KaggleDatasetConfig(handle="owner/amazon-products", nrows=1)

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            products = await pipeline.run(config)

        assert len(products) == 1
        assert products[0].name == "Sony WH-1000XM5"
