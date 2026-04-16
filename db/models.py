from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Index,
    Numeric,
    Integer,
    Text,
    TIMESTAMP,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class NormalizedProduct(Base):
    __tablename__ = "normalized_products"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_site = Column(Text, nullable=False)
    source_product_id = Column(Text)
    source_url = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    brand = Column(Text)
    category = Column(Text)
    price = Column(Numeric(12, 0))
    currency = Column(Text, default="KRW")
    rating = Column(Numeric(3, 2))
    review_count = Column(Integer)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    facts = relationship("ProductFact", back_populates="product", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source_site", "source_url", name="uq_normalized_products_site_url"),
        Index("ix_normalized_products_category_price", "category", "price"),
        Index("ix_normalized_products_brand_rating", "brand", "rating"),
    )


class ProductFact(Base):
    __tablename__ = "product_facts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(
        BigInteger,
        ForeignKey("normalized_products.id", ondelete="CASCADE"),
        nullable=False,
    )
    attribute_name = Column(Text, nullable=False)
    attribute_value = Column(Text, nullable=False)

    product = relationship("NormalizedProduct", back_populates="facts")

    __table_args__ = (
        UniqueConstraint("product_id", "attribute_name", name="uq_product_facts_product_attr"),
        Index("ix_product_facts_attr_name_value", "attribute_name", "attribute_value"),
    )
