from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
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


# ---------------------------------------------------------------------------
# Eval tables
# ---------------------------------------------------------------------------

class EvalRun(Base):
    """평가 실행 단위 — probe + golden set 집계값을 함께 저장."""

    __tablename__ = "eval_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    model_name = Column(Text, nullable=False)          # 어떤 SQL 생성기(LLM/rule)로 측정했는지
    probe_count = Column(Integer, nullable=False)

    # ── probe 지표 (골든셋 불필요) ──────────────────────────────────────────
    schema_invalid_rate = Column(Float)                # 존재하지 않는 테이블·컬럼 참조 비율
    execution_failure_rate = Column(Float)             # EXPLAIN 실패 비율

    # ── golden set 지표 (골든셋 필요) ──────────────────────────────────────
    execution_accuracy = Column(Float)                 # 결과 집합 완전 일치율 (EX)
    result_f1 = Column(Float)                          # 반환 행 F1
    table_match_rate = Column(Float)                   # 테이블명 일치율
    condition_match_rate = Column(Float)               # WHERE 절 일치율

    cases = relationship("EvalCase", back_populates="run", cascade="all, delete-orphan")


class EvalCase(Base):
    """평가 실행 내 쿼리별 상세 결과."""

    __tablename__ = "eval_cases"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(
        BigInteger,
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    query = Column(Text, nullable=False)
    generated_sql = Column(Text)

    # ── probe 지표 ──────────────────────────────────────────────────────────
    is_schema_valid = Column(Boolean)
    is_executable = Column(Boolean)

    # ── golden set 지표 ─────────────────────────────────────────────────────
    reference_sql = Column(Text)
    execution_accuracy = Column(Float)                 # 0.0 or 1.0
    result_f1 = Column(Float)
    table_match = Column(Float)
    condition_match = Column(Float)

    error_msg = Column(Text)                           # 실행 오류 메시지

    run = relationship("EvalRun", back_populates="cases")

    __table_args__ = (
        Index("ix_eval_cases_run_id", "run_id"),
    )
