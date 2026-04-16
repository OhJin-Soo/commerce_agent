"""create normalized_products and product_facts

Revision ID: 551b1035bf91
Revises: 
Create Date: 2026-04-16 17:21:03.020335

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '551b1035bf91'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "normalized_products",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source_site", sa.Text(), nullable=False),
        sa.Column("source_product_id", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("brand", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("price", sa.Numeric(12, 0), nullable=True),
        sa.Column("currency", sa.Text(), nullable=True, server_default="KRW"),
        sa.Column("rating", sa.Numeric(3, 2), nullable=True),
        sa.Column("review_count", sa.Integer(), nullable=True),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_site", "source_url", name="uq_normalized_products_site_url"),
    )
    op.create_index(
        "ix_normalized_products_category_price",
        "normalized_products",
        ["category", "price"],
    )
    op.create_index(
        "ix_normalized_products_brand_rating",
        "normalized_products",
        ["brand", "rating"],
    )

    op.create_table(
        "product_facts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("product_id", sa.BigInteger(), nullable=False),
        sa.Column("attribute_name", sa.Text(), nullable=False),
        sa.Column("attribute_value", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["normalized_products.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "product_id", "attribute_name", name="uq_product_facts_product_attr"
        ),
    )
    op.create_index(
        "ix_product_facts_attr_name_value",
        "product_facts",
        ["attribute_name", "attribute_value"],
    )


def downgrade() -> None:
    op.drop_index("ix_product_facts_attr_name_value", table_name="product_facts")
    op.drop_table("product_facts")
    op.drop_index("ix_normalized_products_brand_rating", table_name="normalized_products")
    op.drop_index("ix_normalized_products_category_price", table_name="normalized_products")
    op.drop_table("normalized_products")
