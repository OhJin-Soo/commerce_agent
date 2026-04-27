"""add web search metrics to model eval runs

Revision ID: c1d8a4e9f2b3
Revises: b7c9d2e4f6a1
Create Date: 2026-04-24 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c1d8a4e9f2b3"
down_revision: Union[str, Sequence[str], None] = "b7c9d2e4f6a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("model_eval_runs", sa.Column("search_success_rate", sa.Float(), nullable=True))
    op.add_column("model_eval_runs", sa.Column("avg_source_count", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_eval_runs", "avg_source_count")
    op.drop_column("model_eval_runs", "search_success_rate")
