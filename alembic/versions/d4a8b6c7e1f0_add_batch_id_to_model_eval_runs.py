"""add batch id to model eval runs

Revision ID: d4a8b6c7e1f0
Revises: c1d8a4e9f2b3
Create Date: 2026-04-24 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4a8b6c7e1f0"
down_revision: Union[str, Sequence[str], None] = "c1d8a4e9f2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("model_eval_runs", sa.Column("batch_id", sa.Text(), nullable=True))
    op.create_index("ix_model_eval_runs_batch_id", "model_eval_runs", ["batch_id"])


def downgrade() -> None:
    op.drop_index("ix_model_eval_runs_batch_id", table_name="model_eval_runs")
    op.drop_column("model_eval_runs", "batch_id")
