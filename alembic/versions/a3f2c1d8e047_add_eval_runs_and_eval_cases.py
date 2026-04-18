"""add eval_runs and eval_cases

Revision ID: a3f2c1d8e047
Revises: 551b1035bf91
Create Date: 2026-04-19 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a3f2c1d8e047"
down_revision: Union[str, Sequence[str], None] = "551b1035bf91"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("probe_count", sa.Integer(), nullable=False),
        # probe 지표
        sa.Column("schema_invalid_rate", sa.Float(), nullable=True),
        sa.Column("execution_failure_rate", sa.Float(), nullable=True),
        # golden set 지표
        sa.Column("execution_accuracy", sa.Float(), nullable=True),
        sa.Column("result_f1", sa.Float(), nullable=True),
        sa.Column("table_match_rate", sa.Float(), nullable=True),
        sa.Column("condition_match_rate", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "eval_cases",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("generated_sql", sa.Text(), nullable=True),
        # probe 지표
        sa.Column("is_schema_valid", sa.Boolean(), nullable=True),
        sa.Column("is_executable", sa.Boolean(), nullable=True),
        # golden set 지표
        sa.Column("reference_sql", sa.Text(), nullable=True),
        sa.Column("execution_accuracy", sa.Float(), nullable=True),
        sa.Column("result_f1", sa.Float(), nullable=True),
        sa.Column("table_match", sa.Float(), nullable=True),
        sa.Column("condition_match", sa.Float(), nullable=True),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_eval_cases_run_id", "eval_cases", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_eval_cases_run_id", table_name="eval_cases")
    op.drop_table("eval_cases")
    op.drop_table("eval_runs")
