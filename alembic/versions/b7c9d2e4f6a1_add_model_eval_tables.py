"""add model eval tables

Revision ID: b7c9d2e4f6a1
Revises: a3f2c1d8e047
Create Date: 2026-04-23 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b7c9d2e4f6a1"
down_revision: Union[str, Sequence[str], None] = "a3f2c1d8e047"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_eval_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("eval_path", sa.Text(), nullable=False),
        sa.Column("probe_count", sa.Integer(), nullable=False),
        sa.Column("has_db_eval", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("category_hit_rate", sa.Float(), nullable=True),
        sa.Column("grounding_rate", sa.Float(), nullable=True),
        sa.Column("answer_faithfulness", sa.Float(), nullable=True),
        sa.Column("execution_accuracy", sa.Float(), nullable=True),
        sa.Column("result_f1", sa.Float(), nullable=True),
        sa.Column("precision_at_5", sa.Float(), nullable=True),
        sa.Column("ndcg_at_5", sa.Float(), nullable=True),
        sa.Column("query_plan_accuracy", sa.Float(), nullable=True),
        sa.Column("fallback_rate", sa.Float(), nullable=True),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("p95_latency_ms", sa.Float(), nullable=True),
        sa.Column("avg_llm_calls", sa.Float(), nullable=True),
        sa.Column("avg_total_tokens", sa.Float(), nullable=True),
        sa.Column("total_estimated_cost", sa.Float(), nullable=True),
        sa.Column("tool_recall", sa.Float(), nullable=True),
        sa.Column("tool_precision", sa.Float(), nullable=True),
        sa.Column("unnecessary_ingest_rate", sa.Float(), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_eval_runs_model_path_created",
        "model_eval_runs",
        ["model_name", "eval_path", "created_at"],
    )

    op.create_table(
        "model_eval_cases",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("expected_csv", sa.Text(), nullable=True),
        sa.Column("actual_csv", sa.Text(), nullable=True),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column("category_hit", sa.Boolean(), nullable=True),
        sa.Column("grounding_rate", sa.Float(), nullable=True),
        sa.Column("answer_faithfulness", sa.Float(), nullable=True),
        sa.Column("execution_accuracy", sa.Float(), nullable=True),
        sa.Column("result_f1", sa.Float(), nullable=True),
        sa.Column("precision_at_5", sa.Float(), nullable=True),
        sa.Column("ndcg_at_5", sa.Float(), nullable=True),
        sa.Column("query_plan_accuracy", sa.Float(), nullable=True),
        sa.Column("used_fallback", sa.Boolean(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("llm_calls", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost", sa.Float(), nullable=True),
        sa.Column("tool_recall", sa.Float(), nullable=True),
        sa.Column("tool_precision", sa.Float(), nullable=True),
        sa.Column("unnecessary_ingest", sa.Boolean(), nullable=True),
        sa.Column("expected_plan", sa.JSON(), nullable=True),
        sa.Column("actual_plan", sa.JSON(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["model_eval_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_eval_cases_run_id", "model_eval_cases", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_model_eval_cases_run_id", table_name="model_eval_cases")
    op.drop_table("model_eval_cases")
    op.drop_index("ix_model_eval_runs_model_path_created", table_name="model_eval_runs")
    op.drop_table("model_eval_runs")
