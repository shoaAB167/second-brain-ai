"""Create personal_patterns table for Personal Model (PR #23)

Revision ID: 013_create_personal_patterns
Revises: 012_rich_exp_emotional_context
Create Date: 2026-09-06 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "013_create_personal_patterns"
down_revision: Union[str, None] = "012_rich_exp_emotional_context"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "personal_patterns",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=50), nullable=False, server_default="GENERAL"),
        sa.Column("evidence_ids", postgresql.JSON(astext_type=sa.Text()).with_variant(sa.JSON(), "sqlite"), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False, server_default="HYPOTHESIS"),
        sa.Column("first_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "superseded_by_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("personal_patterns.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(op.f("ix_personal_patterns_user_id"), "personal_patterns", ["user_id"], unique=False)
    op.create_index(op.f("ix_personal_patterns_domain"), "personal_patterns", ["domain"], unique=False)
    op.create_index(op.f("ix_personal_patterns_status"), "personal_patterns", ["status"], unique=False)
    op.create_index(op.f("ix_personal_patterns_created_at"), "personal_patterns", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_personal_patterns_created_at"), table_name="personal_patterns")
    op.drop_index(op.f("ix_personal_patterns_status"), table_name="personal_patterns")
    op.drop_index(op.f("ix_personal_patterns_domain"), table_name="personal_patterns")
    op.drop_index(op.f("ix_personal_patterns_user_id"), table_name="personal_patterns")
    op.drop_table("personal_patterns")
