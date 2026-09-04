"""Add emotional_context, people_involved, temporal_context, and evidence_level to experiences

Revision ID: 012_rich_exp_emotional_context
Revises: 011_lifecycle_relationships
Create Date: 2026-09-02 23:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "012_rich_exp_emotional_context"
down_revision: Union[str, None] = "011_lifecycle_relationships"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add rich experience columns to experiences table
    op.add_column("experiences", sa.Column("emotional_context", postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column("experiences", sa.Column("people_involved", postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column("experiences", sa.Column("temporal_context", sa.String(length=200), nullable=True))
    op.add_column("experiences", sa.Column("evidence_level", sa.String(length=30), nullable=False, server_default="EXTRACTED"))
    op.create_index(op.f("ix_experiences_evidence_level"), "experiences", ["evidence_level"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_experiences_evidence_level"), table_name="experiences")
    op.drop_column("experiences", "evidence_level")
    op.drop_column("experiences", "temporal_context")
    op.drop_column("experiences", "people_involved")
    op.drop_column("experiences", "emotional_context")
