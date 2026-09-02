"""Add lifecycle_status to experiences and create experience_relationships table

Revision ID: 011_lifecycle_relationships
Revises: 010_importance_lifecycle
Create Date: 2026-09-02 22:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "011_lifecycle_relationships"
down_revision: Union[str, None] = "010_importance_lifecycle"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add lifecycle_status column to experiences with safe server default 'ACTIVE'
    op.add_column(
        "experiences",
        sa.Column("lifecycle_status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
    )
    op.create_index("ix_experiences_lifecycle_status", "experiences", ["lifecycle_status"])

    # 2. Create experience_relationships table
    op.create_table(
        "experience_relationships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "source_experience_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_experience_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("experiences.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relationship_type", sa.String(length=30), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_index(
        "ix_experience_relationships_source_id",
        "experience_relationships",
        ["source_experience_id"],
    )
    op.create_index(
        "ix_experience_relationships_target_id",
        "experience_relationships",
        ["target_experience_id"],
    )
    op.create_index(
        "ix_experience_relationships_rel_type",
        "experience_relationships",
        ["relationship_type"],
    )
    op.create_index(
        "ix_experience_relationships_created_at",
        "experience_relationships",
        ["created_at"],
    )
    op.create_index(
        "ix_experience_rel_source_target_type",
        "experience_relationships",
        ["source_experience_id", "target_experience_id", "relationship_type"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("experience_relationships")
    op.drop_index("ix_experiences_lifecycle_status", table_name="experiences")
    op.drop_column("experiences", "lifecycle_status")
