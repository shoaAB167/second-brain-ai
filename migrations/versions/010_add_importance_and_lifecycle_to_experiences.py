"""Add importance and lifecycle to experiences

Revision ID: 010_importance_lifecycle
Revises: 009_convert_embedding_to_vector
Create Date: 2026-08-31 17:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "010_importance_lifecycle"
down_revision: Union[str, None] = "009_convert_embedding_to_vector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add importance column with safe server default 'MEDIUM'
    op.add_column(
        "experiences",
        sa.Column("importance", sa.String(length=20), nullable=False, server_default="MEDIUM"),
    )
    op.create_index("ix_experiences_importance", "experiences", ["importance"])

    # 2. Add lifecycle column with safe server default 'STABLE'
    op.add_column(
        "experiences",
        sa.Column("lifecycle", sa.String(length=20), nullable=False, server_default="STABLE"),
    )
    op.create_index("ix_experiences_lifecycle", "experiences", ["lifecycle"])


def downgrade() -> None:
    op.drop_index("ix_experiences_lifecycle", table_name="experiences")
    op.drop_column("experiences", "lifecycle")

    op.drop_index("ix_experiences_importance", table_name="experiences")
    op.drop_column("experiences", "importance")
