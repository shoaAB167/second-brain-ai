"""Add embedding vector, model, status, and embedded_at to experiences

Revision ID: 008_experiences_embedding
Revises: 007_experiences_extraction
Create Date: 2026-08-28 22:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "008_experiences_embedding"
down_revision: Union[str, None] = "007_experiences_extraction"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Require pgvector extension in PostgreSQL
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.add_column("experiences", sa.Column("embedding", Vector(1536), nullable=True))
    op.add_column("experiences", sa.Column("embedding_model", sa.String(length=100), nullable=True))
    op.add_column("experiences", sa.Column("embedding_status", sa.String(length=20), server_default="PENDING", nullable=False))
    op.add_column("experiences", sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_experiences_embedding_status", "experiences", ["embedding_status"])


def downgrade() -> None:
    op.drop_index("ix_experiences_embedding_status", table_name="experiences")
    op.drop_column("experiences", "embedded_at")
    op.drop_column("experiences", "embedding_status")
    op.drop_column("experiences", "embedding_model")
    op.drop_column("experiences", "embedding")
