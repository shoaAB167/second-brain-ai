"""Add type, domain, and extraction_confidence columns to experiences

Revision ID: 007_experiences_extraction
Revises: 006_classifications_unique
Create Date: 2026-08-25 22:50:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "007_experiences_extraction"
down_revision: Union[str, None] = "006_classifications_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("experiences", sa.Column("type", sa.String(length=50), nullable=True))
    op.add_column("experiences", sa.Column("domain", sa.String(length=100), nullable=True))
    op.add_column("experiences", sa.Column("extraction_confidence", sa.Float(), nullable=True))
    op.create_index("ix_experiences_type", "experiences", ["type"])
    op.create_index("ix_experiences_domain", "experiences", ["domain"])


def downgrade() -> None:
    op.drop_index("ix_experiences_domain", table_name="experiences")
    op.drop_index("ix_experiences_type", table_name="experiences")
    op.drop_column("experiences", "extraction_confidence")
    op.drop_column("experiences", "domain")
    op.drop_column("experiences", "type")
