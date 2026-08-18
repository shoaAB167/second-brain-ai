"""Add source_message_id to experiences table

Revision ID: 003_add_source_message_id
Revises: 002_experiences
Create Date: 2026-08-15 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "003_add_source_message_id"
down_revision: Union[str, None] = "002_experiences"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add source_message_id column and foreign key pointing to messages.id
    op.add_column(
        "experiences",
        sa.Column("source_message_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_experiences_source_message_id_messages",
        "experiences",
        "messages",
        ["source_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_experiences_source_message_id",
        "experiences",
        ["source_message_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_experiences_source_message_id", table_name="experiences")
    op.drop_constraint(
        "fk_experiences_source_message_id_messages",
        table_name="experiences",
        type_="foreignkey",
    )
    op.drop_column("experiences", "source_message_id")
