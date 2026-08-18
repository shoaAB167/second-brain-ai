"""Create experience_classifications table

Revision ID: 004_experience_classifications
Revises: 003_add_source_message_id
Create Date: 2026-08-15 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "004_experience_classifications"
down_revision: Union[str, None] = "003_add_source_message_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "experience_classifications",
        sa.Column("id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("source_message_id", sa.UUID(), nullable=True),
        sa.Column("experience_id", sa.UUID(), nullable=True),
        sa.Column("is_experience", sa.Boolean(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_message_id"], ["messages.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["experience_id"], ["experiences.id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        "ix_experience_classifications_source_message_id",
        "experience_classifications",
        ["source_message_id"],
    )
    op.create_index(
        "ix_experience_classifications_experience_id",
        "experience_classifications",
        ["experience_id"],
    )
    op.create_index(
        "ix_experience_classifications_created_at",
        "experience_classifications",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_experience_classifications_created_at",
        table_name="experience_classifications",
    )
    op.drop_index(
        "ix_experience_classifications_experience_id",
        table_name="experience_classifications",
    )
    op.drop_index(
        "ix_experience_classifications_source_message_id",
        table_name="experience_classifications",
    )
    op.drop_table("experience_classifications")
