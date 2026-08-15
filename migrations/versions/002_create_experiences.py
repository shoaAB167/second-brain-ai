"""Create experiences table

Revision ID: 002_experiences
Revises: 001_conversations_and_messages
Create Date: 2026-08-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "002_experiences"
down_revision: Union[str, None] = "001_conversations_and_messages"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Explicitly create Enum types checkfirst
    sa.Enum(
        "CHAT", "FILE", "EMAIL", "CALENDAR", "GITHUB", "VOICE", "API", "MANUAL", "TOOL",
        name="experiencesource_enum",
    ).create(op.get_bind(), checkfirst=True)

    sa.Enum(
        "RECEIVED", "PROCESSING", "PROCESSED", "FAILED",
        name="experiencestatus_enum",
    ).create(op.get_bind(), checkfirst=True)

    source_enum = postgresql.ENUM(
        "CHAT", "FILE", "EMAIL", "CALENDAR", "GITHUB", "VOICE", "API", "MANUAL", "TOOL",
        name="experiencesource_enum",
        create_type=False,
    )

    status_enum = postgresql.ENUM(
        "RECEIVED", "PROCESSING", "PROCESSED", "FAILED",
        name="experiencestatus_enum",
        create_type=False,
    )

    # Create experiences table
    op.create_table(
        "experiences",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", source_enum, nullable=False),
        sa.Column("status", status_enum, nullable=False, server_default="RECEIVED"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for efficient retrieval
    op.create_index("ix_experiences_user_id", "experiences", ["user_id"])
    op.create_index("ix_experiences_created_at", "experiences", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_experiences_created_at", table_name="experiences")
    op.drop_index("ix_experiences_user_id", table_name="experiences")
    op.drop_table("experiences")

    sa.Enum(
        "RECEIVED", "PROCESSING", "PROCESSED", "FAILED",
        name="experiencestatus_enum",
    ).drop(op.get_bind(), checkfirst=True)

    sa.Enum(
        "CHAT", "FILE", "EMAIL", "CALENDAR", "GITHUB", "VOICE", "API", "MANUAL", "TOOL",
        name="experiencesource_enum",
    ).drop(op.get_bind(), checkfirst=True)
