"""Create users table and add user_id foreign keys

Revision ID: 005_create_users_and_auth
Revises: 004_experience_classifications
Create Date: 2026-08-18 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "005_create_users_and_auth"
down_revision: Union[str, None] = "004_experience_classifications"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # 2. Add user_id column to conversations
    op.add_column(
        "conversations",
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    # 3. Add unique constraint to experiences.source_message_id
    op.create_unique_constraint(
        "uq_experiences_source_message_id", "experiences", ["source_message_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_experiences_source_message_id", "experiences", type_="unique")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_column("conversations", "user_id")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
