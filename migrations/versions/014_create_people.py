"""Create people table for Relationship / People Model (PR #29)

Revision ID: 014_create_people
Revises: 013_create_personal_patterns
Create Date: 2026-09-06 16:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = "014_create_people"
down_revision: Union[str, None] = "013_create_personal_patterns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "people",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            sa.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("relationship_type", sa.String(length=50), nullable=False, server_default="OTHER"),
        sa.Column("notes", sa.Text(), nullable=True),
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
    op.create_index(op.f("ix_people_user_id"), "people", ["user_id"], unique=False)
    op.create_index(op.f("ix_people_name"), "people", ["name"], unique=False)
    op.create_index(op.f("ix_people_relationship_type"), "people", ["relationship_type"], unique=False)
    op.create_index(op.f("ix_people_created_at"), "people", ["created_at"], unique=False)
    op.create_index("ix_people_user_id_name", "people", ["user_id", "name"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_people_user_id_name", table_name="people")
    op.drop_index(op.f("ix_people_created_at"), table_name="people")
    op.drop_index(op.f("ix_people_relationship_type"), table_name="people")
    op.drop_index(op.f("ix_people_name"), table_name="people")
    op.drop_index(op.f("ix_people_user_id"), table_name="people")
    op.drop_table("people")
