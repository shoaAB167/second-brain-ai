"""Add unique constraint to experience_classifications.source_message_id

Revision ID: 006_classifications_unique
Revises: 005_create_users_and_auth
Create Date: 2026-08-18 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = "006_classifications_unique"
down_revision: Union[str, None] = "005_create_users_and_auth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add unique constraint to experience_classifications.source_message_id
    op.create_unique_constraint(
        "uq_experience_classifications_source_message_id",
        "experience_classifications",
        ["source_message_id"],
    )


def downgrade() -> None:
    # Revert unique constraint on experience_classifications.source_message_id
    op.drop_constraint(
        "uq_experience_classifications_source_message_id",
        "experience_classifications",
        type_="unique",
    )
