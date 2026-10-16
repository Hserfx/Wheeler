"""Postęp generowania i prośba o anulowanie (ticket 05)

Revision ID: t05_progress
Revises: t04_wheel_detection
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "t05_progress"
down_revision: str | None = "t04_wheel_detection"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("progress", postgresql.JSONB(), nullable=True))
    op.add_column(
        "jobs",
        sa.Column("cancel_requested", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("jobs", "cancel_requested")
    op.drop_column("jobs", "progress")
