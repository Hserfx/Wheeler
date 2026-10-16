"""Koła (maski) i rodzaj zadania workera

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("task", sa.String(32), nullable=True))
    op.add_column("jobs", sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True))
    # Zadania z ticketu 02 (atrapa bez felgi i maski) nie dadzą się przetworzyć nowym procesorem.
    op.execute(
        "UPDATE jobs SET status = 'failed', finished_at = now(), "
        "error = 'Zadanie z poprzedniej wersji aplikacji (bez felgi i maski koła).' "
        "WHERE status IN ('queued', 'running')"
    )
    op.create_table(
        "wheels",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "job_id",
            sa.Integer(),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("click_x", sa.Float(), nullable=True),
        sa.Column("click_y", sa.Float(), nullable=True),
        sa.Column("mask_path", sa.String(512), nullable=True),
        sa.Column("overlay_path", sa.String(512), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_wheels_job_id", "wheels", ["job_id"])


def downgrade() -> None:
    op.drop_table("wheels")
    op.drop_column("jobs", "queued_at")
    op.drop_column("jobs", "task")
