"""Biblioteka felg (ticket 06)

Revision ID: t06_rim_library
Revises: 0002
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t06_rim_library"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rims",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("name", sa.String(200), nullable=True),
        sa.Column("original_path", sa.String(512), nullable=False),
        sa.Column("cut_path", sa.String(512), nullable=True),
        sa.Column("click_x", sa.Float(), nullable=True),
        sa.Column("click_y", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("use_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_rims_status", "rims", ["status"])


def downgrade() -> None:
    op.drop_table("rims")
