"""Automatyczna detekcja kół: źródło i box koła, liczba wykrytych kół zadania

Revision ID: t04_wheel_detection
Revises: 0002
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t04_wheel_detection"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("wheels_detected", sa.Integer(), nullable=True))
    op.add_column("wheels", sa.Column("source", sa.String(32), nullable=True))
    for side in ("left", "top", "right", "bottom"):
        op.add_column("wheels", sa.Column(f"box_{side}", sa.Integer(), nullable=True))
    # Koła z ticketu 03 pochodzą z kliknięcia.
    op.execute("UPDATE wheels SET source = 'click'")


def downgrade() -> None:
    for side in ("left", "top", "right", "bottom"):
        op.drop_column("wheels", f"box_{side}")
    op.drop_column("wheels", "source")
    op.drop_column("jobs", "wheels_detected")
