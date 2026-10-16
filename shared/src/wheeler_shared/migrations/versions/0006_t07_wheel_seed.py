"""Seed pojedynczego koła i prośba o jego ponowne wygenerowanie (ticket 07)

Revision ID: t07_wheel_seed
Revises: t05_progress
Create Date: 2026-10-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t07_wheel_seed"
down_revision: str | None = "t05_progress"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("wheels", sa.Column("seed", sa.Integer(), nullable=True))
    op.add_column("wheels", sa.Column("pending_seed", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("wheels", "pending_seed")
    op.drop_column("wheels", "seed")
