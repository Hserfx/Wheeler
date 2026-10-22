"""Rodzaj pracy, która zawiodła (`jobs.error_task`), obok kodu błędu (ticket 14)

Błąd generowania wraca do szkicu i ma przetrwać dodanie koła (praca workera), więc
pochodzenia błędu nie da się już zgadywać z `jobs.task`. Dotychczasowe błędy powstały
zawsze przy bieżącym `task` (`enqueue` czyścił błąd), więc dostają jego wartość.

Revision ID: t14_error_task
Revises: review_failure_codes
Create Date: 2026-10-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "t14_error_task"
down_revision: str | None = "review_failure_codes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("error_task", sa.String(length=32), nullable=True))
    op.execute("UPDATE jobs SET error_task = task WHERE error IS NOT NULL")


def downgrade() -> None:
    op.drop_column("jobs", "error_task")
