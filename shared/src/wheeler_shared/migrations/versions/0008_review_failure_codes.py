"""Kody niepowodzeń zamiast tekstów w `jobs.error` i `rims.error` (poprawki po review)

Dotąd w kolumnach `error` leżały gotowe teksty (po polsku albo surowe wyjątki bibliotek
po angielsku). Teraz leży tam kod `wheeler_shared.failures.Failure`, a tekst dla użytkownika
pochodzi z `texts.py`. Znane dawne teksty dostają swoje kody, pozostałe `internal`.

Revision ID: review_failure_codes
Revises: t07_wheel_seed
Create Date: 2026-10-19
"""

from collections.abc import Sequence

from alembic import op

revision: str = "review_failure_codes"
down_revision: str | None = "t07_wheel_seed"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CODES = (
    "internal",
    "interrupted",
    "legacy_job",
    "nothing_at_click",
    "empty_mask",
    "rim_not_found",
    "no_rim",
    "unsupported_photo",
)
# (wzorzec LIKE dawnego tekstu, kod); pierwsze dopasowanie wygrywa.
LEGACY = (
    ("Przerwane: worker%", "interrupted"),
    ("Zadanie z poprzedniej wersji aplikacji%", "legacy_job"),
    ("%Nie udało się wyznaczyć maski koła%", "nothing_at_click"),
    ("%SAM 2 nie znalazł obiektu%", "nothing_at_click"),
    ("%Pusta maska koła%", "empty_mask"),
    ("%Nie znaleziono felgi na zdjęciu%", "rim_not_found"),
    ("%Zadanie nie ma felgi referencyjnej%", "no_rim"),
)


def upgrade() -> None:
    codes = ", ".join(f"'{code}'" for code in CODES)
    cases = " ".join(f"WHEN error LIKE '{pattern}' THEN '{code}'" for pattern, code in LEGACY)
    for table in ("jobs", "rims"):
        op.execute(
            f"UPDATE {table} SET error = CASE {cases} ELSE 'internal' END "
            f"WHERE error IS NOT NULL AND error NOT IN ({codes})"
        )


def downgrade() -> None:
    # Kody są poprawnym tekstem także dla starszej wersji aplikacji; dawnych tekstów nie da
    # się odtworzyć.
    pass
