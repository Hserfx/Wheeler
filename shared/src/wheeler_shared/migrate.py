"""Programowe uruchamianie migracji Alembica (przy starcie API i workera)."""

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def migrate(database_url: str, revision: str = "head") -> None:
    """Doprowadza schemat do `revision`. Bezpieczne przy równoległym wywołaniu (advisory lock)."""
    command.upgrade(alembic_config(database_url), revision)
