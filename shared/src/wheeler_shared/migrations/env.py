"""Środowisko Alembica. URL bazy przychodzi z `wheeler_shared.migrate.alembic_config`."""

from alembic import context
from sqlalchemy import create_engine, text

import wheeler_shared.models  # noqa: F401  (rejestracja tabel w metadanych)
from wheeler_shared.db import Base

# Stały klucz blokady doradczej: API i worker mogą migrować jednocześnie.
MIGRATION_LOCK_KEY = 0x5748454C  # "WHEL"

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    url = config.get_main_option("sqlalchemy.url")
    assert url is not None
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(:k)"), {"k": MIGRATION_LOCK_KEY})
            connection.commit()
            try:
                context.configure(connection=connection, target_metadata=target_metadata)
                with context.begin_transaction():
                    context.run_migrations()
                connection.commit()
            finally:
                connection.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": MIGRATION_LOCK_KEY})
                connection.commit()
    finally:
        engine.dispose()


run_migrations_online()
