"""Silnik i sesje SQLAlchemy oraz wspólne drobiazgi dostępu do bazy."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.orm.interfaces import ORMOption


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def utcnow() -> datetime:
    """Bieżący czas (UTC) do kolumn `DateTime(timezone=True)`."""
    return datetime.now(UTC)


def get_or_raise[Row: Base](
    session: Session,
    model: type[Row],
    key: int,
    *,
    lock: bool = False,
    options: Sequence[ORMOption] = (),
) -> Row:
    """Wiersz `model` odczytany na nowo z bazy (`populate_existing`: sesja może trzymać starą
    wersję), z `lock` zablokowany `FOR UPDATE`. Brak wiersza: `LookupError`."""
    row = session.get(
        model, key, with_for_update=True if lock else None, options=options, populate_existing=True
    )
    if row is None:
        raise LookupError(key)
    return row
