"""Migracja `review_failure_codes`: dawne teksty błędów w bazie -> kody `Failure`."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from wheeler_shared.db import make_engine
from wheeler_shared.failures import Failure
from wheeler_shared.migrate import migrate

BEFORE = "t07_wheel_seed"


@pytest.fixture
def empty_database_url(pg_admin_url: str) -> Iterator[str]:
    url = make_url(pg_admin_url)
    conninfo = (
        f"host={url.host} port={url.port} user={url.username} "
        f"password={url.password} dbname={url.database}"
    )
    name = f"t_{uuid.uuid4().hex[:16]}"
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    yield url.set(database=name).render_as_string(hide_password=False)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_legacy_error_texts_become_failure_codes(empty_database_url: str) -> None:
    migrate(empty_database_url, BEFORE)
    legacy_jobs = {
        "Przerwane: worker został zatrzymany albo uległ awarii w trakcie zadania.": (
            Failure.INTERRUPTED
        ),
        "Zadanie z poprzedniej wersji aplikacji (bez felgi i maski koła).": Failure.LEGACY_JOB,
        "ValueError: Nie udało się wyznaczyć maski koła w klikniętym miejscu": (
            Failure.NOTHING_AT_CLICK
        ),
        "ValueError: SAM 2 nie znalazł obiektu w miejscu kliknięcia": Failure.NOTHING_AT_CLICK,
        "ValueError: Pusta maska koła": Failure.EMPTY_MASK,
        "RuntimeError: model się wyłożył": Failure.INTERNAL,
    }
    legacy_rims = {
        "ValueError: Nie znaleziono felgi na zdjęciu": Failure.RIM_NOT_FOUND,
        "OSError: cannot identify image file": Failure.INTERNAL,
    }
    engine = make_engine(empty_database_url)
    try:
        with engine.begin() as conn:
            for error in legacy_jobs:
                conn.execute(
                    text("INSERT INTO jobs (status, params, error) VALUES ('failed', '{}', :e)"),
                    {"e": error},
                )
            conn.execute(text("INSERT INTO jobs (status, params) VALUES ('done', '{}')"))
            for error in legacy_rims:
                conn.execute(
                    text(
                        "INSERT INTO rims (status, original_path, error) "
                        "VALUES ('failed', 'rims/x/original.png', :e)"
                    ),
                    {"e": error},
                )

        migrate(empty_database_url)

        with engine.connect() as conn:
            jobs = conn.execute(text("SELECT error FROM jobs ORDER BY id")).scalars().all()
            rims = conn.execute(text("SELECT error FROM rims ORDER BY id")).scalars().all()
    finally:
        engine.dispose()
    assert jobs == [*legacy_jobs.values(), None]
    assert rims == list(legacy_rims.values())


def test_existing_errors_get_the_kind_of_work_that_failed(empty_database_url: str) -> None:
    migrate(empty_database_url, "review_failure_codes")
    engine = make_engine(empty_database_url)
    try:
        with engine.begin() as conn:
            for status, task, error in [
                ("draft", "segment", "nothing_at_click"),
                ("draft", "detect", "internal"),
                ("done", "regenerate_wheel", "internal"),
                ("failed", "generate", "interrupted"),
                ("done", "generate", None),
            ]:
                conn.execute(
                    text(
                        "INSERT INTO jobs (status, task, params, error) "
                        "VALUES (:s, :t, '{}', :e)"
                    ),
                    {"s": status, "t": task, "e": error},
                )

        migrate(empty_database_url)

        with engine.connect() as conn:
            kinds = conn.execute(text("SELECT error_task FROM jobs ORDER BY id")).scalars().all()
    finally:
        engine.dispose()
    assert kinds == ["segment", "detect", "regenerate_wheel", "generate", None]
