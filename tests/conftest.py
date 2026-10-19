"""Wspólne fixture'y testów.

Testy integracyjne używają prawdziwego PostgreSQL w Podmanie, na maszynie `wheels`.
Każda sesja pytest startuje WŁASNY kontener o unikalnej nazwie (`wheeler-test-<hex>`)
z losowym portem hosta (`-p 127.0.0.1::5432`), więc wiele worktree może równolegle
uruchamiać `just test`. Kontener jest usuwany po sesji. Każdy test dostaje świeżą bazę
(`CREATE DATABASE`), na której wykonywane są migracje Alembica.

Zmienne środowiskowe:
- `WHEELER_PODMAN_CONNECTION`: połączenie Podmana (domyślnie `wheels`),
- `WHEELER_TEST_PG_ADMIN_URL`: użyj istniejącego serwera zamiast startować kontener
  (URL do bazy administracyjnej, np. `postgresql+psycopg://u:p@127.0.0.1:5555/postgres`).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_shared.db import make_engine, make_session_factory
from wheeler_shared.migrate import migrate
from wheeler_shared.storage import Storage

POSTGRES_IMAGE = "docker.io/library/postgres:17-alpine"
PG_USER = "wheeler"
PG_PASSWORD = "wheeler"
REPO_ROOT = Path(__file__).resolve().parent.parent
# Etykieta pozwala `just test-clean` usuwać tylko kontenery z TEGO worktree.
WORKTREE_LABEL = "wheeler.test.worktree=" + hashlib.sha1(str(REPO_ROOT).encode()).hexdigest()[:12]


def _podman(*args: str) -> str:
    connection = os.environ.get("WHEELER_PODMAN_CONNECTION", "wheels")
    result = subprocess.run(
        ["podman", "--connection", connection, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _wait_for_postgres(conninfo: str, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with psycopg.connect(conninfo, connect_timeout=2) as conn:
                conn.execute("SELECT 1")
            return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture(scope="session")
def pg_admin_url() -> Iterator[str]:
    """URL (SQLAlchemy) do bazy `postgres` na serwerze testowym."""
    if override := os.environ.get("WHEELER_TEST_PG_ADMIN_URL"):
        yield override
        return

    name = f"wheeler-test-{uuid.uuid4().hex[:12]}"
    _podman(
        "run", "-d", "--rm",
        "--name", name,
        "--label", "wheeler.test=1",
        "--label", WORKTREE_LABEL,
        "-e", f"POSTGRES_USER={PG_USER}",
        "-e", f"POSTGRES_PASSWORD={PG_PASSWORD}",
        "-e", "POSTGRES_DB=postgres",
        "-p", "127.0.0.1::5432",
        POSTGRES_IMAGE,
        "-c", "fsync=off",
        "-c", "full_page_writes=off",
    )  # fmt: skip
    try:
        host_port = _podman("port", name, "5432").splitlines()[0].rsplit(":", 1)[1]
        _wait_for_postgres(
            f"host=127.0.0.1 port={host_port} user={PG_USER} password={PG_PASSWORD} dbname=postgres"
        )
        yield f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{host_port}/postgres"
    finally:
        subprocess.run(
            ["podman", "--connection", os.environ.get("WHEELER_PODMAN_CONNECTION", "wheels"),
             "rm", "-f", "-t", "0", name],
            capture_output=True,
            check=False,
        )  # fmt: skip


def _admin_conninfo(admin_url: str) -> str:
    url = make_url(admin_url)
    return (
        f"host={url.host} port={url.port} user={url.username} "
        f"password={url.password} dbname={url.database}"
    )


@pytest.fixture
def database_url(pg_admin_url: str) -> Iterator[str]:
    """Świeża, zmigrowana baza na każdy test."""
    db_name = f"t_{uuid.uuid4().hex[:16]}"
    conninfo = _admin_conninfo(pg_admin_url)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{db_name}"')
    url = make_url(pg_admin_url).set(database=db_name).render_as_string(hide_password=False)
    migrate(url)
    yield url
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')


@pytest.fixture
def session_factory(database_url: str) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(database_url)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "data")


@pytest.fixture
def client(database_url: str, storage: Storage) -> Iterator[TestClient]:
    """API na świeżej bazie testu, z katalogiem danych `storage`."""
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client
