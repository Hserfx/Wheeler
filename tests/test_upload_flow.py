"""Wgranie zdjęcia → zadanie w kolejce → worker → wynik na stronie (prawdziwy Postgres)."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.processing import WorkItem, WorkResult, placeholder_processor
from wheeler_worker.runner import run_once


def _jpeg(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client(database_url: str, storage: Storage) -> Iterator[TestClient]:
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _upload(client: TestClient, content: bytes, filename: str = "auto.jpg") -> str:
    response = client.post(
        "/jobs",
        files={"photo": (filename, content, "image/jpeg")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert re.fullmatch(r"/jobs/\d+", location)
    return location


def test_home_page_is_polish_upload_form(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="pl"' in response.text
    assert TEXTS["upload_heading"] in response.text
    assert 'enctype="multipart/form-data"' in response.text


def test_uploaded_photo_is_processed_by_worker_and_result_is_shown(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_url = _upload(client, _jpeg(64, 48))

    page = client.get(job_url)
    assert page.status_code == 200
    assert TEXTS["status_queued"] in page.text
    assert 'hx-trigger="every 1s"' in page.text

    assert run_once(session_factory, storage, placeholder_processor) is True
    assert run_once(session_factory, storage, placeholder_processor) is False

    status = client.get(f"{job_url}/status")
    assert TEXTS["status_done"] in status.text
    assert "hx-trigger" not in status.text  # po zakończeniu polling się zatrzymuje
    match = re.search(r'<img[^>]+src="(/files/[^"]+\.png)"[^>]*data-role="result"', status.text)
    assert match, status.text

    served = client.get(match.group(1))
    assert served.status_code == 200
    result = Image.open(io.BytesIO(served.content))
    assert result.format == "PNG"
    assert result.size == (64, 48)

    # Plik leży na wolumenie danych, a URL odpowiada ścieżce względnej.
    on_disk = storage.root / match.group(1).removeprefix("/files/")
    assert on_disk.is_file()


def test_failure_in_processing_marks_job_failed_with_error_on_page(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    def broken_processor(item: WorkItem) -> WorkResult:
        raise RuntimeError("model się wyłożył")

    job_url = _upload(client, _jpeg())
    assert run_once(session_factory, storage, broken_processor) is True

    status = client.get(f"{job_url}/status")
    assert TEXTS["status_failed"] in status.text
    assert "model się wyłożył" in status.text
    assert "hx-trigger" not in status.text


def test_upload_without_file_shows_error(client: TestClient) -> None:
    response = client.post("/jobs", files={"photo": ("", b"", "application/octet-stream")})
    assert response.status_code == 400
    assert TEXTS["error_no_file"] in response.text


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/jobs/999999").status_code == 404


def test_files_outside_data_dir_are_not_served(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("x")
    assert client.get("/files/../secret.txt").status_code == 404
