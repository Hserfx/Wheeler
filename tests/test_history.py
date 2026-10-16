"""Podstrona „Historia”: lista zadań, szczegóły i ręczne usuwanie (prawdziwy Postgres)."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from helpers import click_wheel, finish_job, jpeg, rim_png, upload
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import TEXTS
from wheeler_shared.models import Image as DbImage
from wheeler_shared.models import ImageRole, Job
from wheeler_shared.queue import claim_next_job, create_job
from wheeler_shared.storage import Storage


@pytest.fixture
def client(database_url: str, storage: Storage) -> Iterator[TestClient]:
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _upload(client: TestClient) -> int:
    return upload(client)


def _history_job_ids(html: str) -> list[int]:
    return [int(m) for m in re.findall(r'data-job-id="(\d+)"', html)]


def test_history_lists_jobs_newest_first_with_status_date_and_thumbnail(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    first = _upload(client)
    second = _upload(client)
    finish_job(client, session_factory, storage, first)
    click_wheel(client, second)  # drugie czeka w kolejce na maskę

    page = client.get("/history")
    assert page.status_code == 200
    assert _history_job_ids(page.text) == [second, first]

    item_first = _history_item(page.text, first)
    assert TEXTS["status_done"] in item_first
    assert f'href="/jobs/{first}"' in item_first
    assert re.search(r'<time datetime="\d{4}-\d{2}-\d{2}T', item_first)
    assert f'src="/jobs/{first}/thumbnail"' in item_first

    assert TEXTS["status_segmenting"] in _history_item(page.text, second)


def _history_item(html: str, job_id: int) -> str:
    match = re.search(rf'<li[^>]*data-job-id="{job_id}".*?</li>', html, flags=re.DOTALL)
    assert match, html
    return match.group(0)


def _thumbnail(client: TestClient, job_id: int) -> Image.Image:
    response = client.get(f"/jobs/{job_id}/thumbnail")
    assert response.status_code == 200
    return Image.open(io.BytesIO(response.content))


def test_thumbnail_is_small_and_shows_result_once_job_is_done(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client, jpeg(1600, 1200))

    queued = _thumbnail(client, job_id)
    assert max(queued.size) <= 512
    assert queued.size[0] / queued.size[1] == pytest.approx(4 / 3, rel=0.02)
    center = (queued.size[0] // 2, queued.size[1] // 2)
    r, g, _ = queued.convert("RGB").getpixel(center)  # type: ignore[misc]
    assert r > 150 and g < 80  # czerwone wejście

    finish_job(client, session_factory, storage, job_id, point=(800, 600), radius=200)
    done = _thumbnail(client, job_id)
    r, g, _ = done.convert("RGB").getpixel(center)  # type: ignore[misc]
    assert g > 150 and r < 80  # zielona felga z atrapy modelu w środku koła


def test_thumbnail_of_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/jobs/999999/thumbnail").status_code == 404


def _create_with_params(
    session_factory: sessionmaker[Session], storage: Storage, params: dict[str, object]
) -> int:
    job_dir = storage.new_job_dir()
    storage.write_bytes(f"{job_dir}/input.jpg", jpeg())
    storage.write_bytes(f"{job_dir}/rim.png", rim_png())
    with session_factory.begin() as session:
        return create_job(
            session,
            input_path=f"{job_dir}/input.jpg",
            rim_path=f"{job_dir}/rim.png",
            params=params,
        )


def test_job_details_show_input_result_and_all_used_parameters(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _create_with_params(
        session_factory,
        storage,
        # Klucze dopisywane przez kolejne tickety muszą się pojawić bez zmian w szablonie.
        {"seed": 4242, "prompt_suffix": "chromowana felga", "future_option": [1, 2]},
    )
    finish_job(client, session_factory, storage, job_id)

    page = client.get(f"/jobs/{job_id}")
    assert page.status_code == 200
    assert 'data-role="input"' in page.text
    assert 'data-role="result"' in page.text

    params = re.search(r"<dl[^>]*data-params.*?</dl>", page.text, flags=re.DOTALL)
    assert params, page.text
    for key, value in [
        ("seed", "4242"),
        ("prompt_suffix", "chromowana felga"),
        ("future_option", "[1, 2]"),
    ]:
        assert re.search(
            rf'data-param="{key}".*?<dd[^>]*>\s*{re.escape(value)}\s*</dd>',
            params.group(0),
            flags=re.DOTALL,
        ), (key, params.group(0))


def test_job_details_without_parameters_say_defaults_were_used(client: TestClient) -> None:
    job_id = _upload(client)
    page = client.get(f"/jobs/{job_id}")
    assert TEXTS["job_params_default"] in page.text


def _job_dir(client: TestClient, job_id: int, storage: Storage) -> Path:
    page = client.get(f"/jobs/{job_id}")
    match = re.search(r'src="/files/(jobs/[^/"]+)/', page.text)
    assert match, page.text
    return storage.path(match.group(1))


def test_single_job_is_deleted_after_confirmation_with_all_its_files(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    keep = _upload(client)
    doomed = _upload(client)
    finish_job(client, session_factory, storage, keep)
    finish_job(client, session_factory, storage, doomed)
    assert client.get(f"/jobs/{doomed}/thumbnail").status_code == 200
    doomed_dir = _job_dir(client, doomed, storage)
    keep_dir = _job_dir(client, keep, storage)
    # Pliki, których baza nie zna (np. maski i wycinki z późniejszych ticketów), też znikają.
    (doomed_dir / "wheel-1-mask.png").write_bytes(b"x")
    (doomed_dir / "crops").mkdir()
    (doomed_dir / "crops" / "wheel-1.png").write_bytes(b"x")

    details = client.get(f"/jobs/{doomed}")
    assert f'href="/history/delete?ids={doomed}"' in details.text

    confirm = client.get("/history/delete", params={"ids": [doomed]})
    assert confirm.status_code == 200
    assert TEXTS["delete_confirm_heading"] in confirm.text
    assert f'name="ids" value="{doomed}"' in confirm.text
    assert client.get(f"/jobs/{doomed}").status_code == 200  # samo pytanie niczego nie usuwa
    assert doomed_dir.is_dir()

    response = client.post("/history/delete", data={"ids": [str(doomed)]}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/history")

    assert client.get(f"/jobs/{doomed}").status_code == 404
    assert not doomed_dir.exists()
    assert keep_dir.is_dir() and any(keep_dir.iterdir())
    history = client.get(response.headers["location"])
    assert _history_job_ids(history.text) == [keep]
    assert TEXTS["delete_done"].format(count=1) in history.text


def test_history_has_bulk_selection_form_for_deletion(client: TestClient) -> None:
    job_id = _upload(client)
    page = client.get("/history")
    form = re.search(r'<form[^>]*action="/history/delete"[^>]*>', page.text)
    assert form and 'method="get"' in form.group(0)
    assert re.search(rf'<input[^>]*type="checkbox"[^>]*name="ids"[^>]*value="{job_id}"', page.text)


def test_bulk_delete_removes_selected_jobs_and_skips_running_ones(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    done = _upload(client)
    finish_job(client, session_factory, storage, done)
    running = _upload(client)
    click_wheel(client, running)
    with session_factory.begin() as session:
        claimed = claim_next_job(session)
        assert claimed is not None and claimed.id == running
    queued = _upload(client)
    click_wheel(client, queued)
    untouched = _upload(client)
    dirs = {job: _job_dir(client, job, storage) for job in (done, running, queued, untouched)}

    confirm = client.get("/history/delete", params={"ids": [done, running, queued]})
    assert confirm.status_code == 200
    assert TEXTS["delete_running_skipped"] in confirm.text

    response = client.post("/history/delete", data={"ids": [str(done), str(running), str(queued)]})
    assert response.status_code == 200
    assert _history_job_ids(response.text) == [untouched, running]
    assert TEXTS["delete_done"].format(count=2) in response.text
    assert TEXTS["delete_skipped"].format(count=1) in response.text
    assert not dirs[done].exists() and not dirs[queued].exists()
    assert dirs[running].is_dir() and dirs[untouched].is_dir()


def test_deleting_a_job_keeps_files_outside_its_job_directory(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    # Np. felga z biblioteki użyta w zadaniu: plik jest współdzielony, nie należy do zadania.
    job_id = _upload(client)
    shared_file = storage.write_bytes("rims/abc/rim.png", b"x")
    with session_factory.begin() as session:
        session.add(DbImage(job_id=job_id, role=ImageRole.INPUT, path="rims/abc/rim.png"))

    client.post("/history/delete", data={"ids": [str(job_id)]})

    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert shared_file.is_file()


def test_delete_without_selection_returns_to_history(client: TestClient) -> None:
    response = client.get("/history/delete", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/history"


def test_old_jobs_are_kept_indefinitely(
    database_url: str, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as client:
        old = _upload(client)
        finish_job(client, session_factory, storage, old)
        old_dir = _job_dir(client, old, storage)
    with session_factory.begin() as session:
        job = session.get_one(Job, old)
        job.created_at = job.finished_at = datetime.now(UTC) - timedelta(days=3 * 365)

    # Restart aplikacji (migracje, start) i kolejne zadania niczego nie sprzątają.
    with TestClient(create_app(settings)) as client:
        finish_job(client, session_factory, storage, _upload(client))
        assert old in _history_job_ids(client.get("/history").text)
        assert client.get(f"/jobs/{old}").status_code == 200
    assert any(old_dir.iterdir())


def test_history_is_empty_without_jobs(client: TestClient) -> None:
    page = client.get("/history")
    assert page.status_code == 200
    assert TEXTS["history_empty"] in page.text
