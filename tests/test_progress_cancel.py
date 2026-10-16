"""Postęp generowania, anulowanie, awarie workera, „Ponów” i kolejka (ticket 05).

Przez HTTP i `run_once` (prawdziwy Postgres, atrapy SAM 2 i mflux). Atrapa mflux
(`SolidEditor.after_step`) pozwala zajrzeć do UI albo kliknąć „Anuluj” w trakcie kroku.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, encode, fake_processor, generate, noise_photo, upload
from photo_files import phone_heic, quadrants
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import TEXTS
from wheeler_shared.models import Job, Wheel
from wheeler_shared.queue import INTERRUPTED_ERROR, claim_next_job
from wheeler_shared.storage import Storage
from wheeler_worker.runner import recover_interrupted_jobs, run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


@pytest.fixture
def client(database_url: str, storage: Storage) -> Iterator[TestClient]:
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _status(client: TestClient, job_id: int) -> str:
    response = client.get(f"/jobs/{job_id}/status")
    assert response.status_code == 200
    return response.text


def _with_mask(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    processor: WheelSwapProcessor,
) -> int:
    job_id = upload(client)
    drain(session_factory, storage, processor)  # detekcja po wgraniu (atrapa nic nie znajduje)
    click_wheel(client, job_id)
    assert run_once(session_factory, storage, processor)
    return job_id


def test_status_shows_current_wheel_and_denoising_step_while_generating(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor = SolidEditor(steps=3)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
    )
    job_id = _with_mask(client, session_factory, storage, processor)
    seen: list[str] = []
    editor.after_step = lambda step: seen.append(_status(client, job_id))

    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    during_second_step = seen[1]
    progress = TEXTS["progress_step"].format(wheel=1, wheels=1, step=2, steps=3)
    assert progress in during_second_step
    assert 'hx-trigger="every 1s"' in during_second_step
    assert TEXTS["status_done"] in _status(client, job_id)


def test_cancel_during_generation_stops_after_current_step_and_marks_job_cancelled(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor = SolidEditor(steps=3)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
    )
    job_id = _with_mask(client, session_factory, storage, processor)
    cancel = f'hx-post="/jobs/{job_id}/cancel"'
    responses = []

    def cancel_during_second_step(step: int) -> None:
        if step == 1:
            assert cancel in _status(client, job_id)
            responses.append(client.post(f"/jobs/{job_id}/cancel"))

    editor.after_step = cancel_during_second_step
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    assert editor.steps_run == 2  # bieżący krok dokończony, kolejny już nie
    [cancelling] = responses
    assert cancelling.status_code == 200
    assert TEXTS["status_cancelling"] in cancelling.text and cancel not in cancelling.text
    status = _status(client, job_id)
    assert TEXTS["status_cancelled"] in status
    assert "hx-trigger" not in status and 'data-role="result"' not in status


def test_cancelled_queued_job_is_never_processed(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor = SolidEditor()
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
    )
    job_id = _with_mask(client, session_factory, storage, processor)
    generate(client, job_id)

    response = client.post(f"/jobs/{job_id}/cancel")

    assert response.status_code == 200 and TEXTS["status_cancelled"] in response.text
    assert run_once(session_factory, storage, processor) is False
    assert editor.calls == []


def test_finished_job_cannot_be_cancelled(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=SolidEditor()
    )
    job_id = _with_mask(client, session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    response = client.post(f"/jobs/{job_id}/cancel")

    assert response.status_code == 409
    assert TEXTS["status_done"] in response.text


def _worker_dies_during_job(session_factory: sessionmaker[Session]) -> None:
    """Worker pobiera zadanie i ginie w trakcie (np. zabity proces): wynik nie zostaje zapisany."""
    with session_factory.begin() as session:
        assert claim_next_job(session) is not None


def test_generation_interrupted_by_worker_crash_fails_on_restart_and_is_not_retried(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor = SolidEditor()
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
    )
    job_id = _with_mask(client, session_factory, storage, processor)
    generate(client, job_id)
    _worker_dies_during_job(session_factory)
    assert TEXTS["status_running"] in _status(client, job_id)

    recover_interrupted_jobs(session_factory)  # start nowego procesu workera

    status = _status(client, job_id)
    assert TEXTS["status_failed"] in status and INTERRUPTED_ERROR in status
    assert "hx-trigger" not in status
    assert f'action="/jobs/{job_id}/retry"' in status
    assert run_once(session_factory, storage, processor) is False
    assert editor.calls == []


def test_mask_request_interrupted_by_worker_crash_returns_to_wheel_selection(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    drain(session_factory, storage)
    click_wheel(client, job_id)
    _worker_dies_during_job(session_factory)

    recover_interrupted_jobs(session_factory)

    status = _status(client, job_id)
    assert 'data-role="wheel-picker"' in status and "hx-trigger" not in status
    click_wheel(client, job_id)


def test_detection_interrupted_by_worker_crash_returns_to_wheel_selection(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    _worker_dies_during_job(session_factory)  # detekcja kół po wgraniu

    recover_interrupted_jobs(session_factory)

    status = _status(client, job_id)
    assert 'data-role="wheel-picker"' in status and "hx-trigger" not in status
    click_wheel(client, job_id)


def test_job_cancelled_just_before_worker_crash_ends_cancelled(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=SolidEditor()
    )
    job_id = _with_mask(client, session_factory, storage, processor)
    generate(client, job_id)
    _worker_dies_during_job(session_factory)
    client.post(f"/jobs/{job_id}/cancel")

    recover_interrupted_jobs(session_factory)

    assert TEXTS["status_cancelled"] in _status(client, job_id)


class BrokenEditor(SolidEditor):
    """„mflux”, który zapamiętuje wywołanie i się wykłada."""

    def edit(self, *args: Any, **kwargs: Any) -> Image.Image:
        super().edit(*args, **kwargs)
        raise RuntimeError("model się wyłożył")


def _params_block(client: TestClient, job_id: int) -> str:
    match = re.search(r"<dl[^>]*data-params.*?</dl>", client.get(f"/jobs/{job_id}").text, re.S)
    assert match
    return match.group(0)


def _retry(client: TestClient, job_id: int) -> int:
    response = client.post(f"/jobs/{job_id}/retry", follow_redirects=False)
    assert response.status_code == 303, response.text
    match = re.fullmatch(r"/jobs/(\d+)", response.headers["location"])
    assert match
    return int(match.group(1))


def test_retry_creates_new_job_with_same_inputs_and_parameters(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    broken = BrokenEditor()
    job_id = upload(client, encode(noise_photo(64, 48)), filename="auto.png")
    drain(session_factory, storage)
    click_wheel(client, job_id)
    assert run_once(session_factory, storage, fake_processor())
    generate(client, job_id)
    assert run_once(
        session_factory,
        storage,
        WheelSwapProcessor(
            detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=broken
        ),
    )
    assert TEXTS["status_failed"] in _status(client, job_id)

    new_id = _retry(client, job_id)

    assert new_id != job_id
    assert TEXTS["status_queued"] in _status(client, new_id)  # maska jest, od razu do kolejki
    assert TEXTS["status_failed"] in _status(client, job_id)  # oryginał zostaje w historii
    assert _params_block(client, new_id) == _params_block(client, job_id)
    editor = SolidEditor()
    assert run_once(
        session_factory,
        storage,
        WheelSwapProcessor(
            detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
        ),
    )
    [before], [after] = broken.calls, editor.calls
    assert (after.seed, after.prompt, after.size) == (before.seed, before.prompt, before.size)
    assert after.images[0].tobytes() == before.images[0].tobytes()  # ten sam wycinek zdjęcia
    assert after.images[1].tobytes() == before.images[1].tobytes()  # ta sama felga
    assert after.mask.tobytes() == before.mask.tobytes()

    # Nowe zadanie ma własne kopie plików: usunięcie oryginału go nie psuje.
    client.post("/history/delete", data={"ids": [job_id]})
    done = _status(client, new_id)
    assert TEXTS["status_done"] in done
    for src in re.findall(r'src="(/files/[^"]+)"', done):
        assert client.get(src).status_code == 200


def test_cancelled_job_can_be_retried(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _with_mask(client, session_factory, storage, fake_processor())
    generate(client, job_id)
    client.post(f"/jobs/{job_id}/cancel")

    new_id = _retry(client, job_id)

    assert TEXTS["status_queued"] in _status(client, new_id)


def test_retried_heic_job_keeps_browser_preview_of_the_photo(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client, phone_heic(quadrants(64, 48)), filename="IMG_0001.HEIC")
    drain(session_factory, storage)
    click_wheel(client, job_id, (48, 36))
    assert run_once(session_factory, storage, fake_processor(radius=6))
    generate(client, job_id)
    client.post(f"/jobs/{job_id}/cancel")

    new_id = _retry(client, job_id)

    src = re.search(r'<img[^>]+src="(/files/[^"]+)"[^>]*data-role="input"', _status(client, new_id))
    assert src and src.group(1).endswith(".preview.jpg")
    assert client.get(src.group(1)).status_code == 200


def test_retry_keeps_detected_wheels_with_their_source_and_box(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = WheelSwapProcessor(
        detector=BoxDetector([(30.0, 20.0, 50.0, 40.0)]),
        segmenter=DiscSegmenter(radius=10),
        editor=SolidEditor(),
    )
    job_id = upload(client)
    drain(session_factory, storage, processor)
    generate(client, job_id)
    client.post(f"/jobs/{job_id}/cancel")

    new_id = _retry(client, job_id)

    def wheels(job_id: int) -> tuple[int | None, list[tuple[object, ...]]]:
        with session_factory() as session:
            job = session.get_one(Job, job_id)
            rows = session.scalars(select(Wheel).where(Wheel.job_id == job_id).order_by(Wheel.id))
            return job.wheels_detected, [(w.source, w.box, w.click_x) for w in rows]

    assert wheels(new_id) == wheels(job_id)
    detected, [(source, box, _)] = wheels(new_id)
    assert detected == 1 and source == "detected" and box is not None


def test_only_failed_or_cancelled_job_can_be_retried(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    response = client.post(f"/jobs/{job_id}/retry", follow_redirects=False)
    assert response.status_code == 409
    assert client.post("/jobs/999999/retry").status_code == 404


def _position(client: TestClient, job_id: int) -> int | None:
    match = re.search(r'data-queue-position="(\d+)"[^>]*>([^<]*)<', _status(client, job_id))
    if match is None:
        return None
    position = int(match.group(1))
    assert match.group(2).strip() == TEXTS["queue_position"].format(position=position)
    return position


def test_queued_jobs_wait_in_fifo_order_and_status_shows_queue_position(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = fake_processor()
    first, second, third = (
        _with_mask(client, session_factory, storage, processor) for _ in range(3)
    )
    for job_id in (second, first, third):  # kolejność „Generuj”, nie utworzenia
        generate(client, job_id)

    assert [_position(client, j) for j in (second, first, third)] == [1, 2, 3]

    assert run_once(session_factory, storage, processor)
    assert TEXTS["status_done"] in _status(client, second)
    assert _position(client, second) is None
    assert [_position(client, j) for j in (first, third)] == [1, 2]

    client.post(f"/jobs/{first}/cancel")
    assert _position(client, third) == 1
