"""Stan modelu w UI (prawdziwy Postgres, atrapy modeli): worker zapisuje stan w bazie,
API pokazuje „uśpiony” / „ładowanie” / „gotowy”.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable

import pytest
from fake_models import Box, BoxDetector, SolidEditor, disc_mask
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, upload
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_shared.worker_state import ModelState
from wheeler_worker.model_lifecycle import LoadObserver, ModelLifecycle, db_reporter
from wheeler_worker.runner import run_forever, run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


@pytest.fixture
def lifecycle_factory(
    session_factory: sessionmaker[Session],
) -> Callable[..., ModelLifecycle]:
    def build(
        idle_timeout: float = 600.0, clock: Callable[[], float] = lambda: 0.0
    ) -> ModelLifecycle:
        return ModelLifecycle(
            idle_timeout=idle_timeout, report=db_reporter(session_factory), clock=clock
        )

    return build


def shown_model_state(client: TestClient) -> str:
    response = client.get("/model-state")
    assert response.status_code == 200
    match = re.search(r'data-model-state="(\w+)"', response.text)
    assert match, response.text
    return match.group(1)


def test_pages_poll_model_state_in_header(client: TestClient) -> None:
    assert 'hx-get="/model-state"' in client.get("/").text


def test_model_state_follows_worker(
    client: TestClient, lifecycle_factory: Callable[..., ModelLifecycle]
) -> None:
    assert shown_model_state(client) == "asleep"  # worker jeszcze nic nie zapisał
    lifecycle = lifecycle_factory()
    assert shown_model_state(client) == "asleep"
    assert TEXTS["model_state_asleep"] in client.get("/model-state").text
    with lifecycle.loading():
        assert shown_model_state(client) == "loading"
    assert shown_model_state(client) == "ready"
    assert TEXTS["model_state_ready"] in client.get("/model-state").text
    lifecycle.release(lambda: None)
    assert shown_model_state(client) == ModelState.ASLEEP.value


class LazyDiscSegmenter:
    """„SAM 2” ładowany leniwie przy pierwszym kliknięciu, przez `LoadObserver` workera."""

    def __init__(self, observer: LoadObserver, on_load: Callable[[], None] = lambda: None):
        self.observer = observer
        self.on_load = on_load
        self.loaded = False
        self.loads = self.releases = 0

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        if not self.loaded:
            with self.observer.loading():
                self.on_load()
                self.loaded = True
                self.loads += 1
        return disc_mask(image.size, point, 10)

    def segment_boxes(self, image: Image.Image, boxes: list[Box]) -> list[Image.Image]:
        return []  # detektor atrapa (`BoxDetector()`) nie znajduje kół

    def segment_rim(
        self, image: Image.Image, wheel_mask: Image.Image
    ) -> list[tuple[Image.Image, float]]:
        return []  # testy stanu modelu generują „bez zmian”: felgi się nie mierzy

    def release(self) -> None:
        if self.loaded:
            self.loaded = False
            self.releases += 1


def test_first_task_shows_model_loading_instead_of_stalled_progress(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    lifecycle_factory: Callable[..., ModelLifecycle],
) -> None:
    lifecycle = lifecycle_factory()
    job_id = upload(client)
    drain(session_factory, storage)  # detekcja kół po wgraniu (atrapa bez modelu)
    click_wheel(client, job_id)
    assert TEXTS["model_loading_job"] not in client.get(f"/jobs/{job_id}/status").text

    seen_while_loading: list[str] = []
    segmenter = LazyDiscSegmenter(
        lifecycle,
        on_load=lambda: seen_while_loading.append(client.get(f"/jobs/{job_id}/status").text),
    )
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=segmenter, editor=SolidEditor()
    )
    assert run_once(session_factory, storage, processor)

    [during_load] = seen_while_loading
    assert 'data-status="running"' in during_load
    assert TEXTS["model_loading_job"] in during_load
    assert TEXTS["model_loading_job"] not in client.get(f"/jobs/{job_id}/status").text
    assert shown_model_state(client) == "ready"


def test_worker_loop_releases_models_after_idle_timeout(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    lifecycle_factory: Callable[..., ModelLifecycle],
) -> None:
    stop = threading.Event()
    now = 0.0

    def clock() -> float:  # każde odczytanie zegara to minuta; bezpiecznik zatrzymuje pętlę
        nonlocal now
        now += 60.0
        if now > 24 * 3600:
            stop.set()
        return now

    lifecycle = lifecycle_factory(idle_timeout=600.0, clock=clock)
    segmenter = LazyDiscSegmenter(lifecycle)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=segmenter, editor=SolidEditor()
    )
    job_id = upload(client)
    drain(session_factory, storage)  # detekcja kół po wgraniu (atrapa bez modelu)
    click_wheel(client, job_id)

    original_release = segmenter.release

    def release_and_stop() -> None:
        original_release()
        stop.set()

    segmenter.release = release_and_stop  # type: ignore[method-assign]
    run_forever(
        session_factory, storage, processor, poll_interval=0.0, stop=stop, lifecycle=lifecycle
    )

    assert segmenter.loads == 1 and segmenter.releases == 1
    assert now - 60.0 < 24 * 3600, "pętla nie zwolniła modelu"
    assert shown_model_state(client) == "asleep"
    assert 'data-status="draft"' in client.get(f"/jobs/{job_id}/status").text
