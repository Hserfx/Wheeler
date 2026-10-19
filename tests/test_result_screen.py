"""Ekran wyniku (ticket 07): suwak przed/po, pobieranie PNG i „ponów to koło".

Przez HTTP (`TestClient`) i `run_once` workera z atrapami modeli, na prawdziwym Postgresie.
"""

from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import (
    drain,
    encode,
    fake_processor,
    fetch,
    finish_job,
    generate,
    image_src,
    noise_photo,
    upload,
)
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.models import Wheel
from wheeler_shared.queue import INTERRUPTED_ERROR, claim_next_job
from wheeler_shared.storage import Storage
from wheeler_worker.runner import recover_interrupted_jobs, run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


def test_download_returns_png_attachment_in_original_resolution(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client, encode(noise_photo(320, 200), "JPEG"))
    finish_job(client, session_factory, storage, job_id, point=(120, 90), radius=30)

    response = client.get(f"/jobs/{job_id}/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"].startswith("attachment")
    assert f"wheeler-{job_id}.png" in response.headers["content-disposition"]
    with Image.open(io.BytesIO(response.content)) as result:
        assert result.format == "PNG" and result.size == (320, 200)


def test_download_of_unfinished_job_is_404(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    assert client.get(f"/jobs/{job_id}/download").status_code == 404


def _done_job(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    wheels: list[tuple[float, float]],
    *,
    radius: int = 20,
) -> int:
    """Zadanie 320x200 z kołami wykrytymi w zadanych punktach, wygenerowane do końca."""
    detector = BoxDetector([(x - radius, y - radius, x + radius, y + radius) for x, y in wheels])
    processor = WheelSwapProcessor(
        detector=detector, segmenter=DiscSegmenter(), editor=SolidEditor()
    )
    job_id = upload(client, encode(noise_photo(320, 200)), filename="auto.png")
    drain(session_factory, storage, processor)
    generate(client, job_id)
    assert drain(session_factory, storage, processor)
    return job_id


def test_result_screen_compares_before_and_after_and_zooms_to_each_wheel(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _done_job(client, session_factory, storage, [(80, 120), (240, 120)])

    html = client.get(f"/jobs/{job_id}/status").text

    compare = re.search(r'<div[^>]*data-role="compare".*?</div>\s*<!-- /compare -->', html, re.S)
    assert compare, html
    before, after = image_src(compare.group(0), "input"), image_src(compare.group(0), "result")
    assert fetch(client, before).size == fetch(client, after).size == (320, 200)
    assert 'type="range"' in compare.group(0)  # suwak przed/po
    zooms = re.findall(r'data-role="zoom-wheel"[^>]*data-box="(\d+),(\d+),(\d+),(\d+)"', html)
    # Koła r=20 poszerzone o 3% boku (1 px): prostokąty poszerzonych masek.
    assert [tuple(map(int, box)) for box in zooms] == [(59, 99, 102, 142), (219, 99, 262, 142)]
    assert 'data-role="zoom-reset"' in html
    assert f'href="/jobs/{job_id}/download"' in html


def _wheel_ids(html: str) -> list[int]:
    return [int(i) for i in re.findall(r'<li[^>]*data-wheel-id="(\d+)"', html)]


def _mask_of(session_factory: sessionmaker[Session], storage: Storage, wheel_id: int) -> np.ndarray:
    """Maska koła: dokładnie obszar, który jego generowanie może zmienić."""
    with session_factory() as session:
        mask_path = session.get_one(Wheel, wheel_id).mask_path
    assert mask_path is not None
    return np.asarray(Image.open(storage.path(mask_path)).convert("L"))


def _regenerate(client: TestClient, job_id: int, wheel_id: int) -> None:
    response = client.post(f"/jobs/{job_id}/wheels/{wheel_id}/regenerate", follow_redirects=False)
    assert response.status_code == 303, response.text
    assert response.headers["location"] == f"/jobs/{job_id}"


def test_regenerating_one_wheel_replaces_only_that_wheel_in_the_result(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _done_job(client, session_factory, storage, [(80, 120), (240, 120)])
    done = client.get(f"/jobs/{job_id}/status").text
    first, second = _wheel_ids(done)
    previous_src = image_src(done, "result")
    previous = np.asarray(fetch(client, previous_src).convert("RGB"))
    job_seed = int(re.findall(r'data-role="wheel-seed"[^>]*>\s*(\d+)', done)[1])

    _regenerate(client, job_id, second)
    queued = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_regenerating"] in queued and 'hx-trigger="every 1s"' in queued
    editor = SolidEditor(color=(200, 0, 200))
    processor = WheelSwapProcessor(detector=BoxDetector(), segmenter=DiscSegmenter(), editor=editor)
    assert drain(session_factory, storage, processor) == 1

    done = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_done"] in done and "hx-trigger" not in done
    [call] = editor.calls
    assert call.seed != job_seed
    seeds = [int(s) for s in re.findall(r'data-role="wheel-seed"[^>]*>\s*(\d+)', done)]
    assert seeds == [job_seed, call.seed]  # nowy seed widać przy ponowionym kole
    assert image_src(done, "result") != previous_src
    downloaded = client.get(f"/jobs/{job_id}/download")
    result = np.asarray(Image.open(io.BytesIO(downloaded.content)).convert("RGB"))
    assert result.shape == previous.shape == (200, 320, 3)
    outside = _mask_of(session_factory, storage, second) == 0
    assert np.array_equal(result[outside], previous[outside])  # łącznie z kołem nr 1
    assert tuple(result[120, 240]) == (200, 0, 200)
    assert tuple(result[120, 80]) == (10, 200, 30)
    assert _mask_of(session_factory, storage, first)[120, 80] > 0


def _result_bytes(client: TestClient, job_id: int) -> bytes:
    response = client.get(f"/jobs/{job_id}/download")
    assert response.status_code == 200
    return response.content


def test_failed_wheel_regeneration_keeps_previous_result_and_says_so(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    class BrokenEditor(SolidEditor):
        def edit(self, *args: Any, **kwargs: Any) -> Image.Image:
            raise RuntimeError("model się wyłożył")

    job_id = _done_job(client, session_factory, storage, [(80, 120)])
    [wheel_id] = _wheel_ids(client.get(f"/jobs/{job_id}/status").text)
    previous = _result_bytes(client, job_id)

    _regenerate(client, job_id, wheel_id)
    broken = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(), editor=BrokenEditor()
    )
    assert drain(session_factory, storage, broken) == 1

    status = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_done"] in status and "hx-trigger" not in status
    assert TEXTS["error_regenerate_failed"] in status and "model się wyłożył" in status
    assert _result_bytes(client, job_id) == previous
    assert 'data-role="compare"' in status  # można ponowić jeszcze raz
    _regenerate(client, job_id, wheel_id)


def test_cancelled_or_interrupted_wheel_regeneration_returns_to_previous_result(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _done_job(client, session_factory, storage, [(80, 120)])
    [wheel_id] = _wheel_ids(client.get(f"/jobs/{job_id}/status").text)
    previous = _result_bytes(client, job_id)
    editor = SolidEditor(color=(200, 0, 200))
    processor = WheelSwapProcessor(detector=BoxDetector(), segmenter=DiscSegmenter(), editor=editor)

    _regenerate(client, job_id, wheel_id)
    assert 'data-role="cancel"' in client.get(f"/jobs/{job_id}/status").text
    cancelled = client.post(f"/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert TEXTS["status_done"] in cancelled.text and "hx-trigger" not in cancelled.text
    assert drain(session_factory, storage, processor) == 0

    _regenerate(client, job_id, wheel_id)
    with session_factory.begin() as session:  # worker pobiera zadanie i ginie w trakcie
        assert claim_next_job(session) is not None
    recover_interrupted_jobs(session_factory)  # start nowego procesu workera

    status = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_done"] in status and INTERRUPTED_ERROR in status
    assert _result_bytes(client, job_id) == previous
    assert editor.calls == []


def test_regeneration_progress_is_shown_for_the_single_wheel(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _done_job(client, session_factory, storage, [(80, 120), (240, 120)])
    _, wheel_id = _wheel_ids(client.get(f"/jobs/{job_id}/status").text)
    seen: list[str] = []
    editor = SolidEditor(
        steps=2, after_step=lambda step: seen.append(client.get(f"/jobs/{job_id}/status").text)
    )
    processor = WheelSwapProcessor(detector=BoxDetector(), segmenter=DiscSegmenter(), editor=editor)

    _regenerate(client, job_id, wheel_id)
    assert drain(session_factory, storage, processor) == 1

    expected = TEXTS["progress_step"].format(wheel=1, wheels=1, step=1, steps=2)
    assert TEXTS["status_regenerating"] in seen[0] and expected in seen[0]


def test_wheel_can_be_regenerated_only_in_finished_result(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    processor = WheelSwapProcessor(
        detector=BoxDetector([(20, 10, 60, 50)]), segmenter=DiscSegmenter(), editor=SolidEditor()
    )
    drain(session_factory, storage, processor)
    draft = client.get(f"/jobs/{job_id}/status").text
    [wheel_id] = re.findall(r'data-role="mask-overlay" data-wheel-id="(\d+)"', draft)

    response = client.post(f"/jobs/{job_id}/wheels/{wheel_id}/regenerate")
    assert response.status_code == 409
    assert TEXTS["error_not_regeneratable"] in response.text
    other = _done_job(client, session_factory, storage, [(80, 120)])
    assert client.post(f"/jobs/{other}/wheels/{wheel_id}/regenerate").status_code == 404
    [own] = _wheel_ids(client.get(f"/jobs/{other}/status").text)
    _regenerate(client, other, own)
    assert client.post(f"/jobs/{other}/wheels/{own}/regenerate").status_code == 409  # już w toku


def test_wheel_regeneration_waits_behind_interactive_work_like_generation(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _done_job(client, session_factory, storage, [(80, 120)])
    [wheel_id] = _wheel_ids(client.get(f"/jobs/{job_id}/status").text)
    _regenerate(client, job_id, wheel_id)
    later = upload(client)  # detekcja kół zlecona po ponowieniu koła

    assert 'data-queue-position="2"' in client.get(f"/jobs/{job_id}/status").text
    assert run_once(session_factory, storage, fake_processor())
    assert TEXTS["status_draft"] in client.get(f"/jobs/{later}/status").text
    assert TEXTS["status_regenerating"] in client.get(f"/jobs/{job_id}/status").text
