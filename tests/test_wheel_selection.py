"""Ticket 04: automatyczna detekcja kół po wgraniu zdjęcia, edycja masek i podmiana wielu kół
(przez HTTP i `run_once` workera; prawdziwy Postgres, atrapy detektora, SAM 2 i mflux).
"""

from __future__ import annotations

import io
import re

import numpy as np
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import encode, noise_photo, upload
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor

# Dwa koła na zdjęciu 320x200: boxy z „detektora” (x0, y0, x1, y1).
FRONT = (40.0, 110.0, 120.0, 190.0)
REAR = (200.0, 110.0, 280.0, 190.0)


def _processor(*boxes: tuple[float, float, float, float]) -> WheelSwapProcessor:
    return WheelSwapProcessor(
        detector=BoxDetector(list(boxes)), segmenter=DiscSegmenter(radius=20), editor=SolidEditor()
    )


def _overlays(html: str) -> list[str]:
    return re.findall(r'<img[^>]+src="(/files/[^"]+)"[^>]*data-role="mask-overlay"', html)


def _alpha(client: TestClient, url: str) -> np.ndarray:
    response = client.get(url)
    assert response.status_code == 200
    return np.asarray(Image.open(io.BytesIO(response.content)).convert("RGBA"))[..., 3]


def test_uploaded_photo_gets_detected_wheels_as_mask_overlays(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client, encode(noise_photo(320, 200)), filename="auto.png")

    detecting = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_detecting"] in detecting
    assert 'hx-trigger="every 1s"' in detecting
    assert run_once(session_factory, storage, _processor(FRONT, REAR))

    draft = client.get(f"/jobs/{job_id}/status").text
    assert "hx-trigger" not in draft
    front, rear = (_alpha(client, url) for url in _overlays(draft))
    assert front[150, 80] > 0 and front[150, 240] == 0
    assert rear[150, 240] > 0 and rear[150, 80] == 0
    assert f'action="/jobs/{job_id}/generate"' in draft


def _click(client: TestClient, job_id: int, point: tuple[float, float]) -> str:
    response = client.post(f"/jobs/{job_id}/wheels", data={"x": point[0], "y": point[1]})
    assert response.status_code == 200, response.text
    return response.text


def _detected_job(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    *boxes: tuple[float, float, float, float],
) -> int:
    job_id = upload(client, encode(noise_photo(320, 200)), filename="auto.png")
    assert run_once(session_factory, storage, _processor(*boxes))
    return job_id


def test_click_inside_a_mask_removes_that_wheel_without_the_worker(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _detected_job(client, session_factory, storage, FRONT, REAR)

    draft = _click(client, job_id, (85, 145))  # w masce przedniego koła, nie w środku

    assert "hx-trigger" not in draft  # nic nie czeka na workera
    [rear] = (_alpha(client, url) for url in _overlays(draft))
    assert rear[150, 240] > 0
    assert run_once(session_factory, storage, _processor()) is False


def test_click_outside_masks_adds_a_wheel_and_keeps_the_detected_ones(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _detected_job(client, session_factory, storage, FRONT)

    segmenting = _click(client, job_id, (240, 150))
    assert TEXTS["status_segmenting"] in segmenting
    assert run_once(session_factory, storage, _processor())

    draft = client.get(f"/jobs/{job_id}/status").text
    masks = [_alpha(client, url) for url in _overlays(draft)]
    assert len(masks) == 2
    assert masks[0][150, 80] > 0 and masks[1][150, 240] > 0  # DiscSegmenter: dysk r=20


def test_removing_every_wheel_hides_generation_until_a_wheel_is_added(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _detected_job(client, session_factory, storage, FRONT)

    draft = _click(client, job_id, (80, 150))

    assert _overlays(draft) == []
    assert f'action="/jobs/{job_id}/generate"' not in draft
    assert TEXTS["wheel_click_hint"] in draft
    response = client.post(f"/jobs/{job_id}/generate", follow_redirects=False)
    assert response.status_code == 409


def test_when_no_wheel_is_detected_the_page_says_so_and_asks_for_a_click(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _detected_job(client, session_factory, storage)

    draft = client.get(f"/jobs/{job_id}").text

    assert TEXTS["wheels_none_detected"] in draft
    assert 'data-role="wheel-picker"' in draft
    _click(client, job_id, (240, 150))
    assert run_once(session_factory, storage, _processor())
    assert TEXTS["wheels_none_detected"] not in client.get(f"/jobs/{job_id}/status").text


def test_failed_detection_returns_to_manual_wheel_selection(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    class BrokenDetector(BoxDetector):
        def detect(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
            raise RuntimeError("detektor się wyłożył")

    job_id = upload(client, encode(noise_photo(320, 200)), filename="auto.png")
    processor = WheelSwapProcessor(
        detector=BrokenDetector(), segmenter=DiscSegmenter(radius=20), editor=SolidEditor()
    )
    assert run_once(session_factory, storage, processor)

    draft = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["error_detection_failed"] in draft and "detektor się wyłożył" in draft
    assert 'data-role="wheel-picker"' in draft
    _click(client, job_id, (240, 150))


def test_all_selected_wheels_land_in_one_result_with_same_seed_and_prompt(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    original = noise_photo(320, 200)
    job_id = upload(client, encode(original), filename="auto.png")
    editor = SolidEditor()
    processor = WheelSwapProcessor(
        detector=BoxDetector([FRONT, REAR, (140.0, 10.0, 180.0, 50.0)]),
        segmenter=DiscSegmenter(radius=20),
        editor=editor,
    )
    assert run_once(session_factory, storage, processor)
    _click(client, job_id, (160, 30))  # zbędne „koło” usunięte
    _click(client, job_id, (160, 100))  # koło dodane kliknięciem (dysk r=20)
    assert run_once(session_factory, storage, processor)
    draft = client.get(f"/jobs/{job_id}/status").text
    masks = [_alpha(client, url) > 0 for url in _overlays(draft)]
    assert len(masks) == 3
    assert TEXTS["wheels_selected"].format(count=3) in draft

    response = client.post(f"/jobs/{job_id}/generate", follow_redirects=False)
    assert response.status_code == 303
    assert run_once(session_factory, storage, processor)

    done = client.get(f"/jobs/{job_id}/status").text
    [result_url] = re.findall(r'<img[^>]+src="(/files/[^"]+)"[^>]*data-role="result"', done)
    result = np.asarray(Image.open(io.BytesIO(client.get(result_url).content)).convert("RGB"))
    outside = ~np.logical_or.reduce(masks)
    assert np.array_equal(result[outside], np.asarray(original)[outside])
    for center in [(150, 80), (150, 240), (100, 160)]:
        assert tuple(result[center]) == (10, 200, 30)  # środek każdego koła z „modelu”
    assert tuple(result[30, 160]) == tuple(np.asarray(original)[30, 160])  # usunięte: bez zmian
    assert len(editor.calls) == 3  # każde koło osobnym wywołaniem
    assert len({call.seed for call in editor.calls}) == 1
    assert len({call.prompt for call in editor.calls}) == 1
