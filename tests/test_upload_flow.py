"""Przepływ przez wszystkie warstwy (prawdziwy Postgres, atrapy SAM 2 i mflux):
wgranie zdjęcia auta i felgi -> klik w koło -> maska jako nakładka -> „Generuj” -> PNG.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from fake_models import BoxDetector, DiscSegmenter, NothingSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import (
    click_wheel,
    drain,
    encode,
    fake_processor,
    fetch,
    generate,
    image_src,
    noise_photo,
    upload,
)
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


def test_home_page_asks_for_car_photo_only(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="pl"' in response.text
    assert TEXTS["upload_heading"] in response.text
    assert 'enctype="multipart/form-data"' in response.text
    assert 'name="photo"' in response.text
    # Felgę wybiera się w kroku 3, po detekcji kół (DESIGN.md, „Strona główna”).
    assert 'id="rim-picker"' not in response.text


def test_click_shows_mask_overlay_and_generation_changes_only_pixels_inside_it(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    original = noise_photo(320, 200)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=40), editor=SolidEditor()
    )
    job_id = upload(client, encode(original), filename="auto.png")
    drain(session_factory, storage)

    page = client.get(f"/jobs/{job_id}")
    assert TEXTS["wheels_none_detected"] in page.text
    assert 'data-role="wheel-picker"' in page.text
    assert f'action="/jobs/{job_id}/generate"' not in page.text  # bez maski nie ma generowania

    segmenting = click_wheel(client, job_id, (120, 90))
    assert TEXTS["status_segmenting"] in segmenting
    assert 'hx-trigger="every 1s"' in segmenting
    assert run_once(session_factory, storage, processor)

    draft = client.get(f"/jobs/{job_id}/status").text
    assert "hx-trigger" not in draft  # czeka na użytkownika, bez pollingu
    overlay = fetch(client, image_src(draft, "mask-overlay"))
    assert overlay.mode == "RGBA" and overlay.size == (320, 200)
    assert f'action="/jobs/{job_id}/generate"' in draft

    generate(client, job_id)
    queued = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_queued"] in queued and 'hx-trigger="every 1s"' in queued
    assert run_once(session_factory, storage, processor)
    assert run_once(session_factory, storage, processor) is False

    done = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_done"] in done and "hx-trigger" not in done
    result = fetch(client, image_src(done, "result"))
    assert result.format == "PNG"
    assert result.size == (320, 200)
    pixels = np.asarray(result.convert("RGB"))
    unmarked = np.asarray(overlay)[..., 3] == 0
    assert np.array_equal(pixels[unmarked], np.asarray(original)[unmarked])
    assert tuple(pixels[90, 120]) == (10, 200, 30)  # środek koła pochodzi z „modelu”


def test_generation_records_used_seed_in_job_parameters(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    drain(session_factory, storage)
    click_wheel(client, job_id)
    assert run_once(session_factory, storage, fake_processor())
    generate(client, job_id)

    params = re.search(r"<dl[^>]*data-params.*?</dl>", client.get(f"/jobs/{job_id}").text, re.S)
    assert params and re.search(
        r'data-param="seed".*?<dd[^>]*>\s*\d+\s*</dd>', params.group(0), re.S
    )


def test_generation_requires_a_wheel_mask(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    drain(session_factory, storage)
    response = client.post(f"/jobs/{job_id}/generate", follow_redirects=False)
    assert response.status_code == 409
    assert TEXTS["error_no_wheel"] in response.text


def test_wheel_cannot_be_clicked_while_mask_is_being_computed(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    drain(session_factory, storage)
    click_wheel(client, job_id)
    response = client.post(f"/jobs/{job_id}/wheels", data={"x": 1, "y": 1})
    assert response.status_code == 409


def test_failed_segmentation_lets_user_click_again(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    drain(session_factory, storage)
    click_wheel(client, job_id)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=NothingSegmenter(), editor=SolidEditor()
    )
    assert run_once(session_factory, storage, processor)

    draft = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["error_mask_failed"] in draft
    assert TEXTS["failure_nothing_at_click"] in draft
    assert 'data-role="wheel-picker"' in draft
    assert 'data-role="mask-overlay"' not in draft
    click_wheel(client, job_id)


def test_failure_in_generation_returns_to_draft_with_error_on_page(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    class BrokenEditor(SolidEditor):
        def edit(self, *args: object, **kwargs: object) -> Image.Image:
            raise RuntimeError("model się wyłożył")

    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=BrokenEditor()
    )
    job_id = upload(client)
    drain(session_factory, storage)
    click_wheel(client, job_id)
    assert run_once(session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    status = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_draft"] in status  # ticket 14: szkic z błędem, nie `failed`
    # Wyjątek biblioteki trafia do logu workera, a UI pokazuje polski komunikat.
    assert TEXTS["failure_internal"] in status
    assert "model się wyłożył" not in status and "RuntimeError" not in status
    assert "hx-trigger" not in status


def test_upload_without_photo_shows_error(client: TestClient) -> None:
    response = client.post("/jobs", files={"photo": ("", b"", "application/octet-stream")})
    assert response.status_code == 400
    assert TEXTS["error_no_photo"] in response.text


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/jobs/999999").status_code == 404


def test_files_outside_data_dir_are_not_served(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("x")
    assert client.get("/files/../secret.txt").status_code == 404
