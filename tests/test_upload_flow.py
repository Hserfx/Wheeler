"""Przepływ przez wszystkie warstwy (prawdziwy Postgres, atrapy SAM 2 i mflux):
wgranie zdjęcia auta i felgi -> klik w koło -> maska jako nakładka -> „Generuj” -> PNG.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fake_models import DiscSegmenter, NothingSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import (
    click_wheel,
    encode,
    fake_processor,
    generate,
    jpeg,
    library_rim,
    noise_photo,
    upload,
)
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


@pytest.fixture
def client(database_url: str, storage: Storage) -> Iterator[TestClient]:
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _image_src(html: str, role: str) -> str:
    match = re.search(rf'<img[^>]+src="(/files/[^"]+)"[^>]*data-role="{role}"', html)
    assert match, html
    return match.group(1)


def _fetch(client: TestClient, url: str) -> Image.Image:
    response = client.get(url)
    assert response.status_code == 200
    return Image.open(io.BytesIO(response.content))


def test_home_page_asks_for_car_photo_and_rim(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="pl"' in response.text
    assert TEXTS["upload_heading"] in response.text
    assert 'enctype="multipart/form-data"' in response.text
    assert 'name="photo"' in response.text
    # Felga: wybór z biblioteki albo dodanie nowej (z usunięciem tła).
    assert 'id="rim-picker"' in response.text and 'hx-post="/rims"' in response.text


def test_click_shows_mask_overlay_and_generation_changes_only_pixels_inside_it(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    original = noise_photo(320, 200)
    processor = WheelSwapProcessor(segmenter=DiscSegmenter(radius=40), editor=SolidEditor())
    job_id = upload(client, encode(original), filename="auto.png")

    page = client.get(f"/jobs/{job_id}")
    assert TEXTS["wheel_click_hint"] in page.text
    assert 'data-role="wheel-picker"' in page.text
    assert f'action="/jobs/{job_id}/generate"' not in page.text  # bez maski nie ma generowania

    segmenting = click_wheel(client, job_id, (120, 90))
    assert TEXTS["status_segmenting"] in segmenting
    assert 'hx-trigger="every 1s"' in segmenting
    assert run_once(session_factory, storage, processor)

    draft = client.get(f"/jobs/{job_id}/status").text
    assert "hx-trigger" not in draft  # czeka na użytkownika, bez pollingu
    overlay = _fetch(client, _image_src(draft, "mask-overlay"))
    assert overlay.mode == "RGBA" and overlay.size == (320, 200)
    assert f'action="/jobs/{job_id}/generate"' in draft

    generate(client, job_id)
    queued = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_queued"] in queued and 'hx-trigger="every 1s"' in queued
    assert run_once(session_factory, storage, processor)
    assert run_once(session_factory, storage, processor) is False

    done = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_done"] in done and "hx-trigger" not in done
    result = _fetch(client, _image_src(done, "result"))
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
    click_wheel(client, job_id)
    assert run_once(session_factory, storage, fake_processor())
    generate(client, job_id)

    params = re.search(r"<dl[^>]*data-params.*?</dl>", client.get(f"/jobs/{job_id}").text, re.S)
    assert params and re.search(
        r'data-param="seed".*?<dd[^>]*>\s*\d+\s*</dd>', params.group(0), re.S
    )


def test_clicking_again_replaces_the_selected_wheel(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = fake_processor(radius=8)
    job_id = upload(client)
    click_wheel(client, job_id, (15, 15))
    assert run_once(session_factory, storage, processor)
    click_wheel(client, job_id, (45, 30))
    assert run_once(session_factory, storage, processor)

    draft = client.get(f"/jobs/{job_id}/status").text
    assert len(re.findall(r'data-role="mask-overlay"', draft)) == 1
    alpha = np.asarray(_fetch(client, _image_src(draft, "mask-overlay")))[..., 3]
    assert alpha[30, 45] > 0 and alpha[15, 15] == 0


def test_generation_requires_a_wheel_mask(client: TestClient) -> None:
    job_id = upload(client)
    response = client.post(f"/jobs/{job_id}/generate", follow_redirects=False)
    assert response.status_code == 409
    assert TEXTS["error_no_wheel"] in response.text


def test_wheel_cannot_be_clicked_while_mask_is_being_computed(client: TestClient) -> None:
    job_id = upload(client)
    click_wheel(client, job_id)
    response = client.post(f"/jobs/{job_id}/wheels", data={"x": 1, "y": 1})
    assert response.status_code == 409


def test_failed_segmentation_lets_user_click_again(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client)
    click_wheel(client, job_id)
    processor = WheelSwapProcessor(segmenter=NothingSegmenter(), editor=SolidEditor())
    assert run_once(session_factory, storage, processor)

    draft = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["error_mask_failed"] in draft
    assert 'data-role="wheel-picker"' in draft
    assert 'data-role="mask-overlay"' not in draft
    click_wheel(client, job_id)


def test_failure_in_generation_marks_job_failed_with_error_on_page(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    class BrokenEditor(SolidEditor):
        def edit(self, *args: object, **kwargs: object) -> Image.Image:
            raise RuntimeError("model się wyłożył")

    processor = WheelSwapProcessor(segmenter=DiscSegmenter(radius=10), editor=BrokenEditor())
    job_id = upload(client)
    click_wheel(client, job_id)
    assert run_once(session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    status = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_failed"] in status
    assert "model się wyłożył" in status
    assert "hx-trigger" not in status


def test_upload_without_photo_shows_error_and_keeps_chosen_rim(client: TestClient) -> None:
    rim_id = library_rim(client)
    response = client.post(
        "/jobs",
        files={"photo": ("", b"", "application/octet-stream")},
        data={"rim_id": str(rim_id)},
    )
    assert response.status_code == 400
    assert TEXTS["error_no_photo"] in response.text
    assert re.search(rf'name="rim_id" value="{rim_id}"[^>]*checked', response.text)


def test_upload_without_rim_from_library_shows_error(client: TestClient) -> None:
    library_rim(client)
    response = client.post("/jobs", files={"photo": ("auto.jpg", jpeg(), "image/jpeg")})
    assert response.status_code == 400
    assert TEXTS["error_no_rim"] in response.text


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/jobs/999999").status_code == 404


def test_files_outside_data_dir_are_not_served(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("x")
    assert client.get("/files/../secret.txt").status_code == 404
