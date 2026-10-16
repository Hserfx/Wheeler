"""Ekran wyniku (ticket 07): suwak przed/po, pobieranie PNG i „ponów to koło".

Przez HTTP (`TestClient`) i `run_once` workera z atrapami modeli, na prawdziwym Postgresie.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator

import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import drain, encode, finish_job, generate, noise_photo, upload
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_shared.storage import Storage
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
    before, after = _image_src(compare.group(0), "input"), _image_src(compare.group(0), "result")
    assert _fetch(client, before).size == _fetch(client, after).size == (320, 200)
    assert 'type="range"' in compare.group(0)  # suwak przed/po
    zooms = re.findall(r'data-role="zoom-wheel"[^>]*data-box="(\d+),(\d+),(\d+),(\d+)"', html)
    # Koła r=20 poszerzone o 3% boku (1 px): prostokąty poszerzonych masek.
    assert [tuple(map(int, box)) for box in zooms] == [(59, 99, 102, 142), (219, 99, 262, 142)]
    assert 'data-role="zoom-reset"' in html
    assert f'href="/jobs/{job_id}/download"' in html
