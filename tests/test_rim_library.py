"""Biblioteka felg (prawdziwy Postgres, atrapy modeli): dodanie felgi z usunięciem tła,
akceptacja podglądu, wybór z biblioteki przy nowym zdjęciu i usuwanie z biblioteki.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator

import numpy as np
import pytest
from fake_models import CenterDiscRemover
from fastapi.testclient import TestClient
from helpers import encode, noise_photo, rim_processor
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once


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


def _post_rim(client: TestClient, rim: Image.Image, name: str = "") -> str:
    response = client.post(
        "/rims", files={"rim": ("felga.png", encode(rim), "image/png")}, data={"name": name}
    )
    assert response.status_code == 200, response.text
    return response.text


def test_added_rim_has_background_removed_by_worker_and_waits_for_review(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim = noise_photo(100, 80)

    cutting = _post_rim(client, rim, name="BBS CH-R")
    assert TEXTS["rim_cutting"] in cutting
    assert 'hx-trigger="every 1s"' in cutting
    # „Model” uznaje za felgę dysk o promieniu 32 px wokół środka zdjęcia (50, 40).
    assert run_once(session_factory, storage, rim_processor(CenterDiscRemover(radius=32)))

    review = client.get("/rims/picker").text
    assert "hx-trigger" not in review
    assert TEXTS["rim_review_hint"] in review
    rim_id = re.search(r'data-rim-id="(\d+)"', review)
    assert rim_id
    assert f'hx-post="/rims/{rim_id.group(1)}/accept"' in review
    cut = _fetch(client, _image_src(review, "rim-cut"))
    # Wycięta felga: przezroczyste tło, przycięta do felgi (bok 65 px, od (18, 8)).
    assert cut.format == "PNG" and cut.mode == "RGBA" and cut.size == (65, 65)
    alpha = np.asarray(cut)[..., 3]
    assert alpha[0, 0] == 0 and alpha[32, 32] == 255
    assert cut.getpixel((32, 32))[:3] == rim.getpixel((50, 40))


def _rim_id(html: str) -> int:
    match = re.search(r'data-rim-id="(\d+)"', html)
    assert match, html
    return int(match.group(1))


def _library_ids(html: str) -> list[int]:
    return [int(m) for m in re.findall(r'data-library-rim-id="(\d+)"', html)]


def test_accepted_rim_joins_library_and_is_selected_on_home_page(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _post_rim(client, noise_photo(100, 80), name="BBS CH-R")
    assert run_once(session_factory, storage, rim_processor(CenterDiscRemover(radius=32)))
    rim_id = _rim_id(client.get("/rims/picker").text)

    accepted = client.post(f"/rims/{rim_id}/accept", data={"mode": "select"})
    assert accepted.status_code == 200
    assert _library_ids(accepted.text) == [rim_id]
    assert re.search(rf'name="rim_id" value="{rim_id}"[^>]*checked', accepted.text)
    assert "data-rim-id" not in accepted.text  # podgląd zniknął

    home = client.get("/").text
    assert _library_ids(home) == [rim_id]
    assert "BBS CH-R" in home
    assert 'form="job-form"' in home and 'id="job-form"' in home
