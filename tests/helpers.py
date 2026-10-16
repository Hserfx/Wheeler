"""Wspólne kroki testów przepływu: wgranie zdjęć, klik w koło, generowanie (przez HTTP).

Po wgraniu zadanie czeka w kolejce na detekcję kół (ticket 04): zanim klikniesz koło, przetwórz
kolejkę (`drain`). Atrapa z `fake_processor` nie wykrywa żadnych kół.
"""

from __future__ import annotations

import io
import re

import numpy as np
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor

WHEEL_CENTER = (40.0, 30.0)  # środek koła na domyślnym zdjęciu 64x48


def encode(image: Image.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def noise_photo(width: int = 64, height: int = 48) -> Image.Image:
    rng = np.random.default_rng(2)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def jpeg(width: int = 64, height: int = 48, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    return encode(Image.new("RGB", (width, height), color), "JPEG")


def rim_png() -> bytes:
    return encode(Image.new("RGB", (40, 40), (180, 180, 190)))


def fake_processor(radius: int = 10) -> WheelSwapProcessor:
    """Prawdziwy procesor podmiany felgi z atrapami detektora (nic nie znajduje), SAM 2
    i mflux (zielona felga)."""
    return WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=radius), editor=SolidEditor()
    )


def upload(
    client: TestClient,
    photo: bytes | None = None,
    *,
    filename: str = "auto.jpg",
    rim: bytes | None = None,
) -> int:
    response = client.post(
        "/jobs",
        files={
            "photo": (filename, photo if photo is not None else jpeg(), "image/jpeg"),
            "rim": ("felga.png", rim if rim is not None else rim_png(), "image/png"),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    match = re.fullmatch(r"/jobs/(\d+)", response.headers["location"])
    assert match
    return int(match.group(1))


def click_wheel(client: TestClient, job_id: int, point: tuple[float, float] = WHEEL_CENTER) -> str:
    response = client.post(f"/jobs/{job_id}/wheels", data={"x": point[0], "y": point[1]})
    assert response.status_code == 200, response.text
    return response.text


def generate(client: TestClient, job_id: int) -> None:
    response = client.post(f"/jobs/{job_id}/generate", follow_redirects=False)
    assert response.status_code == 303, response.text


def drain(
    session_factory: sessionmaker[Session],
    storage: Storage,
    processor: WheelSwapProcessor | None = None,
) -> int:
    """Przetwarza kolejkę do końca (np. detekcję po wgraniu); zwraca liczbę zadań."""
    processor = processor or fake_processor()
    count = 0
    while run_once(session_factory, storage, processor):
        count += 1
    return count


def finish_job(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    job_id: int,
    *,
    point: tuple[float, float] = WHEEL_CENTER,
    radius: int = 10,
) -> None:
    """(Detekcja) -> klik w koło -> maska -> generowanie: zadanie kończy się wynikiem.

    Przetwarza całą kolejkę, więc inne oczekujące zadania też posuwają się o krok."""
    processor = fake_processor(radius)
    drain(session_factory, storage, processor)
    click_wheel(client, job_id, point)
    assert drain(session_factory, storage, processor)
    generate(client, job_id)
    assert drain(session_factory, storage, processor)
