"""Wspólne kroki testów przepływu: wgranie zdjęć, klik w koło, generowanie (przez HTTP)."""

from __future__ import annotations

import io
import re

import numpy as np
from fake_models import DiscSegmenter, OpaqueRemover, SolidEditor
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.storage import Storage
from wheeler_worker.background import BackgroundRemover
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
    """Prawdziwy procesor podmiany felgi z atrapami SAM 2 i mflux (zielona felga)."""
    return WheelSwapProcessor(segmenter=DiscSegmenter(radius=radius), editor=SolidEditor())


def rim_processor(remover: BackgroundRemover | None = None) -> WheelSwapProcessor:
    """Procesor z atrapą usuwania tła (domyślnie felga zostaje bez zmian)."""
    return WheelSwapProcessor(
        segmenter=DiscSegmenter(), editor=SolidEditor(), remover=remover or OpaqueRemover()
    )


def library_rim(
    client: TestClient,
    rim: bytes | None = None,
    *,
    name: str = "",
    remover: BackgroundRemover | None = None,
) -> int:
    """Dodaje felgę do biblioteki przez API: wgranie, usunięcie tła (worker z atrapą,
    domyślnie felga bez zmian) i akceptacja podglądu. Zwraca id felgi."""
    app = client.app
    response = client.post(
        "/rims",
        files={"rim": ("felga.png", rim if rim is not None else rim_png(), "image/png")},
        data={"name": name, "mode": "manage"},
    )
    assert response.status_code == 200, response.text
    assert run_once(app.state.session_factory, app.state.storage, rim_processor(remover))  # type: ignore[attr-defined]
    pending = re.findall(r'data-rim-id="(\d+)"', client.get("/rims/picker").text)
    assert len(pending) == 1, pending
    response = client.post(f"/rims/{pending[0]}/accept", data={"mode": "manage"})
    assert response.status_code == 200, response.text
    return int(pending[0])


def upload(
    client: TestClient,
    photo: bytes | None = None,
    *,
    filename: str = "auto.jpg",
    rim: bytes | None = None,
    rim_id: int | None = None,
) -> int:
    """Nowe zadanie: zdjęcie auta + felga z biblioteki (`rim_id` albo nowa felga `rim`)."""
    if rim_id is None:
        rim_id = library_rim(client, rim)
    response = client.post(
        "/jobs",
        files={"photo": (filename, photo if photo is not None else jpeg(), "image/jpeg")},
        data={"rim_id": str(rim_id)},
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


def finish_job(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    job_id: int,
    *,
    point: tuple[float, float] = WHEEL_CENTER,
    radius: int = 10,
) -> None:
    """Klik w koło -> maska (worker) -> generowanie (worker): zadanie kończy się wynikiem."""
    processor = fake_processor(radius)
    click_wheel(client, job_id, point)
    assert run_once(session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)
