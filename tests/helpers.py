"""Wspólne kroki testów przepływu: wgranie zdjęć, klik w koło, generowanie (przez HTTP).

Po wgraniu zadanie czeka w kolejce na detekcję kół (ticket 04): zanim klikniesz koło, przetwórz
kolejkę (`drain`). Atrapa z `fake_processor` nie wykrywa żadnych kół.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
from fake_models import BoxDetector, DiscSegmenter, OpaqueRemover, SolidEditor
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


def noise_file(path: Path, width: int = 400, height: int = 240) -> Path:
    """Zdjęcie z szumem zapisane w `path` (format z rozszerzenia)."""
    noise_photo(width, height).save(path)
    return path


def image_src(html: str, role: str) -> str:
    """Adres `/files/...` obrazka o danym `data-role` w HTML."""
    match = re.search(rf'<img[^>]+src="(/files/[^"]+)"[^>]*data-role="{role}"', html)
    assert match, html
    return match.group(1)


def fetch(client: TestClient, url: str) -> Image.Image:
    response = client.get(url)
    assert response.status_code == 200, url
    return Image.open(io.BytesIO(response.content))


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


def rim_processor(
    remover: BackgroundRemover | None = None, detector: BoxDetector | None = None
) -> WheelSwapProcessor:
    """Procesor z atrapą usuwania tła (domyślnie felga zostaje bez zmian) i detektora kół
    (domyślnie nic nie znajduje: tło usuwane z całego zdjęcia)."""
    return WheelSwapProcessor(
        detector=detector or BoxDetector(),
        segmenter=DiscSegmenter(),
        editor=SolidEditor(),
        remover=remover or OpaqueRemover(),
    )


def library_rim(
    client: TestClient,
    rim: bytes | None = None,
    *,
    name: str = "",
    remover: BackgroundRemover | None = None,
) -> int:
    """Dodaje felgę do biblioteki przez API: wgranie, usunięcie tła (worker z atrapą,
    domyślnie felga bez zmian) i akceptacja podglądu. Zwraca id felgi.

    Felgi mają pierwszeństwo w kolejce workera, więc `run_once` nie rusza innych zadań."""
    state = client.app.state  # type: ignore[attr-defined]
    response = client.post(
        "/rims",
        files={"rim": ("felga.png", rim if rim is not None else rim_png(), "image/png")},
        data={"name": name, "mode": "manage"},
    )
    assert response.status_code == 200, response.text
    assert run_once(state.session_factory, state.storage, rim_processor(remover))
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
    """Nowe zadanie: zdjęcie auta + od razu felga z biblioteki (`rim_id` albo nowa `rim`)."""
    if rim_id is None:
        rim_id = library_rim(client, rim)
    return upload_photo(client, photo, filename=filename, rim_id=rim_id)


def upload_photo(
    client: TestClient,
    photo: bytes | None = None,
    *,
    filename: str = "auto.jpg",
    rim_id: int | None = None,
) -> int:
    """Nowe zadanie ze zdjęciem auta (krok 1); bez `rim_id` felgę wybiera się przed „Generuj”."""
    response = client.post(
        "/jobs",
        files={"photo": (filename, photo if photo is not None else jpeg(), "image/jpeg")},
        data={} if rim_id is None else {"rim_id": str(rim_id)},
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
