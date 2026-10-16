"""Wgrywanie zdjęć przez HTTP: formaty (z HEIC), obrót według EXIF, limity, metadane wyniku."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, encode, fake_processor, generate, jpeg, library_rim, upload
from photo_files import GPS_IFD, MAKE, display_profile, phone_file, phone_heic, quadrants
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
    assert response.status_code == 200, url
    return Image.open(io.BytesIO(response.content))


def test_iphone_heic_is_shown_upright_and_gives_png_with_icc_and_without_gps(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload(client, phone_heic(quadrants(64, 48)), filename="IMG_0001.HEIC")

    page = client.get(f"/jobs/{job_id}").text
    # Przeglądarka nie wyświetli HEIC: wybór koła pokazuje obrócony podgląd w pełnym rozmiarze,
    # więc współrzędne kliknięcia (naturalWidth/Height) są w układzie obróconego zdjęcia.
    preview = _fetch(client, _image_src(page, "input"))
    assert preview.format in {"JPEG", "PNG"} and preview.size == (64, 48)
    top_left = np.asarray(preview.convert("RGB"))[5, 5].astype(int)
    assert np.abs(top_left - (250, 30, 30)).max() < 16
    assert client.get(f"/jobs/{job_id}/thumbnail").status_code == 200

    processor = fake_processor(radius=6)
    drain(session_factory, storage, processor)  # detekcja kół (atrapa: brak)
    click_wheel(client, job_id, (48, 36))
    assert run_once(session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    status = client.get(f"/jobs/{job_id}/status").text
    result = _fetch(client, _image_src(status, "result"))
    assert result.format == "PNG" and result.size == (64, 48)
    assert result.info.get("icc_profile") == display_profile()
    exif = result.getexif()
    assert exif.get(MAKE) == "Apple" and not exif.get_ifd(GPS_IFD)
    assert tuple(np.asarray(result.convert("RGB"))[36, 48]) == (10, 200, 30)
    _fetch(client, _image_src(status, "input"))  # wejście HEIC też ma podgląd po zakończeniu


def test_home_page_accepts_heic_by_picker_drop_and_paste(client: TestClient) -> None:
    page = client.get("/").text
    assert page.count('accept="image/jpeg,image/png,image/webp,image/heic,image/heif') == 2
    assert page.count('data-role="drop-zone"') == 2
    assert "@drop.prevent" in page and "@paste.window" in page
    assert TEXTS["upload_drop_hint"] in page


def _post(client: TestClient, field: str, name: str, data: bytes) -> str:
    """Wgranie złego pliku: zdjęcia auta (`POST /jobs` z felgą z biblioteki) albo nowej felgi
    do biblioteki (`POST /rims`, fragment biblioteki z komunikatem)."""
    upload = (name, data, "application/octet-stream")
    if field == "rim":
        response = client.post("/rims", files={"rim": upload}, follow_redirects=False)
    else:
        response = client.post(
            "/jobs",
            files={"photo": upload},
            data={"rim_id": str(library_rim(client))},
            follow_redirects=False,
        )
    assert response.status_code == 400, response.text
    assert 'role="alert"' in response.text
    return response.text


def _no_jobs(client: TestClient, storage: Storage) -> bool:
    """Nic nie zostało zapisane: brak zadań i plików zadań, brak felg czekających na podgląd."""
    jobs_dir = storage.root / "jobs"
    files = [p for p in jobs_dir.rglob("*") if p.is_file()] if jobs_dir.exists() else []
    pending = "data-rim-id" in client.get("/rims/picker").text
    rims = [
        p for p in storage.root.glob("rims/*/original.*") if not p.with_name("cut.png").exists()
    ]
    return (
        TEXTS["history_empty"] in client.get("/history").text
        and not files
        and not pending
        and not rims
    )


@pytest.mark.parametrize("field", ["photo", "rim"])
@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("animacja.gif", encode(Image.new("RGB", (8, 8)), "GIF")),
        ("skan.tiff", encode(Image.new("RGB", (8, 8)), "TIFF")),
        ("notatka.jpg", b"to nie jest zdjecie"),
    ],
)
def test_unsupported_file_is_rejected_with_message(
    client: TestClient, storage: Storage, field: str, name: str, data: bytes
) -> None:
    page = _post(client, field, name, data)
    assert TEXTS["error_upload_format"].format(name=name) in page
    assert _no_jobs(client, storage)


@pytest.mark.parametrize("field", ["photo", "rim"])
def test_file_over_50_mb_is_rejected_with_message(
    client: TestClient, storage: Storage, field: str
) -> None:
    data = jpeg() + bytes(50 * 1024 * 1024)
    page = _post(client, field, "duze.jpg", data)
    assert TEXTS["error_upload_too_large"].format(name="duze.jpg", limit=50) in page
    assert _no_jobs(client, storage)


@pytest.mark.parametrize("field", ["photo", "rim"])
def test_photo_over_40_megapixels_is_rejected_with_message(
    client: TestClient, storage: Storage, field: str
) -> None:
    data = encode(Image.new("L", (8000, 5001)))  # 40,008 MPx, plik ma kilkadziesiąt kB
    page = _post(client, field, "panorama.png", data)
    assert TEXTS["error_upload_too_many_pixels"].format(name="panorama.png", limit=40) in page
    assert _no_jobs(client, storage)


def test_photo_of_exactly_40_megapixels_is_accepted(client: TestClient) -> None:
    upload(client, encode(Image.new("L", (8000, 5000))), filename="duze.png")


@pytest.mark.parametrize(
    ("format", "name"), [("JPEG", "a.jpg"), ("PNG", "a.png"), ("WEBP", "a.webp")]
)
def test_common_formats_with_exif_orientation_are_shown_upright(
    client: TestClient, format: str, name: str
) -> None:
    job_id = upload(client, phone_file(quadrants(64, 48), format), filename=name)

    page = client.get(f"/jobs/{job_id}").text
    shown = _fetch(client, _image_src(page, "input"))
    # Obrócone według EXIF już na serwerze: nie zależy od tego, jak przeglądarka traktuje EXIF.
    assert shown.size == (64, 48)
    top_left = np.asarray(shown.convert("RGB"))[5, 5].astype(int)
    assert np.abs(top_left - (250, 30, 30)).max() < 16
