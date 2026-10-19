"""Gwarancja „poza maską bit w bit” dla każdego przyjmowanego trybu pikseli (review, spec 2).

Wynik zachowuje tryb oryginału (RGB, RGBA, L, LA): alfa nie zmienia się nigdzie, a kolor
poza maską jest identyczny. Paleta jest rozwijana do RGB (z przezroczystością do RGBA),
obraz 1-bitowy do L, więc odczytane wartości pikseli poza maską też są identyczne. Trybów,
których PNG wyniku nie odda wiernie (16 bitów na kanał, CMYK), nie przyjmujemy przy wgraniu.
"""

from __future__ import annotations

import io
import struct
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import encode, fetch, finish_job, noise_photo, upload
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.processing import GenerateItem, RegenerateWheelItem, SegmentItem
from wheeler_worker.wheel_swap import WheelSwapProcessor

SIZE = (200, 120)
CLICK = (90.0, 60.0)


def _noise_l(seed: int = 3) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (SIZE[1], SIZE[0]), dtype=np.uint8))


def _rgba() -> Image.Image:
    image = noise_photo(*SIZE).convert("RGBA")
    image.putalpha(_noise_l(4))
    return image


def _la() -> Image.Image:
    image = _noise_l().convert("LA")
    image.putalpha(_noise_l(5))
    return image


def _palette(transparent: bool) -> Callable[[], Image.Image]:
    def make() -> Image.Image:
        image = noise_photo(*SIZE).quantize(64)
        if transparent:
            image.info["transparency"] = 0
        return image

    return make


# (tryb wejścia, obraz, tryb wyniku)
MODES = [
    ("RGB", lambda: noise_photo(*SIZE), "RGB"),
    ("RGBA", _rgba, "RGBA"),
    ("L", _noise_l, "L"),
    ("LA", _la, "LA"),
    ("P", _palette(False), "RGB"),
    ("P+tRNS", _palette(True), "RGBA"),
    ("1", lambda: _noise_l().convert("1"), "L"),
]


def _processor() -> WheelSwapProcessor:
    return WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=30), editor=SolidEditor()
    )


def _assert_same_outside(result: Image.Image, original: Image.Image, mask: Path) -> None:
    outside = np.asarray(Image.open(mask).convert("L")) == 0
    expected = np.asarray(original.convert(result.mode))
    pixels = np.asarray(result)
    assert np.array_equal(pixels[outside], expected[outside])
    assert not np.array_equal(pixels, expected)  # koło się zmieniło
    if "A" in result.getbands():  # przezroczystość zostaje wszędzie, także w kole
        assert np.array_equal(pixels[..., -1], expected[..., -1])


@pytest.mark.parametrize(("name", "make", "mode"), MODES, ids=[m[0] for m in MODES])
def test_result_keeps_mode_and_pixels_outside_mask(
    tmp_path: Path, name: str, make: Callable[[], Image.Image], mode: str
) -> None:
    photo = tmp_path / "input.png"
    make().save(photo)
    rim = tmp_path / "rim.png"
    Image.new("RGB", (40, 40), (220, 20, 20)).save(rim)
    processor = _processor()
    segmented = processor.segment(
        SegmentItem(job_id=1, photo_path=photo, output_dir=tmp_path, params={}, clicks={7: CLICK})
    )
    mask = segmented.masks[7].mask

    generated = processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=rim,
            output_dir=tmp_path,
            params={"seed": 1},
            masks={7: mask},
        )
    )
    regenerated = processor.regenerate_wheel(
        RegenerateWheelItem(
            job_id=1,
            photo_path=photo,
            result_path=generated.result_path,
            rim_path=rim,
            output_dir=tmp_path,
            params={"seed": 1},
            wheel_id=7,
            mask_path=mask,
            seed=2,
        )
    )

    with Image.open(photo) as original:
        original.load()
    for path in (generated.result_path, regenerated.result_path):
        with Image.open(path) as result:
            assert result.mode == mode
            _assert_same_outside(result, original, mask)


def test_transparent_png_keeps_its_alpha_through_the_whole_app(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    original = _rgba().resize((64, 48))
    job_id = upload(client, encode(original), filename="auto.png")
    finish_job(client, session_factory, storage, job_id)

    result = fetch(client, f"/jobs/{job_id}/download")
    assert result.mode == "RGBA"
    assert np.array_equal(np.asarray(result)[..., 3], np.asarray(original)[..., 3])


def _png_rgb16(width: int = 64, height: int = 48) -> bytes:
    """PNG z 16 bitami na kanał RGB (Pillow nie zapisze takiego, a czyta go jako 8-bitowy)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    row = b"\0" + b"\x9c\x40\x10\x00\xff\xff" * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(row * height))
        + chunk(b"IEND", b"")
    )


def test_png_with_16_bits_per_channel_is_rejected_on_upload(
    client: TestClient, storage: Storage
) -> None:
    assert Image.open(io.BytesIO(_png_rgb16())).mode == "RGB"  # Pillow obcina do 8 bitów
    response = client.post(
        "/jobs",
        files={"photo": ("auto.png", _png_rgb16(), "image/png")},
    )
    assert response.status_code == 400
    assert TEXTS["error_upload_mode"].format(name="auto.png") in response.text


@pytest.mark.parametrize(
    ("image", "format"),
    [
        (Image.fromarray(np.full((48, 64), 40000, dtype=np.uint16)), "PNG"),  # 16 bitów
        (Image.new("CMYK", (64, 48), (0, 120, 200, 10)), "JPEG"),
    ],
    ids=["png-16bit", "jpeg-cmyk"],
)
def test_photo_that_cannot_be_kept_bit_exact_is_rejected_on_upload(
    client: TestClient, storage: Storage, image: Image.Image, format: str
) -> None:
    response = client.post(
        "/jobs",
        files={"photo": ("auto.img", encode(image, format), "application/octet-stream")},
    )
    assert response.status_code == 400
    assert TEXTS["error_upload_mode"].format(name="auto.img") in response.text
    assert list(storage.root.glob("jobs/*/*")) == []
