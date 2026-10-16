"""Wejście i metadane w procesorze: HEIC, obrót według EXIF, ICC i EXIF bez GPS w wyniku."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from helpers import noise_photo
from photo_files import (
    GPS_IFD,
    MAKE,
    ORIENTATION,
    display_profile,
    phone_file,
    phone_heic,
    quadrants,
)
from PIL import Image

from wheeler_worker.processing import GenerateItem, SegmentItem
from wheeler_worker.wheel_swap import WheelSwapProcessor


def _swap(photo: Path, out: Path, click: tuple[float, float], radius: int) -> tuple[Path, Path]:
    """Klik -> maska -> generowanie; zwraca (maska, wynik)."""
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=radius), editor=SolidEditor()
    )
    mask = (
        processor.segment(
            SegmentItem(job_id=1, photo_path=photo, output_dir=out, params={}, clicks={1: click})
        )
        .masks[1]
        .mask
    )
    rim = out / "rim.png"
    Image.new("RGB", (30, 30), (200, 200, 200)).save(rim)
    result = processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=rim,
            output_dir=out,
            params={"seed": 3},
            masks={1: mask},
        )
    )
    return mask, result.result_path


def test_heic_with_orientation_and_gps_gives_upright_png_with_icc_and_no_gps(
    tmp_path: Path,
) -> None:
    photo = tmp_path / "input.heic"
    photo.write_bytes(phone_heic(quadrants(64, 48)))

    _, result_path = _swap(photo, tmp_path, click=(48, 36), radius=6)

    with Image.open(result_path) as result:
        assert result.format == "PNG"
        assert result.size == (64, 48)  # obrócone: zapisane 48x64
        pixels = np.asarray(result.convert("RGB")).astype(int)
        assert result.info.get("icc_profile") == display_profile()
        exif = result.getexif()
    # Lewa góra czerwona, prawa dół niebieska (HEIC jest stratny: porównanie z tolerancją).
    assert np.abs(pixels[5, 5] - (250, 30, 30)).max() < 12
    assert np.abs(pixels[44, 60] - (0, 0, 220)).max() < 12
    assert exif.get(MAKE) == "Apple"
    assert exif.get(ORIENTATION) in (None, 1)  # piksele są już obrócone
    assert GPS_IFD not in exif and not exif.get_ifd(GPS_IFD)


def test_pixels_outside_mask_equal_rotated_original_bit_for_bit(tmp_path: Path) -> None:
    upright = noise_photo(80, 50)
    photo = tmp_path / "input.png"
    photo.write_bytes(phone_file(upright, "PNG"))

    mask_path, result_path = _swap(photo, tmp_path, click=(20, 15), radius=8)

    with Image.open(result_path) as result:
        pixels = np.asarray(result.convert("RGB"))
    outside = np.asarray(Image.open(mask_path).convert("L")) == 0
    assert pixels.shape == (50, 80, 3)
    assert outside.sum() > 0.8 * outside.size
    assert np.array_equal(pixels[outside], np.asarray(upright)[outside])
    # Koło (dysk wokół kliknięcia w układzie obróconego zdjęcia) dostało nową felgę.
    assert tuple(pixels[15, 20]) == (10, 200, 30)
