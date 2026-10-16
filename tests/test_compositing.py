"""Wycinanie okolicy koła i kompozycja wyniku w oryginał (czysta logika, bez modelu)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from wheeler_worker.compositing import Box, composite, crop_box_for, grow_mask, working_size


def _noise_photo(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _disc_mask(width: int, height: int, cx: int, cy: int, r: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    return Image.fromarray(inside.astype(np.uint8) * 255)


def test_crop_box_is_square_around_wheel_with_margin() -> None:
    wheel = Box(400, 300, 600, 500)  # koło 200x200 na zdjęciu 2000x1000

    assert crop_box_for(wheel, image_size=(2000, 1000), margin=0.25) == Box(350, 250, 650, 550)


def test_crop_box_near_edge_is_shifted_inside_image() -> None:
    wheel = Box(20, 850, 220, 990)  # koło w lewym dolnym rogu zdjęcia 2000x1000

    assert crop_box_for(wheel, image_size=(2000, 1000), margin=0.25) == Box(0, 700, 300, 1000)


def test_crop_box_larger_than_image_is_clipped_to_image() -> None:
    wheel = Box(100, 50, 700, 450)  # zbliżenie koła na zdjęciu 800x500

    assert crop_box_for(wheel, image_size=(800, 500), margin=0.25) == Box(0, 0, 800, 500)


def test_square_crop_is_upscaled_to_working_resolution() -> None:
    assert working_size(Box(0, 0, 300, 300), resolution=1024) == (1024, 1024)
    assert working_size(Box(0, 0, 300, 300), resolution=1056) == (1056, 1056)


def test_working_size_keeps_aspect_and_snaps_to_multiple_of_32() -> None:
    # 800x500 (1,6:1) przy budżecie 1024² px -> ~1295x810, zaokrąglone do 32 px
    assert working_size(Box(0, 0, 800, 500), resolution=1024) == (1280, 800)


def test_composite_keeps_every_pixel_outside_mask_bit_identical() -> None:
    original = _noise_photo(400, 240)
    mask = _disc_mask(400, 240, cx=150, cy=120, r=60)
    generated = Image.new("RGB", (1024, 1024), (10, 200, 30))  # wynik modelu dla wycinka

    result = composite(original, generated, crop=Box(70, 40, 230, 200), mask=mask, feather=4)

    assert result.size == original.size
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(result)[outside], np.asarray(original)[outside])


def test_composite_takes_generated_pixels_inside_mask() -> None:
    original = _noise_photo(400, 240)
    mask = _disc_mask(400, 240, cx=150, cy=120, r=60)
    generated = Image.new("RGB", (1024, 1024), (10, 200, 30))

    result = composite(original, generated, crop=Box(70, 40, 230, 200), mask=mask, feather=4)

    core = np.asarray(_disc_mask(400, 240, cx=150, cy=120, r=50)) > 0
    assert (np.asarray(result)[core] == (10, 200, 30)).all()


def test_grow_mask_widens_wheel_mask_by_given_pixels() -> None:
    mask = _disc_mask(200, 200, cx=100, cy=100, r=40)

    grown = np.asarray(grow_mask(mask, pixels=5)) > 0

    assert grown[100, 100 + 45] and grown[100 - 45, 100]  # 5 px za starą krawędzią
    assert not grown[100, 100 + 48]  # ale nie dużo dalej
    assert grown[np.asarray(mask) > 0].all()
