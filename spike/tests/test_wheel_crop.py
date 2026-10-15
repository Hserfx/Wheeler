import numpy as np
from PIL import Image

from wheeler_spike.wheel_crop import Box, composite, crop_box_for, grow_mask, working_size


def _noise_photo(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _disc_mask(width: int, height: int, cx: int, cy: int, r: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    return Image.fromarray((((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.uint8) * 255)


def test_crop_box_is_square_around_wheel_with_margin():
    wheel = Box(400, 300, 600, 500)  # 200x200 wheel in a 2000x1000 photo

    crop = crop_box_for(wheel, image_size=(2000, 1000), margin=0.25)

    assert crop == Box(350, 250, 650, 550)


def test_crop_box_near_edge_is_shifted_inside_image():
    wheel = Box(20, 850, 220, 990)  # wheel touching the bottom-left corner of a 2000x1000 photo

    crop = crop_box_for(wheel, image_size=(2000, 1000), margin=0.25)

    assert crop == Box(0, 700, 300, 1000)


def test_crop_box_larger_than_image_is_clipped_to_image():
    wheel = Box(100, 50, 700, 450)  # huge wheel close-up in an 800x500 photo

    crop = crop_box_for(wheel, image_size=(800, 500), margin=0.25)

    assert crop == Box(0, 0, 800, 500)


def test_square_crop_is_upscaled_to_working_resolution():
    assert working_size(Box(0, 0, 300, 300), resolution=1024) == (1024, 1024)
    assert working_size(Box(0, 0, 300, 300), resolution=1056) == (1056, 1056)


def test_working_size_keeps_aspect_and_snaps_to_multiple_of_32():
    # 800x500 (1.6:1) at a 1024^2 pixel budget -> ~1295x810, snapped to 32 px
    assert working_size(Box(0, 0, 800, 500), resolution=1024) == (1280, 800)


def test_composite_keeps_every_pixel_outside_mask_bit_identical():
    original = _noise_photo(400, 240)
    mask = _disc_mask(400, 240, cx=150, cy=120, r=60)  # full-size mask of one wheel
    generated = Image.new("RGB", (1024, 1024), (10, 200, 30))  # model output for the crop

    result = composite(original, generated, crop=Box(70, 40, 230, 200), mask=mask, feather=4)

    assert result.size == original.size
    outside = np.asarray(mask) == 0
    assert np.array_equal(np.asarray(result)[outside], np.asarray(original)[outside])


def test_composite_takes_generated_pixels_inside_mask():
    original = _noise_photo(400, 240)
    mask = _disc_mask(400, 240, cx=150, cy=120, r=60)
    generated = Image.new("RGB", (1024, 1024), (10, 200, 30))

    result = composite(original, generated, crop=Box(70, 40, 230, 200), mask=mask, feather=4)

    core = np.asarray(_disc_mask(400, 240, cx=150, cy=120, r=50)) > 0
    assert (np.asarray(result)[core] == (10, 200, 30)).all()


def test_grow_mask_widens_wheel_mask_by_given_pixels():
    mask = _disc_mask(200, 200, cx=100, cy=100, r=40)

    grown = np.asarray(grow_mask(mask, pixels=5)) > 0

    assert grown[100, 100 + 45] and grown[100 - 45, 100]  # 5 px beyond the old edge
    assert not grown[100, 100 + 48]  # but not much further
    assert grown[np.asarray(mask) > 0].all()
