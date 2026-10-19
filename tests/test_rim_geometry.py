"""Geometria zmiany rozmiaru felgi (ticket 12): pomiar elipsy felgi i radialny warp wycinka.

Czyste funkcje `wheeler_worker.rim_geometry` na syntetycznych kołach: jasna felga w ciemnej
oponie na zaszumionym tle.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from wheeler_worker.rim_geometry import (
    TYPICAL_K,
    Ellipse,
    RimGeometry,
    measure_rim,
    warp_rim,
)

TIRE = (30, 30, 30)
RIM = (220, 220, 220)


def _ellipse_inside(
    size: tuple[int, int], ellipse: Ellipse, scale: float = 1.0
) -> npt.NDArray[np.bool_]:
    """Piksele (środki) wewnątrz `ellipse` przeskalowanej o `scale` wokół środka."""
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    angle = np.deg2rad(ellipse.angle)
    dx, dy = xs - ellipse.cx, ys - ellipse.cy
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    a, b = ellipse.width / 2 * scale, ellipse.height / 2 * scale
    return np.asarray((u / a) ** 2 + (v / b) ** 2 <= 1)


def _wheel_photo(size: tuple[int, int], rim: Ellipse, k: float) -> Image.Image:
    """Szum, na nim opona (elipsa felgi * k) i felga."""
    rng = np.random.default_rng(5)
    width, height = size
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    pixels[_ellipse_inside(size, rim, k)] = TIRE
    pixels[_ellipse_inside(size, rim)] = RIM
    return Image.fromarray(pixels)


def _bright(image: Image.Image) -> npt.NDArray[np.bool_]:
    return np.asarray(np.asarray(image.convert("L")) > 125)


def test_larger_rim_grows_to_target_radius_and_keeps_pixels_outside_tire_edge() -> None:
    # Felga rho_f = 0,7 zewnętrznej krawędzi opony (promień 80): +2″ -> rho = 0,78 (62,4 px).
    rim = Ellipse(cx=100, cy=100, width=112, height=112, angle=0)
    geometry = RimGeometry(rim=rim, k=80 / 56)
    photo = _wheel_photo((200, 200), rim, geometry.k)

    warped = warp_rim(photo, geometry, rim_size=2)

    assert warped.size == photo.size and warped.mode == "RGB"
    bright = _bright(warped)
    assert bright[_ellipse_inside(photo.size, rim, 60 / 56)].all()
    assert not bright[
        ~_ellipse_inside(photo.size, rim, 65 / 56) & _ellipse_inside(photo.size, rim, 78 / 56)
    ].any()  # bok opony cieńszy, ale jest
    outside = ~_ellipse_inside(photo.size, rim, geometry.k)
    assert np.array_equal(np.asarray(warped)[outside], np.asarray(photo)[outside])


def test_smaller_rim_in_three_quarter_view_shrinks_along_the_rotated_rim_ellipse() -> None:
    # Ujęcie 3/4: felga to obrócona elipsa. rho_f = 1/1,4 ≈ 0,714, -2″ -> rho ≈ 0,634,
    # czyli felga ≈ 0,888 obecnej.
    rim = Ellipse(cx=110, cy=95, width=60, height=100, angle=20)
    geometry = RimGeometry(rim=rim, k=1.4)
    photo = _wheel_photo((220, 200), rim, geometry.k)

    warped = warp_rim(photo, geometry, rim_size=-2)

    bright = _bright(warped)
    assert bright[_ellipse_inside(photo.size, rim, 0.85)].all()
    ring = _ellipse_inside(photo.size, rim, 1.0) & ~_ellipse_inside(photo.size, rim, 0.93)
    assert not bright[ring].any()  # tu była felga, teraz grubszy bok opony
    outside = ~_ellipse_inside(photo.size, rim, geometry.k)
    assert np.array_equal(np.asarray(warped)[outside], np.asarray(photo)[outside])


def _mask(inside: npt.NDArray[np.bool_]) -> Image.Image:
    return Image.fromarray(inside.astype(np.uint8) * 255)


SIZE = (200, 200)
SIDE_RIM = Ellipse(cx=100, cy=100, width=112, height=112, angle=0)


def _assert_close(found: Ellipse, expected: Ellipse, tolerance: float = 2.0) -> None:
    assert abs(found.cx - expected.cx) <= tolerance and abs(found.cy - expected.cy) <= tolerance
    assert sorted((found.width, found.height)) == pytest.approx(
        sorted((expected.width, expected.height)), abs=2 * tolerance
    )


def test_rim_is_the_sam_candidate_inside_the_wheel_and_tire_edge_is_rim_times_profile_k() -> None:
    wheel = _mask(_ellipse_inside(SIZE, SIDE_RIM, 80 / 56))
    hub = _mask(_ellipse_inside(SIZE, SIDE_RIM, 10 / 56))  # samo logo/piasta
    rim = _mask(_ellipse_inside(SIZE, SIDE_RIM))
    whole_wheel = wheel  # SAM często zwraca też całe koło

    geometry = measure_rim(wheel, [(hub, 0.99), (whole_wheel, 0.98), (rim, 0.90)])

    assert geometry is not None
    _assert_close(geometry.rim, SIDE_RIM)
    assert geometry.k == pytest.approx(80 / 56, abs=0.03)


def test_occluded_rim_is_fitted_only_to_its_edge_against_the_tire() -> None:
    # Zderzak innego auta zasłania prawą część koła (x ≥ 130): bez opony za tym brzegiem.
    visible = np.ones((SIZE[1], SIZE[0]), dtype=bool)
    visible[:, 130:] = False
    wheel = _mask(_ellipse_inside(SIZE, SIDE_RIM, 80 / 56) & visible)
    rim = _mask(_ellipse_inside(SIZE, SIDE_RIM) & visible)

    geometry = measure_rim(wheel, [(rim, 0.9)])

    assert geometry is not None
    _assert_close(geometry.rim, SIDE_RIM)


def test_implausible_tire_profile_falls_back_to_typical_k() -> None:
    # Maska koła dwa razy większa od felgi (np. z kawałkiem nadkola): k = 2 poza 1,15-1,6.
    wheel = _mask(_ellipse_inside(SIZE, SIDE_RIM, 50 / 56))
    rim = _mask(_ellipse_inside(SIZE, SIDE_RIM, 25 / 56))

    geometry = measure_rim(wheel, [(rim, 0.9)])

    assert geometry is not None and geometry.k == TYPICAL_K == 1.32


@pytest.mark.parametrize(
    "edge_scale",
    [0.42, 0.935],  # między regułą wyboru 45-92% a granicą pomiaru 40-95%
)
def test_candidate_at_the_edge_of_plausible_size_loses_to_a_typical_rim(edge_scale: float) -> None:
    # Reguła wyboru z rozpoznania (`rim_ellipse/NOTES.md`): felga ma 45-92% średnicy koła.
    # Kandydat tuż poza nią (np. piasta z ramionami albo felga z kawałkiem opony) przegrywa,
    # choć SAM ocenia go wyżej.
    wheel = _mask(_ellipse_inside(SIZE, SIDE_RIM, 80 / 56))
    edge = _mask(_ellipse_inside(SIZE, SIDE_RIM, edge_scale * 80 / 56))
    rim = _mask(_ellipse_inside(SIZE, SIDE_RIM))  # 70%

    geometry = measure_rim(wheel, [(edge, 0.99), (rim, 0.8)])

    assert geometry is not None
    _assert_close(geometry.rim, SIDE_RIM)


@pytest.mark.parametrize(
    "rim_scale",
    [0.35, 0.97],  # felga poza 40-95% średnicy koła
)
def test_rim_outside_plausible_size_is_not_measured(rim_scale: float) -> None:
    wheel = _mask(_ellipse_inside(SIZE, SIDE_RIM, 80 / 56))
    rim = _mask(_ellipse_inside(SIZE, SIDE_RIM, rim_scale * 80 / 56))

    assert measure_rim(wheel, [(rim, 0.9)]) is None


def test_no_rim_mask_is_not_measured() -> None:
    wheel = _mask(_ellipse_inside(SIZE, SIDE_RIM, 80 / 56))

    assert measure_rim(wheel, []) is None
    assert measure_rim(wheel, [(Image.new("L", SIZE, 0), 0.9)]) is None
