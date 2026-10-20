"""Pomiar felgi przed zmianą rozmiaru koła (ticket 12): elipsa felgi i krawędź opony.

Czyste funkcje `wheeler_worker.rim_geometry` na syntetycznych maskach. Zmiana rozmiaru koła:
`test_wheel_resize.py`.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from helpers import ellipse_inside
from PIL import Image

from wheeler_worker.rim_geometry import (
    TYPICAL_TIRE_RATIO,
    Ellipse,
    measure_rim,
)


def test_ellipse_follows_the_opencv_convention_also_when_rotated() -> None:
    # Elipsy z `cv2.fitEllipse` (pomiar) i warp w układzie `Ellipse` (`wheel_resize`) muszą
    # się zgadzać:
    # wnętrze `Ellipse` to wypełniona elipsa narysowana przez OpenCV.
    ellipse = Ellipse(cx=110, cy=95, width=60, height=100, angle=20)
    drawn = np.zeros((200, 220), dtype=np.uint8)
    cv2.ellipse(drawn, ellipse.as_cv2(), 1, -1)

    inside = ellipse_inside((220, 200), ellipse)

    iou = (inside & (drawn > 0)).sum() / (inside | (drawn > 0)).sum()
    assert iou > 0.95  # obrót w drugą stronę: 0,65


def _mask(inside: npt.NDArray[np.bool_]) -> Image.Image:
    return Image.fromarray(inside.astype(np.uint8) * 255)


SIZE = (200, 200)
SIDE_RIM = Ellipse(cx=100, cy=100, width=112, height=112, angle=0)


def _assert_close(found: Ellipse, expected: Ellipse, tolerance: float = 2.0) -> None:
    assert abs(found.cx - expected.cx) <= tolerance and abs(found.cy - expected.cy) <= tolerance
    assert sorted((found.width, found.height)) == pytest.approx(
        sorted((expected.width, expected.height)), abs=2 * tolerance
    )


def test_rim_is_the_sam_candidate_inside_the_wheel_and_tire_edge_is_rim_times_profile_ratio() -> (
    None
):
    wheel = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(80 / 56)))
    hub = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(10 / 56)))  # samo logo/piasta
    rim = _mask(ellipse_inside(SIZE, SIDE_RIM))
    whole_wheel = wheel  # SAM często zwraca też całe koło

    geometry = measure_rim(wheel, [(hub, 0.99), (whole_wheel, 0.98), (rim, 0.90)])

    assert geometry is not None
    _assert_close(geometry.rim, SIDE_RIM)
    assert geometry.tire_ratio == pytest.approx(80 / 56, abs=0.03)


def test_occluded_rim_is_fitted_only_to_its_edge_against_the_tire() -> None:
    # Zderzak innego auta zasłania prawą część koła (x ≥ 130): bez opony za tym brzegiem.
    visible = np.ones((SIZE[1], SIZE[0]), dtype=bool)
    visible[:, 130:] = False
    wheel = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(80 / 56)) & visible)
    rim = _mask(ellipse_inside(SIZE, SIDE_RIM) & visible)

    geometry = measure_rim(wheel, [(rim, 0.9)])

    assert geometry is not None
    _assert_close(geometry.rim, SIDE_RIM)


def test_implausible_tire_profile_falls_back_to_typical_tire_ratio() -> None:
    # Maska koła dwa razy większa od felgi (np. z kawałkiem nadkola): k = 2 poza 1,15-1,6.
    wheel = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(50 / 56)))
    rim = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(25 / 56)))

    geometry = measure_rim(wheel, [(rim, 0.9)])

    assert geometry is not None and geometry.tire_ratio == TYPICAL_TIRE_RATIO == 1.32


@pytest.mark.parametrize(
    "edge_scale",
    [0.42, 0.935],  # między regułą wyboru 45-92% a granicą pomiaru 40-95%
)
def test_candidate_at_the_edge_of_plausible_size_loses_to_a_typical_rim(edge_scale: float) -> None:
    # Reguła wyboru z rozpoznania (`rim_ellipse/NOTES.md`): felga ma 45-92% średnicy koła.
    # Kandydat tuż poza nią (np. piasta z ramionami albo felga z kawałkiem opony) przegrywa,
    # choć SAM ocenia go wyżej.
    wheel = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(80 / 56)))
    edge = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(edge_scale * 80 / 56)))
    rim = _mask(ellipse_inside(SIZE, SIDE_RIM))  # 70%

    geometry = measure_rim(wheel, [(edge, 0.99), (rim, 0.8)])

    assert geometry is not None
    _assert_close(geometry.rim, SIDE_RIM)


@pytest.mark.parametrize(
    "rim_scale",
    [0.35, 0.97],  # felga poza 40-95% średnicy koła
)
def test_rim_outside_plausible_size_is_not_measured(rim_scale: float) -> None:
    wheel = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(80 / 56)))
    rim = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(rim_scale * 80 / 56)))

    assert measure_rim(wheel, [(rim, 0.9)]) is None


def test_no_rim_mask_is_not_measured() -> None:
    wheel = _mask(ellipse_inside(SIZE, SIDE_RIM.scaled(80 / 56)))

    assert measure_rim(wheel, []) is None
    assert measure_rim(wheel, [(Image.new("L", SIZE, 0), 0.9)]) is None
