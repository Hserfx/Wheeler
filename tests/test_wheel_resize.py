"""Zmiana rozmiaru całego koła (ticket 16): skala cala, nowe koło i plan wycinka dla modelu.

Czyste funkcje `wheeler_worker.wheel_resize` na syntetycznych kołach (przeniesione z prototypu
`prototype/whole-wheel-resize`, `test_whole_wheel.py`): ciemna opona z jasną felgą na tle
z lekkim szumem, nad nią błotnik. Przez workera z atrapami: `test_rim_size.py`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from helpers import ellipse_inside
from PIL import Image

from wheeler_worker.compositing import grow_wheel_mask
from wheeler_worker.rim_geometry import Ellipse, RimGeometry
from wheeler_worker.wheel_resize import inch_px, plan_resize, resized_geometry

BACKGROUND = 160
TIRE = 10
RIM = 230
BODY = (40, 60, 200)


def test_inch_is_visible_rim_diameter_over_nominal_size_plus_rim_flange() -> None:
    # Felga 17″ widoczna na 368 px: 368 / (17 + 1,4) = 20 px na cal. W 3/4 felga jest zwężona
    # w poziomie, więc skala bierze dłuższą oś.
    rim = Ellipse(cx=200, cy=200, width=250, height=368, angle=3)

    assert inch_px(rim, current_rim=17) == pytest.approx(20.0)
    # Felga 22″ na tych samych pikselach: mniej pikseli na cal.
    assert inch_px(rim, current_rim=22) == pytest.approx(368 / 23.4)


def test_larger_wheel_grows_rim_and_tire_by_the_same_inches_and_keeps_ground_contact() -> None:
    # Felga 17″ = 368 px (20 px/cal), opona 1,32 x: 485,76 px. +2″: felga 408 px, opona
    # 525,76 px, bok opony tej samej grubości; najniższy punkt opony (styk z ziemią)
    # y = 200 + 242,88.
    old = RimGeometry(Ellipse(cx=300, cy=200, width=368, height=368, angle=0), tire_ratio=1.32)

    new = resized_geometry(old, rim_size=2, inch=20.0)

    assert new.rim.width == pytest.approx(408) and new.rim.height == pytest.approx(408)
    assert new.tire.height == pytest.approx(525.76)
    assert new.rim.cx == pytest.approx(300)
    assert new.rim.cy + new.tire.height / 2 == pytest.approx(200 + 485.76 / 2)


def test_smaller_wheel_shrinks_rim_and_tire_by_the_same_inches_down_to_the_ground() -> None:
    old = RimGeometry(Ellipse(cx=300, cy=200, width=368, height=368, angle=0), tire_ratio=1.32)

    new = resized_geometry(old, rim_size=-1, inch=20.0)

    assert new.rim.height == pytest.approx(348)
    assert new.tire.height == pytest.approx(465.76)
    assert new.rim.cy + new.tire.height / 2 == pytest.approx(200 + 485.76 / 2)


def _lowest_point(ellipse: Ellipse) -> tuple[float, float]:
    """Najniższy punkt brzegu elipsy, z gęstego próbkowania (niezależnie od wzoru w kodzie)."""
    angles = np.linspace(0, 2 * np.pi, 200_000, endpoint=False)
    xs, ys = ellipse.from_unit(np.cos(angles), np.sin(angles))
    lowest = int(np.argmax(ys))
    return float(xs[lowest]), float(ys[lowest])


@pytest.mark.parametrize("rim_size", [-2, -1, 1, 2])
def test_ground_contact_stays_also_for_a_rotated_three_quarter_wheel(rim_size: int) -> None:
    # Ujęcie 3/4: elipsa zwężona i obrócona; najniższy punkt opony nie leży na jej osi.
    old = RimGeometry(Ellipse(cx=300, cy=200, width=220, height=368, angle=15), tire_ratio=1.32)

    new = resized_geometry(old, rim_size=rim_size, inch=20.0)

    assert _lowest_point(new.tire) == pytest.approx(_lowest_point(old.tire), abs=0.05)
    assert new.rim.angle == old.rim.angle
    assert new.rim.width / new.rim.height == pytest.approx(220 / 368)
    assert new.rim.height == pytest.approx(368 + rim_size * 20)


OLD = RimGeometry(Ellipse(cx=200, cy=230, width=184, height=184, angle=0), tire_ratio=1.32)
SIZE = (400, 400)
FENDER_Y = 230 - 121 + 15  # błotnik zasłania 15 px góry opony (promień opony 121,4 px)
MASK_GROW = 0.03


def _car(
    size: tuple[int, int] = SIZE, wheel: RimGeometry = OLD, fender_y: int = FENDER_Y
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Zdjęcie, maska koła (widoczna opona + felga) i maska karoserii: tło z lekkim szumem,
    koło, a nad `fender_y` błotnik zasłaniający górę opony."""
    width, height = size
    rng = np.random.default_rng(1)
    pixels = rng.integers(BACKGROUND - 4, BACKGROUND + 5, (height, width, 3)).astype(np.uint8)
    tire = ellipse_inside(size, wheel.tire)
    pixels[tire] = TIRE
    pixels[ellipse_inside(size, wheel.rim)] = RIM
    body = np.zeros((height, width), dtype=bool)
    body[:fender_y] = True
    pixels[body] = BODY
    return Image.fromarray(pixels), _image(tire & ~body), _image(body)


def _image(mask: npt.NDArray[np.bool_]) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255)


def _mask(image: Image.Image) -> npt.NDArray[np.bool_]:
    return np.asarray(np.asarray(image.convert("L")) > 0)


def test_larger_wheel_reaches_new_ellipses_behind_untouched_body() -> None:
    photo, wheel, body = _car()

    plan = plan_resize(photo, wheel, OLD, body, rim_size=2, inch=10.0, mask_grow=MASK_GROW)

    # Felga 184 px -> 204 px, opona 242,9 px -> 262,9 px; styk z ziemią y = 351,4.
    new = plan.geometry
    assert new.rim.height == pytest.approx(204)
    assert new.rim.cy + new.tire.height / 2 == pytest.approx(230 + 242.88 / 2)
    pixels = np.asarray(plan.model_input)
    visible = ~_mask(body)
    rim_core = ellipse_inside(SIZE, new.rim.scaled(0.97)) & visible
    assert (pixels[rim_core] == RIM).mean() > 0.97
    sidewall = ellipse_inside(SIZE, new.tire.scaled(0.97)) & ~ellipse_inside(
        SIZE, new.rim.scaled(1.04)
    )
    assert (pixels[sidewall & visible] <= TIRE + 5).mean() > 0.97
    # Nowe koło wychodzi poza stare (na boki i w górę, pod błotnik), błotnik zostaje.
    assert (sidewall & visible & ~_mask(wheel)).sum() > 500
    assert not (_mask(plan.wheel) & ~visible).any()


def test_edit_mask_is_old_and_new_wheel_grown_minus_body_and_nothing_else_changes() -> None:
    photo, wheel, body = _car()

    for rim_size in (-2, 2):
        plan = plan_resize(
            photo, wheel, OLD, body, rim_size=rim_size, inch=10.0, mask_grow=MASK_GROW
        )

        both = _image(_mask(wheel) | _mask(plan.wheel))
        expected = _mask(grow_wheel_mask(both, MASK_GROW)) & ~_mask(body)
        edit = _mask(plan.edit_mask)
        assert np.array_equal(edit, expected)
        untouched = ~edit
        assert np.array_equal(np.asarray(plan.model_input)[untouched], np.asarray(photo)[untouched])


def test_smaller_wheel_drops_to_the_ground_and_the_old_wheel_stays_editable() -> None:
    photo, wheel, body = _car()

    plan = plan_resize(photo, wheel, OLD, body, rim_size=-2, inch=10.0, mask_grow=MASK_GROW)

    new_wheel = _mask(plan.wheel)
    old_wheel = _mask(wheel)
    assert new_wheel.sum() < 0.9 * old_wheel.sum()
    # Koło opada do punktu styku: najniższy wiersz nowego koła = najniższy wiersz starego.
    assert np.nonzero(new_wheel)[0].max() == np.nonzero(old_wheel)[0].max()
    # Odsłonięte nadkole (stare koło minus nowe) model może domalować.
    assert (old_wheel <= _mask(plan.edit_mask)).all()


def test_rotated_three_quarter_wheel_keeps_its_ground_contact_row() -> None:
    wheel_3q = RimGeometry(
        Ellipse(cx=200, cy=230, width=120, height=184, angle=12), tire_ratio=1.32
    )
    photo, wheel, body = _car(wheel=wheel_3q)

    for rim_size in (-2, 2):
        plan = plan_resize(
            photo, wheel, wheel_3q, body, rim_size=rim_size, inch=10.0, mask_grow=MASK_GROW
        )

        rows = np.nonzero(_mask(plan.wheel))[0]
        assert abs(int(rows.max()) - int(np.nonzero(_mask(wheel))[0].max())) <= 1


def test_wheel_cut_by_the_photo_edge_grows_without_touching_anything_outside_the_edit() -> None:
    # Prawy brzeg zdjęcia przecina oponę 100 px od środka: nowe koło też kończy się na brzegu.
    photo, wheel, body = _car()
    cut = (0, 0, 300, SIZE[1])
    photo, wheel, body = photo.crop(cut), wheel.crop(cut), body.crop(cut)

    plan = plan_resize(photo, wheel, OLD, body, rim_size=2, inch=10.0, mask_grow=MASK_GROW)

    assert plan.model_input.size == photo.size
    assert _mask(plan.wheel)[:, -1].sum() > _mask(wheel)[:, -1].sum()
    untouched = ~_mask(plan.edit_mask)
    assert np.array_equal(np.asarray(plan.model_input)[untouched], np.asarray(photo)[untouched])
