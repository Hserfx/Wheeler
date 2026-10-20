"""Maska karoserii przed zmianą rozmiaru koła (ticket 15): kryteria, wybór i czyszczenie.

Czyste funkcje `wheeler_worker.body_mask` na syntetycznych kołach (przeniesione z prototypu
`prototype/whole-wheel-resize`, `test_whole_wheel.py`): ciemna opona z jasną felgą na tle
z lekkim szumem, nad nią błotnik. Przez workera z atrapami: `test_rim_size.py`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from helpers import ellipse_inside
from PIL import Image

from wheeler_worker.body_mask import (
    BodyCandidate,
    BodyMaskProblem,
    BodyMaskStats,
    BodySegments,
    arch_gap_points,
    body_for_resize,
    body_mask_stats,
    body_prompt,
    choose_body,
    clean_body_mask,
    foreign_occlusion,
)
from wheeler_worker.compositing import Box, crop_box_for, mask_box
from wheeler_worker.rim_geometry import Ellipse, RimGeometry

BACKGROUND = 160
TIRE = 10
RIM = 230
BODY = (40, 60, 200)

OLD = RimGeometry(Ellipse(cx=200, cy=230, width=184, height=184, angle=0), tire_ratio=1.32)
SIZE = (400, 400)
FENDER_Y = 230 - 121 + 15  # błotnik zasłania 15 px góry opony (promień opony 121,4 px)


def _car(
    size: tuple[int, int], wheel: RimGeometry, fender_y: int
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


def _body_problem(body: npt.NDArray[np.bool_]) -> BodyMaskProblem | None:
    _, wheel, _ = _car(SIZE, OLD, FENDER_Y)
    box = mask_box(wheel)
    assert box is not None
    crop = crop_box_for(box, SIZE, margin=0.25)
    return body_mask_stats(_image(body), wheel, OLD.tire, crop).problem()


def test_fender_above_the_wheel_is_a_good_body_mask() -> None:
    _, _, body = _car(SIZE, OLD, FENDER_Y)

    assert _body_problem(_mask(body)) is None


def test_bad_body_masks_are_recognised() -> None:
    _, wheel, body = _car(SIZE, OLD, FENDER_Y)
    nothing = np.zeros(SIZE[::-1], dtype=bool)
    assert _body_problem(nothing) is BodyMaskProblem.EMPTY
    # SAM wziął całe auto razem z kołem.
    assert _body_problem(_mask(body) | _mask(wheel)) is BodyMaskProblem.COVERS_WHEEL
    # Wszystko poza kołem (karoseria, ziemia, tło).
    assert _body_problem(~_mask(wheel)) is BodyMaskProblem.COVERS_CROP
    # Coś obok (np. naklejka, próg), ale nie błotnik nad nadkolem.
    sill = nothing.copy()
    sill[300:330, 20:80] = True
    assert _body_problem(sill) is BodyMaskProblem.MISSES_ARCH


GOOD = BodyMaskStats(pixels=9000, wheel_share=0.0, crop_share=0.4, arch_share=0.9)
WITH_WHEEL = BodyMaskStats(pixels=30000, wheel_share=0.6, crop_share=0.5, arch_share=0.9)
NO_FENDER = BodyMaskStats(pixels=500, wheel_share=0.0, crop_share=0.02, arch_share=0.1)


def test_body_mask_is_the_best_scored_good_candidate_preferring_arch_gap_negatives() -> None:
    candidates = [
        BodyCandidate(with_gap=False, score=0.99, stats=GOOD),
        BodyCandidate(with_gap=True, score=0.95, stats=WITH_WHEEL),
        BodyCandidate(with_gap=True, score=0.60, stats=GOOD),
        BodyCandidate(with_gap=True, score=0.80, stats=GOOD),
    ]

    assert choose_body(candidates) == 3
    assert choose_body(candidates[:2]) == 0


def test_without_a_good_body_mask_the_wheel_is_unavailable_with_the_reason() -> None:
    # Żadnego kandydata nie bierzemy „na siłę”: powód = problem kandydata z najwyższą oceną SAM.
    candidates = [
        BodyCandidate(with_gap=True, score=0.70, stats=NO_FENDER),
        BodyCandidate(with_gap=True, score=0.95, stats=WITH_WHEEL),
    ]

    assert choose_body(candidates) is BodyMaskProblem.COVERS_WHEEL
    assert choose_body([]) is BodyMaskProblem.EMPTY


def test_arch_gap_points_lie_just_above_the_visible_tire_top() -> None:
    # Nadkole widoczne nad oponą jako szczelina 12 px między błotnikiem a oponą: punkty do
    # maski nadkola leżą w niej, nad środkiem koła i obok.
    tire_top = 230 - 121
    _, wheel, body = _car(SIZE, OLD, tire_top - 12)

    points = arch_gap_points(OLD, wheel)

    assert len(points) == 3
    for x, y in points:
        assert y < 230 - 100 and abs(x - 200) < 70  # nad kołem
        assert not _mask(wheel)[round(y), round(x)]
        assert not _mask(body)[round(y), round(x)]
        assert OLD.tire.radius(np.array([x]), np.array([y]))[0] < 1.06  # tuż przy oponie


def test_no_arch_gap_points_where_the_wheel_leaves_the_image() -> None:
    # Górny brzeg zdjęcia przecina oponę 20 px nad felgą: nad oponą nie ma szczeliny nadkola,
    # więc żaden punkt nie może trafić na brzeg zdjęcia (promień wychodzący z koła poza
    # zdjęciem odpada, nie jest przycinany do brzegu).
    _, wheel, _ = _car(SIZE, OLD, 0)
    top = 230 - 92 - 20
    cut = wheel.crop((0, top, SIZE[0], SIZE[1]))
    moved = RimGeometry(OLD.rim.shifted(0, -top), OLD.tire_ratio)

    assert arch_gap_points(moved, cut) == []


def test_foreign_occlusion_is_the_part_of_the_tire_hidden_by_something_other_than_the_body() -> (
    None
):
    # Górę opony zasłania własny błotnik (nie liczy się), a przód (x > 290) zderzak innego auta,
    # którego maska karoserii nie obejmuje.
    _, wheel, body = _car(SIZE, OLD, FENDER_Y)
    assert foreign_occlusion(OLD.tire, wheel, body) < 0.01
    bumper = _mask(wheel).copy()
    bumper[:, 290:] = False

    share = foreign_occlusion(OLD.tire, _image(bumper), body)

    assert 0.05 < share < 0.15


def test_clean_body_mask_fills_sam_grid_holes_without_reaching_into_the_wheel() -> None:
    # SAM przy niepewnej masce daje siatkę dziur; między błotnikiem a oponą jest szczelina
    # nadkola (8 px), której czyszczenie nie może zalać.
    _, wheel, body = _car(SIZE, OLD, FENDER_Y - 8)
    holey = _mask(body).copy()
    holey[::6, ::6] = False
    holey[40:44, 100:104] = False

    cleaned = _mask(clean_body_mask(_image(holey), wheel))

    assert np.array_equal(cleaned, _mask(body) & ~_mask(wheel))


def test_body_prompt_points_at_the_fender_the_wheel_and_the_arch_gap() -> None:
    # Szczelina nadkola 12 px między błotnikiem a oponą.
    tire_top = 230 - 121
    _, wheel, body = _car(SIZE, OLD, tire_top - 12)
    on_wheel, on_body = _mask(wheel), _mask(body)

    prompt = body_prompt(OLD, wheel)

    def at(mask: npt.NDArray[np.bool_], point: tuple[float, float]) -> bool:
        return bool(mask[round(point[1]), round(point[0])])

    # Dodatnie: błotnik nad oponą (środek i ±30°).
    assert len(prompt.positives) == 3
    assert all(at(on_body, p) and p[1] < 230 - 121 for p in prompt.positives)
    # Ujemne: środek felgi i opona widoczna w masce koła.
    assert prompt.negatives[0] == (200, 230)
    assert len(prompt.negatives) > 3 and all(at(on_wheel, p) for p in prompt.negatives)
    # Szczelina nadkola: ujemne do karoserii (5 kątów) i dodatnie do maski nadkola (3 kąty).
    assert (len(prompt.gap), len(prompt.liner)) == (5, 3)
    for point in (*prompt.gap, *prompt.liner):
        assert not at(on_wheel, point) and not at(on_body, point)


TIRE_TOP = 230 - 121
GAP_FENDER_Y = TIRE_TOP - 12  # szczelina nadkola 12 px między błotnikiem a oponą


def _crop(wheel: Image.Image) -> Box:
    box = mask_box(wheel)
    assert box is not None
    return crop_box_for(box, SIZE, margin=0.25)


def _rows(top: int, bottom: int, left: int = 0, right: int = SIZE[0]) -> npt.NDArray[np.bool_]:
    region = np.zeros(SIZE[::-1], dtype=bool)
    region[top:bottom, left:right] = True
    return region


def test_body_for_resize_is_the_good_raw_candidate_without_the_arch_liner() -> None:
    _, wheel, _ = _car(SIZE, OLD, GAP_FENDER_Y)
    # SAM: auto z kołem (lepsza ocena, odpada), błotnik razem z wnętrzem nadkola nad oponą.
    whole_car = _rows(0, GAP_FENDER_Y) | _mask(wheel)
    fender_and_liner = _rows(0, TIRE_TOP - 3)
    liner = _rows(GAP_FENDER_Y, TIRE_TOP - 3, 140, 260)
    segments = BodySegments(
        with_gap=[(_image(whole_car), 0.95), (_image(fender_and_liner), 0.8)],
        without_gap=[],
        # Wnętrze nadkola: kandydat obejmujący błotnik (punkt dodatni karoserii) odpada.
        liner=[(_image(fender_and_liner), 0.9), (_image(liner), 0.5)],
    )

    body = body_for_resize(segments, wheel, OLD, _crop(wheel))

    assert isinstance(body, Image.Image)
    chosen = _mask(body)
    assert chosen[_rows(0, GAP_FENDER_Y)].all()
    # Wnętrze nadkola wraca do koła (domknięcie zaokrągla tylko narożniki: kilka pikseli).
    assert chosen[liner].mean() < 0.02
    assert not (chosen & _mask(wheel)).any()
    assert foreign_occlusion(OLD.tire, wheel, body) < 0.01


def test_body_for_resize_refuses_when_only_bad_candidates_or_a_foreign_occlusion() -> None:
    _, wheel, body = _car(SIZE, OLD, FENDER_Y)
    whole_car = _image(_mask(body) | _mask(wheel))
    only_bad = BodySegments(with_gap=[(whole_car, 0.9)], without_gap=[(whole_car, 0.8)], liner=[])

    assert body_for_resize(only_bad, wheel, OLD, _crop(wheel)) is BodyMaskProblem.COVERS_WHEEL
    # Przód opony (x > 290) zasłania zderzak innego auta, którego karoseria nie obejmuje.
    bumper = _mask(wheel).copy()
    bumper[:, 290:] = False
    good = BodySegments(with_gap=[(body, 0.9)], without_gap=[], liner=[])

    refused = body_for_resize(good, _image(bumper), OLD, _crop(wheel))

    assert refused is BodyMaskProblem.FOREIGN_OCCLUSION
