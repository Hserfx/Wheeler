"""Maska karoserii zasłaniającej koło (ticket 15): kryteria złej maski, wybór i czyszczenie.

Przy zmianie rozmiaru koła (ticket 16) powiększone koło wchodzi w nadkole i chowa się za
karoserią, która zostaje bit w bit. Maskę karoserii daje SAM 2 (`Segmenter.segment_body`),
a ten moduł wybiera z jego kandydatów maskę dobrą albo podaje powód, dla którego zmiana
rozmiaru jest niemożliwa (`rim_size_unavailable`). Reguły z prototypu
`prototype/whole-wheel-resize` (`VERDICT.md`).

Czysty numpy + OpenCV, bez modeli. Układ: piksele obrazu, elipsy jak w `rim_geometry`.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from wheeler_worker.compositing import Box
from wheeler_worker.rim_geometry import BoolArray, Ellipse, RimGeometry, wheel_rays

# Kryteria złej maski karoserii na wycinku koła (surowa maska SAM): maska obejmuje więcej niż
# tyle maski koła (SAM wziął auto z kołami) ...
BODY_MAX_WHEEL_SHARE = 0.05
# ... albo więcej niż tyle wycinka poza kołem (wzięła tło i ziemię) ...
BODY_MAX_CROP_SHARE = 0.85
# ... albo nie pokrywa choć tyle strefy błotnika: pas nad oponą, rho 1,15-1,6 elipsy opony,
# ±40° od pionu.
BODY_MIN_ARCH_SHARE = 0.25
ARCH_ZONE_RHO = (1.15, 1.6)
ARCH_ZONE_DEGREES = 40
# Punkty w szczelinie nadkola (wnętrze nadkola między oponą a błotnikiem): na tych kątach od
# pionu, tyle (w jednostkach elipsy opony) za brzegiem maski koła.
ARCH_GAP_DEGREES = (-20, 0, 20)
ARCH_GAP_OFFSET = 0.03
# Prompt karoserii dla SAM 2: punkty dodatnie nad oponą (promień w elipsie opony, kąty od
# pionu), ujemne na kole (środek felgi i opona na tym promieniu, tam, gdzie widać ją w masce
# koła) i ujemne w szczelinie nadkola na tych kątach, żeby karoseria nie objęła wnętrza nadkola.
BODY_POINT_RHO = 1.45
BODY_POINT_DEGREES = (-30, 0, 30)
BODY_NEGATIVE_RHO = 0.88
BODY_GAP_DEGREES = (-50, -25, 0, 25, 50)
# Obce zasłonięcie: część elipsy opony poza maską koła i karoserią (zderzak innego auta,
# słupek). Powyżej tego udziału powiększone koło weszłoby na coś niechronionego.
MAX_FOREIGN_OCCLUSION = 0.05
# Diagnostyka: karoseria jako czerwona poświata na wycinku.
BODY_TINT = (255, 40, 40)
BODY_TINT_ALPHA = 0.4
# Kandydat SAM na wnętrze nadkola odpada, gdy obejmuje punkt błotnika, więcej niż tyle maski
# koła albo tyle wycinka.
LINER_MAX_WHEEL_SHARE = 0.05
LINER_MAX_CROP_SHARE = 0.40
# Czyszczenie maski karoserii z SAM: domknięcie dyskiem (bok koła razy tyle) i wypełnienie dziur.
BODY_CLOSING = 0.02


class BodyMaskProblem(enum.Enum):
    """Powód, dla którego maski karoserii nie da się użyć (do logu workera)."""

    EMPTY = "pusta"
    COVERS_WHEEL = "obejmuje koło"
    COVERS_CROP = "obejmuje cały wycinek"
    MISSES_ARCH = "nie sięga błotnika nad nadkolem"
    FOREIGN_OCCLUSION = "koło zasłania coś innego niż karoseria"


@dataclass(frozen=True)
class BodyMaskStats:
    """Udziały maski karoserii w wycinku koła."""

    pixels: int
    wheel_share: float
    crop_share: float
    arch_share: float

    def problem(self) -> BodyMaskProblem | None:
        if self.pixels == 0:
            return BodyMaskProblem.EMPTY
        if self.wheel_share > BODY_MAX_WHEEL_SHARE:
            return BodyMaskProblem.COVERS_WHEEL
        if self.crop_share > BODY_MAX_CROP_SHARE:
            return BodyMaskProblem.COVERS_CROP
        if self.arch_share < BODY_MIN_ARCH_SHARE:
            return BodyMaskProblem.MISSES_ARCH
        return None


def body_mask_stats(
    body_mask: Image.Image, wheel_mask: Image.Image, tire: Ellipse, crop: Box
) -> BodyMaskStats:
    """Udziały maski karoserii w wycinku `crop` koła (`wheel_mask` bez poszerzenia, `tire`:
    zewnętrzna krawędź opony z pomiaru felgi)."""
    window = (slice(crop.top, crop.bottom), slice(crop.left, crop.right))
    body = _bool(body_mask)[window]
    wheel = _bool(wheel_mask)[window]
    ys, xs = np.mgrid[crop.top : crop.bottom, crop.left : crop.right].astype(np.float64)
    us, vs = tire.to_unit(xs, ys)
    rho = np.hypot(us, vs)
    # Kąt od kierunku „w górę” obrazu w układzie elipsy (uwzględnia obrót).
    off = np.abs((np.arctan2(vs, us) - up_angle(tire) + np.pi) % (2 * np.pi) - np.pi)
    low, high = ARCH_ZONE_RHO
    arch = (rho >= low) & (rho <= high) & (off <= math.radians(ARCH_ZONE_DEGREES)) & ~wheel
    return BodyMaskStats(
        pixels=int(body.sum()),
        wheel_share=_share(body & wheel, wheel),
        crop_share=_share(body & ~wheel, ~wheel),
        arch_share=_share(body & arch, arch),
    )


@dataclass(frozen=True)
class BodyCandidate:
    """Kandydat SAM 2 na maskę karoserii: czy z punktami ujemnymi w szczelinie nadkola, ocena
    SAM i udziały surowej maski (przed odjęciem nadkola i czyszczeniem)."""

    with_gap: bool
    score: float
    stats: BodyMaskStats


def choose_body(candidates: Sequence[BodyCandidate]) -> int | BodyMaskProblem:
    """Indeks maski karoserii: kandydat bez problemu, najpierw z ujemnymi w szczelinie nadkola
    (bez wnętrza nadkola), najwyższa ocena SAM. Bez dobrego kandydata najlepszego ze złych
    nie bierzemy: wynik to problem kandydata o najwyższej ocenie (powód błędu)."""
    if not candidates:
        return BodyMaskProblem.EMPTY
    good = [i for i, c in enumerate(candidates) if c.stats.problem() is None]
    preferred = [i for i in good if candidates[i].with_gap] or good
    if preferred:
        return max(preferred, key=lambda i: candidates[i].score)
    problem = max(candidates, key=lambda c: c.score).stats.problem()
    assert problem is not None
    return problem


Point = tuple[float, float]
# Maska z multimask SAM 2: L (>0 = obiekt) w rozmiarze zdjęcia i ocena IoU modelu.
ScoredMask = tuple[Image.Image, float]


@dataclass(frozen=True)
class BodySegments:
    """Kandydaci SAM 2 dla jednego koła (`Segmenter.segment_body`): karoseria z promptem
    `BodyPrompt` z ujemnymi w szczelinie nadkola (`with_gap`) i bez nich (`without_gap`)
    oraz wnętrze nadkola (`liner`)."""

    with_gap: Sequence[ScoredMask]
    without_gap: Sequence[ScoredMask]
    liner: Sequence[ScoredMask]


def body_for_resize(
    segments: BodySegments, wheel_mask: Image.Image, geometry: RimGeometry, crop: Box
) -> Image.Image | BodyMaskProblem:
    """Maska karoserii chroniona przy zmianie rozmiaru koła albo powód, dla którego jej nie ma.

    Kolejność z `VERDICT.md`: (a) kryteria `BodyMaskStats.problem` na surowych kandydatach SAM
    w wycinku `crop` i wybór `choose_body` (bez dobrego kandydata: problem); (b) odjęcie
    wnętrza nadkola (`choose_liner`) i `clean_body_mask`; (c) obce zasłonięcie opony powyżej
    `MAX_FOREIGN_OCCLUSION` na masce po czyszczeniu: `FOREIGN_OCCLUSION`. `wheel_mask`: maska
    SAM koła bez poszerzenia."""
    raw = [*segments.with_gap, *segments.without_gap]
    candidates = [
        BodyCandidate(
            with_gap=index < len(segments.with_gap),
            score=score,
            stats=body_mask_stats(mask, wheel_mask, geometry.tire, crop),
        )
        for index, (mask, score) in enumerate(raw)
    ]
    choice = choose_body(candidates)
    if isinstance(choice, BodyMaskProblem):
        return choice
    body = _bool(raw[choice][0])
    liner = choose_liner(segments.liner, body_prompt(geometry, wheel_mask), wheel_mask, crop)
    if liner is not None:
        body &= ~_bool(liner)
    cleaned = clean_body_mask(_image(body), wheel_mask)
    if foreign_occlusion(geometry.tire, wheel_mask, cleaned) > MAX_FOREIGN_OCCLUSION:
        return BodyMaskProblem.FOREIGN_OCCLUSION
    return cleaned


def choose_liner(
    candidates: Sequence[ScoredMask], prompt: BodyPrompt, wheel_mask: Image.Image, crop: Box
) -> Image.Image | None:
    """Maska wnętrza nadkola (nadkole za oponą, które karoseria z SAM często obejmuje, a które
    nowe koło ma zakryć): kandydat o najwyższej ocenie SAM, który nie obejmuje punktu błotnika
    (`prompt.positives`), więcej niż `LINER_MAX_WHEEL_SHARE` koła ani `LINER_MAX_CROP_SHARE`
    wycinka. `None`: brak takiego kandydata."""
    wheel = _bool(wheel_mask)
    window = (slice(crop.top, crop.bottom), slice(crop.left, crop.right))
    best: ScoredMask | None = None
    for mask, score in candidates:
        liner = _bool(mask)
        height, width = liner.shape
        on_fender = any(
            0 <= round(y) < height and 0 <= round(x) < width and liner[round(y), round(x)]
            for x, y in prompt.positives
        )
        ok = (
            liner.any()
            and not on_fender
            and _share(liner & wheel, wheel) <= LINER_MAX_WHEEL_SHARE
            and float(liner[window].mean()) <= LINER_MAX_CROP_SHARE
        )
        if ok and (best is None or score > best[1]):
            best = (mask, score)
    return None if best is None else best[0]


@dataclass(frozen=True)
class BodyPrompt:
    """Punkty promptu SAM 2 dla jednego koła (piksele zdjęcia). Karoseria: `positives`
    (błotnik nad oponą), `negatives` (środek felgi i opona) i opcjonalnie `gap` (ujemne
    w szczelinie nadkola). Wnętrze nadkola: `liner` dodatnie, a `positives` i `negatives`
    ujemne."""

    positives: list[Point]
    negatives: list[Point]
    gap: list[Point]
    liner: list[Point]


def body_prompt(geometry: RimGeometry, wheel_mask: Image.Image) -> BodyPrompt:
    """Prompt karoserii i wnętrza nadkola koła `wheel_mask` (maska SAM bez poszerzenia)
    o feldze i oponie `geometry` (prototyp `prototype/whole-wheel-resize`, `VERDICT.md`)."""
    tire = geometry.tire
    up = up_angle(tire)
    angles = up + np.deg2rad(np.asarray(BODY_POINT_DEGREES, dtype=np.float64))
    xs, ys = tire.from_unit(BODY_POINT_RHO * np.cos(angles), BODY_POINT_RHO * np.sin(angles))
    positives = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
    wheel = _bool(wheel_mask)
    height, width = wheel.shape
    around = np.deg2rad(np.arange(0, 360, 45))
    xs, ys = tire.from_unit(BODY_NEGATIVE_RHO * np.cos(around), BODY_NEGATIVE_RHO * np.sin(around))
    negatives = [(geometry.rim.cx, geometry.rim.cy)]
    for x, y in zip(xs, ys, strict=True):
        col, row = round(x), round(y)
        if 0 <= col < width and 0 <= row < height and wheel[row, col]:
            negatives.append((float(x), float(y)))
    return BodyPrompt(
        positives=positives,
        negatives=negatives,
        gap=arch_gap_points(geometry, wheel_mask, BODY_GAP_DEGREES),
        liner=arch_gap_points(geometry, wheel_mask),
    )


def arch_gap_points(
    geometry: RimGeometry,
    wheel_mask: Image.Image,
    degrees: Sequence[float] = ARCH_GAP_DEGREES,
) -> list[Point]:
    """Punkty tuż nad widocznym brzegiem opony (wnętrze nadkola, jeśli między oponą a błotnikiem
    jest szczelina): promienie od środka koła w górę (`degrees` od pionu), pierwszy piksel poza
    maską koła + `ARCH_GAP_OFFSET`. Promień, który wychodzi z koła poza zdjęciem, nie daje
    punktu."""
    tire = geometry.tire
    angles = up_angle(tire) + np.deg2rad(np.asarray(degrees, dtype=np.float64))
    rays = wheel_rays(tire, _bool(wheel_mask), angles=angles, steps=np.arange(0.5, 1.6, 0.002))
    points = []
    for angle, leaves, on_image in zip(angles, rays.leaves, rays.exit_on_image, strict=True):
        if not on_image:
            continue
        edge = rays.steps[leaves] + ARCH_GAP_OFFSET
        xs, ys = tire.from_unit(
            np.array([edge * math.cos(angle)]), np.array([edge * math.sin(angle)])
        )
        points.append((float(xs[0]), float(ys[0])))
    return points


def foreign_occlusion(tire: Ellipse, wheel_mask: Image.Image, body_mask: Image.Image) -> float:
    """Część elipsy opony, której nie ma ani w masce koła, ani w karoserii: koło zasłania coś
    obcego (inne auto, słupek), na co powiększone koło weszłoby, bo nie jest chronione."""
    wheel = _bool(wheel_mask)
    body = _bool(body_mask)
    height, width = wheel.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    inside = tire.radius(xs, ys) <= 1
    return _share(inside & ~wheel & ~body, inside)


def clean_body_mask(body_mask: Image.Image, wheel_mask: Image.Image) -> Image.Image:
    """Maska karoserii bez siatki dziur, którą SAM daje przy niepewnej masce: domknięcie
    małym dyskiem i wypełnienie dziur zamkniętych w masce; piksele koła zawsze odpadają
    (koło SAM wygrywa na wspólnym brzegu)."""
    body = _bool(body_mask)
    wheel = _bool(wheel_mask)
    ys, xs = np.nonzero(wheel)
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 if xs.size else 0
    size = max(3, 2 * round(side * BODY_CLOSING / 2) + 1)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, disk) > 0
    # Dziury = tło niepołączone z brzegiem obrazu.
    _count, labels = cv2.connectedComponents((~closed).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    holes = (labels > 0) & ~np.isin(labels, border)
    return _image((closed | holes) & ~wheel)


def outline_body(image: Image.Image, body_mask: Image.Image) -> Image.Image:
    """Obraz diagnostyczny: karoseria (`body_mask` w rozmiarze `image`) jako czerwona
    poświata; wynik RGB."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    body = _bool(body_mask)
    tint = np.asarray(BODY_TINT, dtype=np.float32)
    rgb[body] = (1 - BODY_TINT_ALPHA) * rgb[body] + BODY_TINT_ALPHA * tint
    return Image.fromarray(np.rint(rgb).astype(np.uint8))


def up_angle(ellipse: Ellipse) -> float:
    """Kąt promienia biegnącego od środka elipsy pionowo w górę obrazu, w układzie elipsy
    (`to_unit`; uwzględnia obrót): oś strefy błotnika i punktów nad kołem."""
    us, vs = ellipse.to_unit(np.array([ellipse.cx]), np.array([ellipse.cy - 1.0]))
    return math.atan2(float(vs[0]), float(us[0]))


def _share(part: BoolArray, whole: BoolArray) -> float:
    return float(part.sum()) / max(int(whole.sum()), 1)


def _bool(mask: Image.Image) -> BoolArray:
    return np.asarray(np.asarray(mask.convert("L")) > 0)


def _image(mask: BoolArray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255)
