"""Elipsa felgi i krawędź opony koła (ticket 12): pomiar przed zmianą rozmiaru koła.

Sam prompt rozmiaru model ignoruje (trzyma geometrię wycinka), więc przy stopniu ≠ 0 worker
mierzy felgę i oponę (SAM 2 + ten moduł), a `wheel_resize` zmienia rozmiar koła w wycinku,
zanim ten trafi do modelu (ticket 16).

Czysty numpy + OpenCV, bez modeli. Układ: piksele obrazu, elipsy jak w `cv2.fitEllipse`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from wheeler_worker.compositing import mask_box

# Średnica felgi (pionowa, jak w ujęciu 3/4 nie zawyża jej bieżnik) jako ułamek wysokości maski
# koła. Kandydata SAM wybieramy tylko z tego zakresu (reguła z rozpoznania, `rim_ellipse/NOTES.md`):
# poza nim to nie felga (piasta, logo, całe koło).
RIM_CANDIDATE_RANGE = (0.45, 0.92)
# Ostateczna elipsa (dopasowana do brzegu przy oponie) może od kandydata trochę odejść; poza tym
# zakresem pomiar się nie udał.
RIM_DIAMETER_RANGE = (0.40, 0.95)
# Maska kandydata musi leżeć w masce koła (SAM czasem dokleja kawałek karoserii).
RIM_INSIDE_WHEEL = 0.95
# Zewnętrzna krawędź opony = elipsa felgi * k (`RimGeometry.tire_ratio`). k z profilu maski koła;
# poza zakresem (zasłonięte lico opony, maska z nadkolem) przyjmujemy typowe k (rozpoznanie na
# 10 kołach).
TIRE_RATIO_RANGE = (1.15, 1.6)
TYPICAL_TIRE_RATIO = 1.32
# k = ten percentyl profilu promieniowego: strona lica (bieżnik po drugiej stronie wydłuża
# profil w ujęciu 3/4 do 2,3-2,6).
TIRE_RATIO_PERCENTILE = 20
# Brzeg felgi graniczy z oponą, gdy w pasie tej szerokości (ułamek wysokości koła) za nim jest
# opona; brzegi ucięte zasłonięciem odpadają.
TIRE_BAND = 0.02
MIN_EDGE_POINTS = 20

FloatArray = npt.NDArray[np.floating[Any]]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
# Kandydat na maskę samej felgi (SAM 2, multimask): maska L (>0 = felga) i ocena IoU modelu.
RimCandidate = tuple[Image.Image, float]


@dataclass(frozen=True)
class Ellipse:
    """Elipsa jak w `cv2.fitEllipse`: środek, pełne osie (`width` wzdłuż obróconej osi x)
    i obrót `angle` w stopniach (zgodnie z ruchem wskazówek zegara w układzie obrazu)."""

    cx: float
    cy: float
    width: float
    height: float
    angle: float

    def scaled(self, factor: float) -> Ellipse:
        """Ta sama elipsa (środek, obrót) z osiami razy `factor`."""
        return Ellipse(self.cx, self.cy, self.width * factor, self.height * factor, self.angle)

    def shifted(self, dx: float, dy: float) -> Ellipse:
        return Ellipse(self.cx + dx, self.cy + dy, self.width, self.height, self.angle)

    def to_unit(self, xs: FloatArray, ys: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Punkty obrazu (`xs`, `ys`) w układzie tej elipsy: obrót do jej osi i podział przez
        półosie, więc brzeg elipsy to okrąg jednostkowy."""
        cos, sin = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        dx, dy = xs - self.cx, ys - self.cy
        return (dx * cos + dy * sin) / (self.width / 2), (dy * cos - dx * sin) / (self.height / 2)

    def from_unit(self, us: FloatArray, vs: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Odwrotność `to_unit`: punkty układu elipsy w pikselach obrazu."""
        cos, sin = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        du, dv = us * (self.width / 2), vs * (self.height / 2)
        return self.cx + du * cos - dv * sin, self.cy + du * sin + dv * cos

    def radius(self, xs: FloatArray, ys: FloatArray) -> FloatArray:
        """Promień znormalizowany do tej elipsy (1 = na brzegu) w punktach (`xs`, `ys`)."""
        return np.hypot(*self.to_unit(xs, ys))

    def half_height(self) -> float:
        """Połowa pionowej rozpiętości (obróconej) elipsy."""
        angle = math.radians(self.angle)
        a, b = self.width / 2, self.height / 2
        return math.hypot(a * math.sin(angle), b * math.cos(angle))

    def as_cv2(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return (self.cx, self.cy), (self.width, self.height), self.angle


@dataclass(frozen=True)
class RimGeometry:
    """Felga koła na zdjęciu: jej elipsa i zewnętrzna krawędź opony = elipsa felgi
    * `tire_ratio` (k; felga i ściana boczna opony są koncentryczne, także w ujęciu 3/4)."""

    rim: Ellipse
    tire_ratio: float

    @property
    def tire(self) -> Ellipse:
        return self.rim.scaled(self.tire_ratio)

    def shifted(self, dx: float, dy: float) -> RimGeometry:
        return RimGeometry(self.rim.shifted(dx, dy), self.tire_ratio)


def wheel_ellipse(wheel_mask: Image.Image) -> Ellipse | None:
    """Elipsa (bez obrotu) wpisana w bbox maski koła, w układzie środków pikseli jak
    `cv2.fitEllipse`; `None` dla pustej maski."""
    box = mask_box(wheel_mask)
    if box is None:
        return None
    center_x, center_y = (box.left + box.right - 1) / 2, (box.top + box.bottom - 1) / 2
    return Ellipse(center_x, center_y, box.width, box.height, 0.0)


def measure_rim(wheel_mask: Image.Image, candidates: Sequence[RimCandidate]) -> RimGeometry | None:
    """Elipsa felgi i krawędź opony koła z maski koła i kandydatów SAM 2 na maskę felgi.

    Kandydat: największa składowa, elipsa z `cv2.fitEllipse` na jej konturze; odpada, gdy jego
    średnica wypada poza `RIM_CANDIDATE_RANGE` wysokości koła albo wystaje z maski koła.
    Wygrywa maksimum IoU(maska, elipsa) * ocena SAM. Ostateczna elipsa jest dopasowana tylko do
    brzegu felgi graniczącego z oponą (odporność na zasłonięcie) i musi mieścić się
    w `RIM_DIAMETER_RANGE`. `None`: felgi nie wyznaczono.
    """
    wheel_box = wheel_ellipse(wheel_mask)
    if wheel_box is None:
        return None
    wheel = np.asarray(wheel_mask.convert("L")) > 0
    wheel_half_height = wheel_box.height / 2
    best: tuple[float, BoolArray] | None = None
    for mask_image, score in candidates:
        rim = _largest_component(np.asarray(mask_image.convert("L")) > 0)
        if rim is None or (rim & ~wheel).sum() > (1 - RIM_INSIDE_WHEEL) * rim.sum():
            continue
        contour = _outer_contour(rim)
        if len(contour) < 5:
            continue
        ellipse = _fit(contour)
        if not _within(ellipse, wheel_half_height, RIM_CANDIDATE_RANGE):
            continue
        quality = _iou(rim, ellipse) * score
        if best is None or quality > best[0]:
            best = (quality, rim)
    if best is None:
        return None
    rim_ellipse = _fit_tire_edge(best[1], wheel, band=wheel_half_height * 2 * TIRE_BAND)
    if rim_ellipse is None or not _within(rim_ellipse, wheel_half_height, RIM_DIAMETER_RANGE):
        return None
    return RimGeometry(rim_ellipse, _tire_ratio(rim_ellipse, wheel))


def _within(
    ellipse: Ellipse, wheel_half_height: float, diameter_range: tuple[float, float]
) -> bool:
    low, high = diameter_range
    return low <= ellipse.half_height() / wheel_half_height <= high


def _largest_component(mask: BoolArray) -> BoolArray | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == largest)


def _outer_contour(mask: BoolArray) -> npt.NDArray[np.int32]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.asarray(max(contours, key=cv2.contourArea)[:, 0, :], dtype=np.int32)


def _fit(points: npt.NDArray[np.int32]) -> Ellipse:
    (cx, cy), (width, height), angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
    return Ellipse(float(cx), float(cy), float(width), float(height), float(angle))


def _iou(mask: BoolArray, ellipse: Ellipse) -> float:
    drawn = np.zeros(mask.shape, dtype=np.uint8)
    cv2.ellipse(drawn, ellipse.as_cv2(), 1, -1)
    inside = drawn > 0
    union = int((mask | inside).sum())
    return float((mask & inside).sum()) / max(union, 1)


def _fit_tire_edge(rim: BoolArray, wheel: BoolArray, *, band: float) -> Ellipse | None:
    """Elipsa tylko z punktów konturu felgi, za którymi (na zewnątrz, w pasie `band`) leży
    opona = maska koła minus felga. Brzeg ucięty zasłonięciem nie ma za sobą koła i odpada."""
    contour = _outer_contour(rim)
    tire = (wheel & ~rim).astype(np.float32)
    side = 2 * max(1, round(band)) + 1
    tire_near = cv2.boxFilter(tire, -1, (side, side), normalize=True)
    edge = contour[tire_near[contour[:, 1], contour[:, 0]] > 0.2]
    if len(edge) < MIN_EDGE_POINTS:
        return None
    return _fit(edge)


def _tire_ratio(rim: Ellipse, wheel: BoolArray) -> float:
    """Odległość od środka felgi do brzegu maski koła w jednostkach elipsy felgi (1 = brzeg
    felgi), percentyl `TIRE_RATIO_PERCENTILE` po kątach; poza `TIRE_RATIO_RANGE`:
    `TYPICAL_TIRE_RATIO`."""
    rays = wheel_rays(
        rim,
        wheel,
        angles=np.deg2rad(np.arange(0, 360, 10)),
        steps=np.arange(1.0, 3.0, 0.005),
    )
    # Kąt, w którym już brzeg felgi jest poza kołem, pomijamy.
    profile = rays.steps[np.maximum(rays.leaves - 1, 0)][rays.leaves > 0]
    if profile.size == 0:
        return TYPICAL_TIRE_RATIO
    ratio = float(np.percentile(profile, TIRE_RATIO_PERCENTILE))
    low, high = TIRE_RATIO_RANGE
    return ratio if low <= ratio <= high else TYPICAL_TIRE_RATIO


@dataclass(frozen=True)
class WheelRays:
    """Promienie od środka elipsy pod kątami `angles` (radiany, w układzie elipsy, `to_unit`),
    próbkowane krokami `steps` (w jednostkach elipsy, 1 = jej brzeg). `leaves`: indeks
    pierwszego kroku poza maską koła (piksele spoza zdjęcia są poza kołem; `len(steps)`:
    promień nie wychodzi z koła). `exit_xs`, `exit_ys`: piksel tego kroku przycięty do
    zdjęcia, prawdziwy tylko przy `exit_on_image` (promień, który wychodzi z koła poza
    zdjęciem, nie trafia na krawędź opony i jego użytkownicy go odrzucają)."""

    steps: FloatArray
    leaves: IntArray
    exit_on_image: BoolArray
    exit_xs: IntArray
    exit_ys: IntArray


def wheel_rays(
    ellipse: Ellipse, wheel: BoolArray, *, angles: FloatArray, steps: FloatArray
) -> WheelRays:
    """Marsz promieni od środka `ellipse` do wyjścia z maski koła `wheel`."""
    height, width = wheel.shape
    ray_xs, ray_ys = ellipse.from_unit(
        steps[None, :] * np.cos(angles)[:, None], steps[None, :] * np.sin(angles)[:, None]
    )
    xs, ys = np.rint(ray_xs).astype(np.intp), np.rint(ray_ys).astype(np.intp)
    on_image = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys = np.clip(xs, 0, width - 1), np.clip(ys, 0, height - 1)
    in_wheel = on_image & wheel[ys, xs]
    leaves = np.where(in_wheel.all(axis=1), steps.size, np.argmin(in_wheel, axis=1))
    exits = np.minimum(leaves, steps.size - 1)
    rows = np.arange(leaves.size)
    return WheelRays(
        steps=steps,
        leaves=leaves,
        exit_on_image=on_image[rows, exits] & (leaves < steps.size),
        exit_xs=xs[rows, exits],
        exit_ys=ys[rows, exits],
    )


# Kolory diagnostyki (RGB): felga, krawędź opony.
RIM_COLOR = (0, 255, 0)
TIRE_COLOR = (255, 220, 0)


def outline_rim(
    image: Image.Image,
    geometry: RimGeometry,
    *,
    rim_color: tuple[int, int, int] = RIM_COLOR,
    tire_color: tuple[int, int, int] = TIRE_COLOR,
) -> Image.Image:
    """Obraz diagnostyczny: elipsa felgi (domyślnie zielona) i zewnętrzna krawędź opony
    (domyślnie żółta)."""
    rgb = np.array(image.convert("RGB"))  # kopia zapisywalna dla cv2
    thickness = max(1, image.width // 300)
    cv2.ellipse(rgb, geometry.rim.as_cv2(), rim_color, thickness)
    cv2.ellipse(rgb, geometry.tire.as_cv2(), tire_color, thickness)
    return Image.fromarray(rgb)
