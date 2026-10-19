"""Rozmiar felgi sterowany geometrią (plus-sizing, ticket 12): elipsa felgi i radialny warp.

Sam prompt rozmiaru model ignoruje (trzyma geometrię wycinka), więc przy stopniu ≠ 0 worker
przeskalowuje felgę w wycinku, zanim ten trafi do modelu (prototyp `prototype/plus-sizing-warp`).

Czysty numpy + OpenCV, bez modeli. Układ: piksele obrazu, elipsy jak w `cv2.fitEllipse`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

# Zmiana średnicy felgi na cal stopnia jako ułamek średnicy koła (zewnętrznej krawędzi opony).
# Cale są przybliżone: obecnego rozmiaru felgi nie znamy, sterujemy proporcją felga/koło.
RIM_FRACTION_PER_INCH = 0.04

# Średnica felgi (pionowa, jak w ujęciu 3/4 nie zawyża jej bieżnik) jako ułamek wysokości maski
# koła: poza tym zakresem to nie felga (piasta, logo, całe koło) albo pomiar się nie udał.
RIM_DIAMETER_RANGE = (0.40, 0.95)
# Maska kandydata musi leżeć w masce koła (SAM czasem dokleja kawałek karoserii).
RIM_INSIDE_WHEEL = 0.95
# Zewnętrzna krawędź opony = elipsa felgi * k. k z profilu maski koła; poza zakresem
# (zasłonięte lico opony, maska z nadkolem) przyjmujemy typowe k (rozpoznanie na 10 kołach).
K_RANGE = (1.15, 1.6)
TYPICAL_K = 1.32
# k = ten percentyl profilu promieniowego: strona lica (bieżnik po drugiej stronie wydłuża
# profil w ujęciu 3/4 do 2,3-2,6).
K_PERCENTILE = 20
# Brzeg felgi graniczy z oponą, gdy w pasie tej szerokości (ułamek wysokości koła) za nim jest
# opona; brzegi ucięte zasłonięciem odpadają.
TIRE_BAND = 0.02
MIN_EDGE_POINTS = 20

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
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

    def radius(self, xs: FloatArray, ys: FloatArray) -> FloatArray:
        """Promień znormalizowany do tej elipsy (1 = na brzegu) w punktach (`xs`, `ys`)."""
        angle = math.radians(self.angle)
        dx, dy = xs - self.cx, ys - self.cy
        u = dx * math.cos(angle) + dy * math.sin(angle)
        v = -dx * math.sin(angle) + dy * math.cos(angle)
        rho = np.sqrt((u / (self.width / 2)) ** 2 + (v / (self.height / 2)) ** 2)
        return np.asarray(rho, dtype=np.float32)

    def half_height(self) -> float:
        """Połowa pionowej rozpiętości (obróconej) elipsy."""
        angle = math.radians(self.angle)
        a, b = self.width / 2, self.height / 2
        return math.hypot(a * math.sin(angle), b * math.cos(angle))

    def as_cv2(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return (self.cx, self.cy), (self.width, self.height), self.angle


@dataclass(frozen=True)
class RimGeometry:
    """Felga koła na zdjęciu: jej elipsa i zewnętrzna krawędź opony = elipsa felgi * `k`
    (felga i ściana boczna opony są koncentryczne, także w ujęciu 3/4)."""

    rim: Ellipse
    k: float

    @property
    def tire(self) -> Ellipse:
        return self.rim.scaled(self.k)

    def shifted(self, dx: float, dy: float) -> RimGeometry:
        return RimGeometry(self.rim.shifted(dx, dy), self.k)


def warp_rim(image: Image.Image, geometry: RimGeometry, *, rim_size: int) -> Image.Image:
    """Felga większa albo mniejsza o `rim_size` cali, zewnętrzna krawędź opony bez zmian.

    Radialnie w układzie elipsy opony (rho = 1 na zewnętrznej krawędzi opony), rho_f = felga,
    rho_c = rho_f + `rim_size` * 4%: [0, rho_f] -> [0, rho_c] liniowo (felga),
    [rho_f, 1] -> [rho_c, 1] liniowo (bok opony), rho ≥ 1 bez zmian (piksele bit w bit).
    Docelowego rozmiaru nie przycinamy ani nie korygujemy. `geometry` w pikselach `image`;
    wynik RGB.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    tire = geometry.tire
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    rho = tire.radius(xs, ys)
    rho_rim = 1 / geometry.k
    rho_new = rho_rim + rim_size * RIM_FRACTION_PER_INCH
    # Odwzorowanie odwrotne (cel -> źródło), jak wymaga `cv2.remap`.
    source = np.where(
        rho < rho_new,
        rho * rho_rim / rho_new,
        rho_rim + (rho - rho_new) * (1 - rho_rim) / (1 - rho_new),
    )
    # Promień skaluje wektor od środka liniowo, więc obrót elipsy nie wymaga osobnego układu.
    scale = np.divide(source, rho, out=np.ones_like(rho), where=rho > 1e-6)
    map_x = (tire.cx + (xs - tire.cx) * scale).astype(np.float32)
    map_y = (tire.cy + (ys - tire.cy) * scale).astype(np.float32)
    warped = cv2.remap(rgb, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    inside = rho < 1
    return Image.fromarray(np.where(inside[..., None], warped, rgb))


def measure_rim(wheel_mask: Image.Image, candidates: Sequence[RimCandidate]) -> RimGeometry | None:
    """Elipsa felgi i krawędź opony koła z maski koła i kandydatów SAM 2 na maskę felgi.

    Kandydat: największa składowa, elipsa z `cv2.fitEllipse` na jej konturze; odpada, gdy jego
    średnica wypada poza `RIM_DIAMETER_RANGE` wysokości koła albo wystaje z maski koła.
    Wygrywa maksimum IoU(maska, elipsa) * ocena SAM. Ostateczna elipsa jest dopasowana tylko do
    brzegu felgi graniczącego z oponą (odporność na zasłonięcie). `None`: felgi nie wyznaczono.
    """
    wheel = np.asarray(wheel_mask.convert("L")) > 0
    rows = np.nonzero(wheel.any(axis=1))[0]
    if rows.size == 0:
        return None
    wheel_half_height = (rows[-1] + 1 - rows[0]) / 2
    best: tuple[float, BoolArray] | None = None
    for mask_image, score in candidates:
        rim = _largest_component(np.asarray(mask_image.convert("L")) > 0)
        if rim is None or (rim & ~wheel).sum() > (1 - RIM_INSIDE_WHEEL) * rim.sum():
            continue
        contour = _outer_contour(rim)
        if len(contour) < 5:
            continue
        ellipse = _fit(contour)
        if not _plausible(ellipse, wheel_half_height):
            continue
        quality = _iou(rim, ellipse) * score
        if best is None or quality > best[0]:
            best = (quality, rim)
    if best is None:
        return None
    ellipse_or_none = _fit_tire_edge(best[1], wheel, band=wheel_half_height * 2 * TIRE_BAND)
    if ellipse_or_none is None or not _plausible(ellipse_or_none, wheel_half_height):
        return None
    return RimGeometry(ellipse_or_none, _tire_k(ellipse_or_none, wheel))


def _plausible(ellipse: Ellipse, wheel_half_height: float) -> bool:
    low, high = RIM_DIAMETER_RANGE
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


def _tire_k(rim: Ellipse, wheel: BoolArray) -> float:
    """Odległość od środka felgi do brzegu maski koła w jednostkach elipsy felgi (1 = brzeg
    felgi), percentyl `K_PERCENTILE` po kątach; poza `K_RANGE`: `TYPICAL_K`."""
    height, width = wheel.shape
    angles = np.deg2rad(np.arange(0, 360, 10))[:, None]
    steps = np.arange(1.0, 3.0, 0.005)[None, :]
    turn = math.radians(rim.angle)
    dx, dy = rim.width / 2 * np.cos(angles), rim.height / 2 * np.sin(angles)
    vx = dx * math.cos(turn) - dy * math.sin(turn)
    vy = dx * math.sin(turn) + dy * math.cos(turn)
    xs = np.rint(rim.cx + steps * vx).astype(int)
    ys = np.rint(rim.cy + steps * vy).astype(int)
    on_image = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    in_wheel = on_image & wheel[np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)]
    # Pierwszy krok poza kołem; kąt, w którym już brzeg felgi jest poza kołem, pomijamy.
    leaves = np.where(in_wheel.all(axis=1), steps.shape[1], np.argmin(in_wheel, axis=1))
    profile = steps[0, np.maximum(leaves - 1, 0)][leaves > 0]
    if profile.size == 0:
        return TYPICAL_K
    k = float(np.percentile(profile, K_PERCENTILE))
    low, high = K_RANGE
    return k if low <= k <= high else TYPICAL_K


def ungrown_wheel_mask(mask: Image.Image, mask_grow: float) -> Image.Image:
    """Maska koła bez poszerzenia o `mask_grow` boku koła (`wheel_swap._save_mask`), czyli
    w przybliżeniu maska SAM: jej brzeg to zewnętrzna krawędź opony."""
    inside = np.asarray(mask.convert("L")) > 0
    ys, xs = np.nonzero(inside)
    if xs.size == 0:
        return Image.fromarray(inside.astype(np.uint8) * 255)
    side = max(xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    pixels = round(side / (1 + 2 * mask_grow) * mask_grow)
    if pixels > 0:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
        inside = cv2.erode(inside.astype(np.uint8), disk) > 0
    return Image.fromarray(inside.astype(np.uint8) * 255)


def outline_rim(image: Image.Image, geometry: RimGeometry) -> Image.Image:
    """Obraz diagnostyczny: elipsa felgi (zielona) i zewnętrzna krawędź opony (żółta)."""
    rgb = np.array(image.convert("RGB"))  # kopia zapisywalna dla cv2
    thickness = max(1, image.width // 300)
    cv2.ellipse(rgb, geometry.rim.as_cv2(), (0, 255, 0), thickness)
    cv2.ellipse(rgb, geometry.tire.as_cv2(), (255, 220, 0), thickness)
    return Image.fromarray(rgb)
