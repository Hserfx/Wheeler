"""Zmiana rozmiaru całego koła (ticket 16): felga i zewnętrzna krawędź opony rosną albo maleją
o tyle samo cali, a punkt styku z ziemią stoi.

Sam prompt rozmiaru model ignoruje (trzyma geometrię wycinka), więc przy stopniu ≠ 0 worker
dostarcza modelowi wycinek z kołem już w nowym rozmiarze, za karoserią, która zostaje bit w bit
(prototyp `prototype/whole-wheel-resize`, `VERDICT.md`).

Czysty numpy + OpenCV, bez modeli. Układ: piksele obrazu, elipsy jak w `rim_geometry`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from wheeler_worker.compositing import Box, grow_wheel_mask, mask_box
from wheeler_worker.rim_geometry import BoolArray, Ellipse, RimGeometry

# Rant felgi (kołnierz ~17-18 mm z każdej strony, ETRTO): widoczny na zdjęciu, a nieliczony
# w rozmiarze nominalnym felgi. Stała z grillingu ticketu 13, niesprawdzana na zdjęciach
# ze znanym rozmiarem.
RIM_FLANGE_INCHES = 1.4


def inch_px(rim: Ellipse, *, current_rim: int) -> float:
    """Pikseli na cal: średnica widocznej felgi / (obecny rozmiar nominalny + rant). Dłuższa oś
    elipsy, bo perspektywa (3/4) skraca tylko jedną."""
    return max(rim.width, rim.height) / (current_rim + RIM_FLANGE_INCHES)


def lowest_point(ellipse: Ellipse) -> tuple[float, float]:
    """Najniższy punkt (największe y) obróconej elipsy: styk opony z ziemią. W obróconej,
    spłaszczonej elipsie (3/4) nie leży na jej osi."""
    angle = math.radians(ellipse.angle)
    # y(t) = cy + a cos t sin(angle) + b sin t cos(angle): maksimum w kierunku tego wektora.
    du, dv = ellipse.width / 2 * math.sin(angle), ellipse.height / 2 * math.cos(angle)
    norm = math.hypot(du, dv)
    xs, ys = ellipse.from_unit(np.array([du / norm]), np.array([dv / norm]))
    return float(xs[0]), float(ys[0])


def resized_geometry(old: RimGeometry, *, rim_size: float, inch: float) -> RimGeometry:
    """Koło o `rim_size` cali większe (ujemne: mniejsze): felga i zewnętrzna krawędź opony
    rosną o `rim_size` * `inch` px (dłuższa oś), więc bok opony zostaje tej samej grubości.
    Ten sam obrót i proporcje osi; najniższy punkt opony stoi (jednokładność wokół niego)."""
    rim_diameter = max(old.rim.width, old.rim.height)
    tire_diameter = rim_diameter * old.tire_ratio
    rim_scale = (rim_diameter + rim_size * inch) / rim_diameter
    tire_scale = (tire_diameter + rim_size * inch) / tire_diameter
    ground_x, ground_y = lowest_point(old.tire)
    rim = Ellipse(
        ground_x + tire_scale * (old.rim.cx - ground_x),
        ground_y + tire_scale * (old.rim.cy - ground_y),
        old.rim.width * rim_scale,
        old.rim.height * rim_scale,
        old.rim.angle,
    )
    return RimGeometry(rim, old.tire_ratio * tire_scale / rim_scale)


@dataclass(frozen=True)
class ResizePlan:
    """Zmiana rozmiaru jednego koła (rozmiar zdjęcia): `geometry` nowej felgi i opony, `wheel`
    nowe koło widoczne spod karoserii, `edit_mask` suma starego i nowego koła poszerzona
    o `mask_grow` minus karoseria (obszar, który model może zmienić, i maska kompozycji),
    `model_input` zdjęcie RGB z kołem już w nowym rozmiarze (poza `edit_mask` bit w bit
    oryginał)."""

    geometry: RimGeometry
    wheel: Image.Image
    edit_mask: Image.Image
    model_input: Image.Image


def plan_resize(
    photo: Image.Image,
    wheel_mask: Image.Image,
    old: RimGeometry,
    body_mask: Image.Image,
    *,
    rim_size: float,
    inch: float,
    mask_grow: float,
) -> ResizePlan:
    """Koło (`wheel_mask`: maska SAM bez poszerzenia, `old`: jego felga i opona) o `rim_size`
    cali większe albo mniejsze (`inch`: px na cal), za karoserią (`body_mask`), która zostaje
    nietknięta.

    Warp całego koła w układzie elipsy nowej opony (rho = 1 na jej krawędzi), rho_f = felga:
    felga [0, rho_f'] -> [0, rho_f], bok opony [rho_f', 1] -> [rho_f, 1], dalej (bieżnik
    w 3/4) jednokładność z oponą wokół punktu styku. Nowa maska koła = stara maska tym samym
    odwzorowaniem, widoczna tylko poza karoserią. Przy zmniejszaniu odsłonięte nadkole zostaje
    w masce edycji ze starymi pikselami (model je domalowuje)."""
    new = resized_geometry(old, rim_size=rim_size, inch=inch)
    old_wheel = _bool(wheel_mask)
    visible = ~_bool(body_mask)
    window = _window(old_wheel, old, new, mask_grow)
    local = (slice(window.top, window.bottom), slice(window.left, window.right))
    map_x, map_y = _wheel_maps(window, old, new)
    rgb = np.array(photo.convert("RGB"))
    warped = cv2.remap(rgb[local], map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    moved = np.zeros_like(old_wheel)
    moved[local] = (
        cv2.remap(
            old_wheel[local].astype(np.uint8),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    new_wheel = moved & visible
    edit = _bool(grow_wheel_mask(_image(old_wheel | moved), mask_grow)) & visible
    region = rgb[local]
    region[new_wheel[local]] = warped[new_wheel[local]]
    return ResizePlan(
        geometry=new,
        wheel=_image(new_wheel),
        edit_mask=_image(edit),
        model_input=Image.fromarray(rgb),
    )


def _window(old_wheel: BoolArray, old: RimGeometry, new: RimGeometry, mask_grow: float) -> Box:
    """Fragment zdjęcia, w którym warp może cokolwiek zmienić: stare koło (maska i elipsa
    opony) i jego obraz w jednokładności wokół punktu styku, z zapasem na `mask_grow`."""
    height, width = old_wheel.shape
    tire = old.tire
    angle = math.radians(tire.angle)
    half_width = math.hypot(tire.width / 2 * math.cos(angle), tire.height / 2 * math.sin(angle))
    half_height = tire.half_height()
    lefts = [tire.cx - half_width]
    rights = [tire.cx + half_width]
    tops = [tire.cy - half_height]
    bottoms = [tire.cy + half_height]
    box = mask_box(_image(old_wheel))
    if box is not None:
        lefts.append(box.left)
        rights.append(box.right)
        tops.append(box.top)
        bottoms.append(box.bottom)
    scale = new.tire.height / tire.height
    ground_x, ground_y = lowest_point(tire)
    left, right, top, bottom = min(lefts), max(rights), min(tops), max(bottoms)
    left = min(left, ground_x + scale * (left - ground_x))
    right = max(right, ground_x + scale * (right - ground_x))
    top = min(top, ground_y + scale * (top - ground_y))
    bottom = max(bottom, ground_y + scale * (bottom - ground_y))
    pad = math.ceil(max(right - left, bottom - top) * mask_grow) + 2
    return Box(
        max(0, math.floor(left) - pad),
        max(0, math.floor(top) - pad),
        min(width, math.ceil(right) + pad),
        min(height, math.ceil(bottom) + pad),
    )


def _wheel_maps(
    window: Box, old: RimGeometry, new: RimGeometry
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Odwzorowanie odwrotne (cel -> źródło) dla `cv2.remap` w pikselach `window`, w układzie
    elipsy nowej opony (rho = 1 na jej krawędzi): felga [0, rho_f'] -> [0, rho_f], bok opony
    [rho_f', 1] -> [rho_f, 1], dalej jednokładność z oponą."""
    ys, xs = np.mgrid[window.top : window.bottom, window.left : window.right].astype(np.float64)
    us, vs = new.tire.to_unit(xs, ys)
    rho = np.hypot(us, vs)
    rim_old, rim_new = 1 / old.tire_ratio, 1 / new.tire_ratio
    source = np.where(
        rho < rim_new,
        rho * rim_old / rim_new,
        rim_old + (rho - rim_new) * (1 - rim_old) / (1 - rim_new),
    )
    source = np.where(rho >= 1, rho, source)
    scale = np.divide(source, rho, out=np.ones_like(rho), where=rho > 1e-9)
    map_x, map_y = old.tire.from_unit(us * scale, vs * scale)
    return (map_x - window.left).astype(np.float32), (map_y - window.top).astype(np.float32)


def _bool(mask: Image.Image) -> BoolArray:
    return np.asarray(np.asarray(mask.convert("L")) > 0)


def _image(mask: BoolArray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255)
