"""Wycinanie okolicy koła i kompozycja wyniku modelu z powrotem w oryginał.

Czysty numpy + Pillow (erozja maski: OpenCV), bez modelu (przeniesione ze spike'a 01,
`spike/.../wheel_crop.py`). Gwarancja projektu: `composite` nie zmienia żadnego piksela poza maską.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter


class Box(NamedTuple):
    """Prostokąt w pikselach, (left, top) włącznie, (right, bottom) wyłącznie, jak w Pillow."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def mask_box(mask: Image.Image) -> Box | None:
    """Najmniejszy prostokąt obejmujący maskę (>0) albo None dla pustej maski."""
    ys, xs = np.nonzero(np.asarray(mask.convert("L")) > 0)
    if xs.size == 0:
        return None
    return Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def crop_box_for(wheel: Box, image_size: tuple[int, int], margin: float = 0.25) -> Box:
    """Kwadratowy wycinek wokół koła, z marginesem `margin` razy bok koła z każdej strony.

    Wycinek przy krawędzi jest przesuwany do wnętrza zdjęcia. Jeśli kwadrat nie mieści się
    w zdjęciu, w danej osi jest przycinany do pełnego wymiaru zdjęcia.
    """
    image_w, image_h = image_size
    side = round(max(wheel.width, wheel.height) * (1 + 2 * margin))
    left, right = _fit_span((wheel.left + wheel.right) / 2, side, image_w)
    top, bottom = _fit_span((wheel.top + wheel.bottom) / 2, side, image_h)
    return Box(left, top, right, bottom)


def working_size(crop: Box, resolution: int = 1024, multiple: int = 32) -> tuple[int, int]:
    """Rozmiar, do którego skalujemy wycinek przed modelem: pole ≈ resolution², proporcje
    wycinka, oba boki zaokrąglone do wielokrotności `multiple` (mflux wymaga 32)."""
    ratio = crop.width / crop.height
    width = (resolution * resolution * ratio) ** 0.5
    height = width / ratio
    return _snap(width, multiple), _snap(height, multiple)


def composite(
    original: Image.Image, generated: Image.Image, crop: Box, mask: Image.Image, feather: int = 6
) -> Image.Image:
    """Wkleja wynik modelu (dla wycinka `crop`) w oryginał przez maskę koła.

    `mask` ma rozmiar oryginału (L, >0 = koło). Krawędź jest wygładzana tylko do wewnątrz maski,
    więc każdy piksel poza maską (i poza wycinkiem) pozostaje bit w bit z oryginału.
    Oryginał w trybie L/LA daje wynik w tym samym trybie (koło w odcieniach szarości), inne
    tryby RGB/RGBA. Kanał alfa oryginału (RGBA, LA) zostaje bez zmian wszędzie.
    """
    color = "L" if original.mode in ("L", "LA") else "RGB"
    base = original.convert(color)
    patch = generated.convert(color).resize((crop.width, crop.height), Image.Resampling.LANCZOS)
    alpha = feathered_alpha(mask.crop(crop), feather)
    region = np.asarray(base.crop(crop))
    if region.ndim == 3:
        alpha = alpha[..., None]
    mixed = alpha * np.asarray(patch, dtype=np.float32) + (1 - alpha) * region.astype(np.float32)
    blended = np.where(alpha > 0, np.rint(mixed).astype(np.uint8), region)
    result = base.copy()
    result.paste(Image.fromarray(blended), (crop.left, crop.top))
    if original.mode in ("RGBA", "LA"):
        result.putalpha(original.getchannel("A"))
    return result


def grow_wheel_mask(mask: Image.Image, mask_grow: float) -> Image.Image:
    """Maska koła z SAM poszerzona o `mask_grow` razy bok koła: obszar, który generowanie
    może zmienić. Pusta maska zostaje pusta."""
    box = mask_box(mask)
    side = 0 if box is None else max(box.width, box.height)
    return grow_mask(mask, pixels=_grow_pixels(side, mask_grow))


def ungrown_wheel_mask(mask: Image.Image, mask_grow: float) -> Image.Image:
    """Odwrotność `grow_wheel_mask` (w przybliżeniu maska z SAM: jej brzeg to zewnętrzna
    krawędź opony). Bok koła sprzed poszerzenia odtwarzamy z boku `mask`."""
    inside = np.asarray(mask.convert("L")) > 0
    box = mask_box(mask)
    if box is not None:
        pixels = _grow_pixels(max(box.width, box.height) / (1 + 2 * mask_grow), mask_grow)
        if pixels > 0:
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
            inside = cv2.erode(inside.astype(np.uint8), disk) > 0
    return Image.fromarray(inside.astype(np.uint8) * 255)


def _grow_pixels(wheel_side: float, mask_grow: float) -> int:
    return round(wheel_side * mask_grow)


def grow_mask(mask: Image.Image, pixels: int) -> Image.Image:
    """Poszerza maskę o `pixels` px (dylatacja dyskiem), zwraca maskę L 0/255."""
    inside = np.asarray(mask.convert("L")) > 0
    if pixels <= 0 or not inside.any():
        return Image.fromarray(inside.astype(np.uint8) * 255)
    ys, xs = np.nonzero(inside)
    h, w = inside.shape
    top, bottom = max(0, ys.min() - pixels), min(h, ys.max() + pixels + 1)
    left, right = max(0, xs.min() - pixels), min(w, xs.max() + pixels + 1)
    padded = np.pad(inside[top:bottom, left:right], pixels)
    grown = np.zeros_like(padded)
    for dy in range(-pixels, pixels + 1):
        for dx in range(-pixels, pixels + 1):
            if dy * dy + dx * dx <= pixels * pixels:
                grown |= np.roll(padded, (dy, dx), axis=(0, 1))
    out = np.zeros_like(inside)
    out[top:bottom, left:right] = grown[pixels:-pixels, pixels:-pixels]
    return Image.fromarray(out.astype(np.uint8) * 255)


def feathered_alpha(mask: Image.Image, feather: int) -> npt.NDArray[np.float32]:
    """Alfa w [0, 1]: 0 na krawędzi i poza maską, rośnie do 1 na pasie ~`feather` px w masce."""
    binary = mask.convert("L").point(lambda v: 255 if v > 0 else 0)
    inside = np.asarray(binary, dtype=np.float32) / 255
    if feather <= 0:
        return np.asarray(inside, dtype=np.float32)
    # Rozmyta maska ma ~0.5 na krawędzi; przeskalowanie [0.5, 0.98] -> [0, 1] daje rampę
    # leżącą w całości wewnątrz maski, a mnożenie przez `inside` gwarantuje zero na zewnątrz.
    blurred = binary.filter(ImageFilter.GaussianBlur(feather / 2))
    soft = np.asarray(blurred, dtype=np.float32) / 255
    return np.asarray(np.clip((soft - 0.5) / 0.48, 0, 1) * inside, dtype=np.float32)


def _snap(value: float, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _fit_span(center: float, length: int, limit: int) -> tuple[int, int]:
    length = min(length, limit)
    start = round(center - length / 2)
    start = max(0, min(start, limit - length))
    return start, start + length
