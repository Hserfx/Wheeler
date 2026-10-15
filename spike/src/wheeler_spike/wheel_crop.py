"""Wycinanie okolicy koła i kompozycja wyniku z powrotem w oryginał.

Kod wielokrotnego użytku dla ticketu 03: czysty numpy + Pillow, bez modelu.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
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


def crop_box_for(wheel: Box, image_size: tuple[int, int], margin: float = 0.25) -> Box:
    """Kwadratowy wycinek wokół koła, z marginesem `margin` × bok koła z każdej strony.

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


def composite(original: Image.Image, generated: Image.Image, crop: Box, mask: Image.Image, feather: int = 6) -> Image.Image:
    """Wkleja wynik modelu (dla wycinka `crop`) w oryginał przez maskę koła.

    `mask` ma rozmiar oryginału (L, >0 = koło). Krawędź jest wygładzana tylko do wewnątrz maski,
    więc każdy piksel poza maską (i poza wycinkiem) pozostaje bit w bit z oryginału.
    """
    base = original.convert("RGB")
    patch = generated.convert("RGB").resize((crop.width, crop.height), Image.Resampling.LANCZOS)
    alpha = feathered_alpha(mask.crop(crop), feather)[..., None]
    region = np.asarray(base.crop(crop), dtype=np.float32)
    blended = np.rint(alpha * np.asarray(patch, dtype=np.float32) + (1 - alpha) * region).astype(np.uint8)
    blended = np.where(alpha > 0, blended, np.asarray(base.crop(crop)))
    result = base.copy()
    result.paste(Image.fromarray(blended), (crop.left, crop.top))
    return result


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


def feathered_alpha(mask: Image.Image, feather: int) -> np.ndarray:
    """Alfa w [0, 1]: 0 na krawędzi i poza maską, rośnie do 1 na pasie ~`feather` px wewnątrz maski."""
    binary = mask.convert("L").point(lambda v: 255 if v > 0 else 0)
    inside = np.asarray(binary, dtype=np.float32) / 255
    if feather <= 0:
        return inside
    # Rozmyta maska ma ~0.5 na krawędzi; przeskalowanie [0.5, 0.98] -> [0, 1] daje rampę
    # leżącą w całości wewnątrz maski, a mnożenie przez `inside` gwarantuje zero na zewnątrz.
    soft = np.asarray(binary.filter(ImageFilter.GaussianBlur(feather / 2)), dtype=np.float32) / 255
    return np.clip((soft - 0.5) / 0.48, 0, 1) * inside


def _snap(value: float, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _fit_span(center: float, length: int, limit: int) -> tuple[int, int]:
    length = min(length, limit)
    start = round(center - length / 2)
    start = max(0, min(start, limit - length))
    return start, start + length
