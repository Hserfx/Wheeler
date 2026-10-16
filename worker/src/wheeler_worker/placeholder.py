"""Atrapa modeli (`WHEELER_PROCESSOR=placeholder`): cały pipeline bez GPU i bez wag.

Przydaje się do pracy nad UI: „SAM 2” zwraca okrąg wokół kliknięcia, a „mflux” zwraca
wycinek w odcieniach szarości. Kompozycja, pliki i statusy są prawdziwe.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from wheeler_worker.wheel_swap import WheelSwapProcessor


class CircleSegmenter:
    def __init__(self, radius_fraction: float = 0.08) -> None:
        self.radius_fraction = radius_fraction

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        width, height = image.size
        radius = max(2.0, min(width, height) * self.radius_fraction)
        yy, xx = np.mgrid[0:height, 0:width]
        inside = (xx - point[0]) ** 2 + (yy - point[1]) ** 2 <= radius * radius
        return Image.fromarray(inside.astype(np.uint8) * 255)

    def release(self) -> None:
        pass


class GrayscaleEditor:
    def edit(
        self,
        images: list[Image.Image],
        *,
        mask: Image.Image,
        prompt: str,
        seed: int,
        size: tuple[int, int],
        resolution: int,
    ) -> Image.Image:
        return ImageOps.grayscale(images[0].resize(size)).convert("RGB")

    def release(self) -> None:
        pass


def placeholder_processor() -> WheelSwapProcessor:
    return WheelSwapProcessor(segmenter=CircleSegmenter(), editor=GrayscaleEditor())
