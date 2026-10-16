"""Atrapa modeli (`WHEELER_PROCESSOR=placeholder`): cały pipeline bez GPU i bez wag.

Przydaje się do pracy nad UI: „detektor” znajduje dwa koła w stałych miejscach (dolne rogi
zdjęcia), „SAM 2” zwraca okrąg wokół kliknięcia albo wpisany w box, a „mflux” zwraca
wycinek w odcieniach szarości. Kompozycja, pliki i statusy są prawdziwe.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver
from wheeler_worker.rim_editor import StepCallback
from wheeler_worker.wheel_swap import WheelSwapProcessor


class _FakeWeights:
    """„Ładowanie” atrapy: przechodzi przez `LoadObserver`, więc UI pokazuje stany modelu."""

    def __init__(self, observer: LoadObserver) -> None:
        self.observer = observer
        self.loaded = False

    def load(self) -> None:
        if not self.loaded:
            with self.observer.loading():
                self.loaded = True

    def release(self) -> None:
        self.loaded = False


class CornerDetector:
    """Dwa „koła” w dolnej części zdjęcia, po lewej i po prawej."""

    def __init__(self, observer: LoadObserver = NO_OBSERVER) -> None:
        self._weights = _FakeWeights(observer)

    def detect(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
        self._weights.load()
        width, height = image.size
        side = min(width, height) * 0.2
        top = height * 0.7
        return [
            (width * 0.2, top, width * 0.2 + side, top + side),
            (width * 0.8 - side, top, width * 0.8, top + side),
        ]

    def release(self) -> None:
        self._weights.release()


class CircleSegmenter:
    def __init__(self, radius_fraction: float = 0.08, observer: LoadObserver = NO_OBSERVER) -> None:
        self.radius_fraction = radius_fraction
        self._weights = _FakeWeights(observer)

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        self._weights.load()
        width, height = image.size
        radius = max(2.0, min(width, height) * self.radius_fraction)
        yy, xx = np.mgrid[0:height, 0:width]
        inside = (xx - point[0]) ** 2 + (yy - point[1]) ** 2 <= radius * radius
        return Image.fromarray(inside.astype(np.uint8) * 255)

    def segment_boxes(
        self, image: Image.Image, boxes: list[tuple[float, float, float, float]]
    ) -> list[Image.Image]:
        width, height = image.size
        yy, xx = np.mgrid[0:height, 0:width]
        masks = []
        for x0, y0, x1, y1 in boxes:
            cx, cy, radius = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
            masks.append(Image.fromarray(inside.astype(np.uint8) * 255))
        return masks

    def release(self) -> None:
        self._weights.release()


class GrayscaleEditor:
    def __init__(self, observer: LoadObserver = NO_OBSERVER) -> None:
        self._weights = _FakeWeights(observer)

    def edit(
        self,
        images: list[Image.Image],
        *,
        mask: Image.Image,
        prompt: str,
        seed: int,
        size: tuple[int, int],
        resolution: int,
        on_step: StepCallback | None = None,
    ) -> Image.Image:
        self._weights.load()
        if on_step is not None:
            on_step(1, 1)
        return ImageOps.grayscale(images[0].resize(size)).convert("RGB")

    def release(self) -> None:
        self._weights.release()


def placeholder_processor(observer: LoadObserver = NO_OBSERVER) -> WheelSwapProcessor:
    return WheelSwapProcessor(
        detector=CornerDetector(observer),
        segmenter=CircleSegmenter(observer=observer),
        editor=GrayscaleEditor(observer),
    )
