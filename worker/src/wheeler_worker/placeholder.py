"""Atrapa modeli (`WHEELER_PROCESSOR=placeholder`): cały pipeline bez GPU i bez wag.

Przydaje się do pracy nad UI: „SAM 2” zwraca okrąg wokół kliknięcia, a „mflux” zwraca
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
        segmenter=CircleSegmenter(observer=observer), editor=GrayscaleEditor(observer)
    )
