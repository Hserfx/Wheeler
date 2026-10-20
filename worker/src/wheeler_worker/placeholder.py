"""Atrapa modeli (`WHEELER_PROCESSOR=placeholder`): cały pipeline bez GPU i bez wag.

Przydaje się do pracy nad UI: „detektor” znajduje dwa koła w stałych miejscach (dolne rogi
zdjęcia), „SAM 2” zwraca okrąg wokół kliknięcia albo wpisany w box (felga: mniejszy okrąg
w środku koła, karoseria: pas nad kołem), „mflux” zwraca wycinek w odcieniach szarości,
a „BiRefNet” zostawia elipsę wpisaną w kadr. Kompozycja, pliki i statusy są prawdziwe.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from wheeler_worker.body_mask import BodySegments, body_prompt
from wheeler_worker.compositing import mask_box
from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver
from wheeler_worker.rim_editor import StepCallback
from wheeler_worker.rim_geometry import RimCandidate, RimGeometry, wheel_ellipse
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

    def segment_rim(self, image: Image.Image, wheel_mask: Image.Image) -> list[RimCandidate]:
        """Felga: elipsa wpisana w bbox maski koła, zmniejszona do 0,7."""
        wheel = wheel_ellipse(wheel_mask)
        if wheel is None:
            return []
        width, height = image.size
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        inside = wheel.scaled(0.7).radius(xx, yy) <= 1
        return [(Image.fromarray(inside.astype(np.uint8) * 255), 0.9)]

    def segment_body(
        self, image: Image.Image, wheel_mask: Image.Image, rim: RimGeometry
    ) -> BodySegments:
        """Karoseria: pas zdjęcia nad kołem, ze szczeliną nadkola 2% boku koła."""
        body = Image.new("L", image.size, 0)
        box = mask_box(wheel_mask)
        if box is not None:
            top = box.top - max(1, round(0.02 * max(box.width, box.height)))
            if top > 0:
                body.paste(255, (0, 0, image.width, top))
        return BodySegments(
            prompt=body_prompt(rim, wheel_mask), with_gap=[(body, 0.9)], without_gap=[], liner=[]
        )

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


class EllipseRemover:
    """Pierwszy plan: elipsa wpisana w obraz (felga zwykle wypełnia kadr)."""

    def __init__(self, observer: LoadObserver = NO_OBSERVER) -> None:
        self._weights = _FakeWeights(observer)

    def foreground(self, image: Image.Image) -> Image.Image:
        self._weights.load()
        width, height = image.size
        yy, xx = np.mgrid[0:height, 0:width]
        inside = ((xx + 0.5) / width - 0.5) ** 2 + ((yy + 0.5) / height - 0.5) ** 2 <= 0.25
        return Image.fromarray(inside.astype(np.uint8) * 255)

    def release(self) -> None:
        self._weights.release()


def placeholder_processor(observer: LoadObserver = NO_OBSERVER) -> WheelSwapProcessor:
    return WheelSwapProcessor(
        detector=CornerDetector(observer),
        segmenter=CircleSegmenter(observer=observer),
        editor=GrayscaleEditor(observer),
        remover=EllipseRemover(observer),
    )
