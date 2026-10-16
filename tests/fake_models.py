"""Atrapy modeli AI (granica systemu) dla testów: SAM 2 i mflux bez GPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from wheeler_worker.rim_editor import StepCallback


def disc_mask(size: tuple[int, int], center: tuple[float, float], radius: int) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = round(center[0]), round(center[1])
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
    return Image.fromarray(inside.astype(np.uint8) * 255)


@dataclass
class DiscSegmenter:
    """„SAM 2”: koło to dysk o promieniu `radius` wokół kliknięcia."""

    radius: int = 50
    clicks: list[tuple[float, float]] = field(default_factory=list)

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        self.clicks.append(point)
        return disc_mask(image.size, point, self.radius)

    def release(self) -> None:
        pass


class NothingSegmenter:
    """„SAM 2”, który w klikniętym miejscu nic nie znalazł (pusta maska)."""

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        return Image.new("L", image.size, 0)

    def release(self) -> None:
        pass


@dataclass(frozen=True)
class EditCall:
    images: list[Image.Image]
    mask: Image.Image
    prompt: str
    seed: int
    size: tuple[int, int]


@dataclass
class SolidEditor:
    """„mflux”: zwraca jednolity kolor w żądanym rozmiarze i zapamiętuje wywołania.

    Udaje `steps` kroków denoisingu: po każdym woła `on_step`, a potem `after_step(krok)`
    (np. test zagląda wtedy do UI albo klika „Anuluj”, jak użytkownik w trakcie kroku).
    `steps_run` liczy wykonane (rozpoczęte) kroki.
    """

    color: tuple[int, int, int] = (10, 200, 30)
    calls: list[EditCall] = field(default_factory=list)
    steps: int = 1
    after_step: Callable[[int], None] | None = None
    steps_run: int = 0

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
        self.calls.append(EditCall(list(images), mask, prompt, seed, size))
        for step in range(1, self.steps + 1):
            self.steps_run += 1
            if on_step is not None:
                on_step(step, self.steps)
            if self.after_step is not None:
                self.after_step(step)
        return Image.new("RGBA", size, (*self.color, 255))

    def release(self) -> None:
        pass
