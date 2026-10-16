"""Atrapy modeli AI (granica systemu) dla testów: SAM 2 i mflux bez GPU."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


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
    """„mflux”: zwraca jednolity kolor w żądanym rozmiarze i zapamiętuje wywołania."""

    color: tuple[int, int, int] = (10, 200, 30)
    calls: list[EditCall] = field(default_factory=list)

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
        self.calls.append(EditCall(list(images), mask, prompt, seed, size))
        return Image.new("RGBA", size, (*self.color, 255))

    def release(self) -> None:
        pass


@dataclass
class CenterDiscRemover:
    """„BiRefNet”: pierwszy plan to dysk o promieniu `radius` wokół środka zdjęcia."""

    radius: int = 30
    images: list[Image.Image] = field(default_factory=list)

    def foreground(self, image: Image.Image) -> Image.Image:
        self.images.append(image)
        return disc_mask(image.size, (image.width // 2, image.height // 2), self.radius)

    def release(self) -> None:
        pass


class OpaqueRemover:
    """„BiRefNet”, który całe zdjęcie uznaje za pierwszy plan (felga bez zmian)."""

    def foreground(self, image: Image.Image) -> Image.Image:
        return Image.new("L", image.size, 255)

    def release(self) -> None:
        pass


class NothingRemover:
    """„BiRefNet”, który nie znalazł na zdjęciu żadnego obiektu (pusta maska)."""

    def foreground(self, image: Image.Image) -> Image.Image:
        return Image.new("L", image.size, 0)

    def release(self) -> None:
        pass
