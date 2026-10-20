"""Atrapy modeli AI (granica systemu) dla testów: SAM 2 i mflux bez GPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from wheeler_worker.body_mask import BodySegments
from wheeler_worker.compositing import mask_box
from wheeler_worker.rim_editor import StepCallback
from wheeler_worker.rim_geometry import RimCandidate, RimGeometry, wheel_ellipse

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 w pikselach zdjęcia
# „SAM 2” na karoserię: (zdjęcie, maska koła bez poszerzenia) -> maska karoserii.
BodyFake = Callable[[Image.Image, Image.Image], Image.Image]


def disc_mask(size: tuple[int, int], center: tuple[float, float], radius: int) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = round(center[0]), round(center[1])
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
    return Image.fromarray(inside.astype(np.uint8) * 255)


def fender_mask(wheel_mask: Image.Image, gap: int = 4) -> Image.Image:
    """Błotnik: wszystko powyżej koła, z `gap` px szczeliny nadkola nad górą maski koła."""
    body = Image.new("L", wheel_mask.size, 0)
    box = mask_box(wheel_mask)
    if box is not None and box.top - gap > 0:
        body.paste(255, (0, 0, wheel_mask.width, box.top - gap))
    return body


def _fender(photo: Image.Image, wheel_mask: Image.Image) -> Image.Image:
    return fender_mask(wheel_mask)


@dataclass
class DiscSegmenter:
    """„SAM 2”: koło to dysk o promieniu `radius` wokół kliknięcia, a felga dysk o promieniu
    `rim_radius` w środku maski koła (`None`: felgi nie widać, brak kandydatów).

    Karoseria: jeden kandydat z `body` (domyślnie błotnik nad kołem, `fender_mask`), bez
    wnętrza nadkola. `rim_photos` i `body_photos` zapamiętują zdjęcia, na których szukano
    felgi i karoserii, a `log` kolejność pomiarów i zwolnień modelu („rim”, „body”,
    „release”)."""

    radius: int = 50
    rim_radius: int | None = None
    body: BodyFake = _fender
    clicks: list[tuple[float, float]] = field(default_factory=list)
    rim_photos: list[Image.Image] = field(default_factory=list)
    body_photos: list[Image.Image] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        self.clicks.append(point)
        return disc_mask(image.size, point, self.radius)

    def segment_boxes(self, image: Image.Image, boxes: list[Box]) -> list[Image.Image]:
        """Koło wpisane w box (dysk o średnicy krótszego boku)."""
        return [
            disc_mask(
                image.size,
                ((x0 + x1) / 2, (y0 + y1) / 2),
                round(min(x1 - x0, y1 - y0) / 2),
            )
            for x0, y0, x1, y1 in boxes
        ]

    def segment_rim(self, image: Image.Image, wheel_mask: Image.Image) -> list[RimCandidate]:
        self.rim_photos.append(image.copy())
        self.log.append("rim")
        wheel = wheel_ellipse(wheel_mask)
        if self.rim_radius is None or wheel is None:
            return []
        return [(disc_mask(image.size, (wheel.cx, wheel.cy), self.rim_radius), 0.9)]

    def segment_body(
        self, image: Image.Image, wheel_mask: Image.Image, rim: RimGeometry
    ) -> BodySegments:
        self.body_photos.append(image.copy())
        self.log.append("body")
        return BodySegments(
            with_gap=[(self.body(image, wheel_mask), 0.9)], without_gap=[], liner=[]
        )

    def release(self) -> None:
        self.log.append("release")


@dataclass
class BoxDetector:
    """„Florence-2”: zwraca zadane boxy kół niezależnie od zdjęcia."""

    boxes: list[Box] = field(default_factory=list)

    def detect(self, image: Image.Image) -> list[Box]:
        return list(self.boxes)

    def release(self) -> None:
        pass


class NothingSegmenter:
    """„SAM 2”, który w klikniętym miejscu nic nie znalazł (pusta maska)."""

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        return Image.new("L", image.size, 0)

    def segment_boxes(self, image: Image.Image, boxes: list[Box]) -> list[Image.Image]:
        return [Image.new("L", image.size, 0) for _ in boxes]

    def segment_rim(self, image: Image.Image, wheel_mask: Image.Image) -> list[RimCandidate]:
        return []

    def segment_body(
        self, image: Image.Image, wheel_mask: Image.Image, rim: RimGeometry
    ) -> BodySegments:
        return BodySegments(with_gap=[], without_gap=[], liner=[])

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
    """„mflux”: zwraca jednolity kolor w żądanym rozmiarze i zapamiętuje wywołania
    (`echo`: zamiast koloru zwraca otrzymany wycinek, jakby model nic nie zmienił).

    Udaje `steps` kroków denoisingu: po każdym woła `on_step`, a potem `after_step(krok)`
    (np. test zagląda wtedy do UI albo klika „Anuluj”, jak użytkownik w trakcie kroku).
    `steps_run` liczy wykonane (rozpoczęte) kroki.
    """

    color: tuple[int, int, int] = (10, 200, 30)
    calls: list[EditCall] = field(default_factory=list)
    steps: int = 1
    after_step: Callable[[int], None] | None = None
    steps_run: int = 0
    echo: bool = False

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
        if self.echo:
            return images[0].resize(size)
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
