"""Podmiana felg: maski kół (Florence-2 + SAM 2) i edycja każdego koła osobno (mflux).

Pipeline (docs/DESIGN.md, „Pipeline edycji”):
1. `detect` (po wgraniu): boxy kół z detektora -> maski SAM z promptem box. `segment` (klik
   w miejsce bez maski): klik -> maska SAM całego koła. Obie maski są poszerzane o `mask_grow`
   razy bok koła; zapisana maska to dokładnie obszar, który generowanie może zmienić.
2. `generate`: dla każdego koła kwadratowy wycinek z marginesem 25%, przeskalowany do
   rozdzielczości roboczej; model dostaje `[wycinek, felga]` i maskę wycinka; wynik wraca do
   rozmiaru wycinka i jest wklejany w zdjęcie z krawędzią wygładzaną do wewnątrz maski.

Pliki w katalogu zadania: `wheels/<id>/mask.png`, `overlay.png` (segment; dla kół z detekcji
`wheels/detected-<n>/`, bo id koła powstaje dopiero przy zapisie) oraz `crop.png`,
`crop_mask.png`, `generated.png` (generate), wynik `result.png`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from wheeler_shared.edit_params import EditParams
from wheeler_shared.progress import Progress
from wheeler_worker.compositing import composite, crop_box_for, grow_mask, mask_box, working_size
from wheeler_worker.detector import WheelDetector
from wheeler_worker.photos import load_photo, read_metadata, save_png
from wheeler_worker.processing import (
    DetectItem,
    DetectResult,
    GenerateItem,
    GenerateResult,
    MaskFiles,
    SegmentItem,
    SegmentResult,
)
from wheeler_worker.rim_editor import RimEditor, StepCallback
from wheeler_worker.segmenter import Segmenter

# Obraz 1 = wycinek koła (do niego odnosi się maska mflux), obraz 2 = felga. Kolejność
# `[felga, wycinek]` w 2 z 3 przypadków spike'a zwracała samo zdjęcie felgi.
PROMPT_TEMPLATE = (
    "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. "
    "Keep the tire, the car body, the wheel arch, the brake caliper position, the lighting, "
    "the shadows and the camera angle of image 1 unchanged. The new rim must follow the "
    "perspective of the wheel in image 1 and fill the tire opening exactly. Photorealistic."
)
CROP_MARGIN = 0.25  # margines wycinka: ułamek boku koła z każdej strony
FEATHER = 0.015  # pas wygładzania krawędzi (do wewnątrz maski): ułamek boku koła
OVERLAY_RGBA = (37, 99, 235, 110)  # półprzezroczysty niebieski


def wheel_dir(output_dir: Path, wheel_id: int) -> Path:
    target = output_dir / "wheels" / str(wheel_id)
    target.mkdir(parents=True, exist_ok=True)
    return target


class WheelSwapProcessor:
    def __init__(self, *, detector: WheelDetector, segmenter: Segmenter, editor: RimEditor) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.editor = editor

    def detect(self, item: DetectItem) -> DetectResult:
        photo = load_photo(item.photo_path)
        params = EditParams.from_params(item.params)
        boxes = self.detector.detect(photo)
        raw_masks = self.segmenter.segment_boxes(photo, boxes) if boxes else []
        wheels: list[MaskFiles] = []
        for raw in raw_masks:
            target = item.output_dir / "wheels" / f"detected-{len(wheels) + 1}"
            files = _save_mask(raw, params, target)
            if files is not None:  # SAM nic nie znalazł w boxie: pomijamy
                wheels.append(files)
        return DetectResult(wheels=wheels)

    def segment(self, item: SegmentItem) -> SegmentResult:
        photo = load_photo(item.photo_path)
        params = EditParams.from_params(item.params)
        masks: dict[int, MaskFiles] = {}
        for wheel_id, point in item.clicks.items():
            raw = self.segmenter.segment_click(photo, point)
            files = _save_mask(raw, params, wheel_dir(item.output_dir, wheel_id))
            if files is None:
                raise ValueError("Nie udało się wyznaczyć maski koła w klikniętym miejscu")
            masks[wheel_id] = files
        return SegmentResult(masks=masks)

    def generate(self, item: GenerateItem) -> GenerateResult:
        params = EditParams.from_params(item.params)
        if params.seed is None:
            raise ValueError("Brak seeda w parametrach zadania")
        # Florence-2 i SAM 2 (~3 GB każdy) z mflux (szczyt ~27 GB) nie mieszczą się w 32 GB.
        self.detector.release()
        self.segmenter.release()
        photo = load_photo(item.photo_path)
        rim = load_photo(item.rim_path)
        edit = _WheelEdit(
            editor=self.editor,
            rim=rim,
            prompt=build_prompt(params.prompt_suffix),
            seed=params.seed,
            resolution=params.resolution,
        )
        result = photo
        wheels = len(item.masks)
        # Koła po kolei, każde osobnym wywołaniem modelu, z tym samym seedem i promptem.
        # Raport przed kołem i po każdym kroku: tu odbiorca może przerwać pracę (anulowanie).
        for index, (wheel_id, mask_path) in enumerate(item.masks.items(), start=1):
            item.on_progress(Progress(wheel=index, wheels=wheels))
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L")

            def on_step(step: int, steps: int, wheel: int = index) -> None:
                item.on_progress(Progress(wheel=wheel, wheels=wheels, step=step, steps=steps))

            result = edit.apply(result, mask, wheel_dir(item.output_dir, wheel_id), on_step)
        target = save_png(
            result, item.output_dir / "result.png", metadata=read_metadata(item.photo_path)
        )
        return GenerateResult(result_path=target, width=result.width, height=result.height)

    def release(self) -> None:
        self.detector.release()
        self.segmenter.release()
        self.editor.release()


@dataclass(frozen=True)
class _WheelEdit:
    editor: RimEditor
    rim: Image.Image
    prompt: str
    seed: int
    resolution: int

    def apply(
        self,
        photo: Image.Image,
        mask: Image.Image,
        out: Path,
        on_step: StepCallback | None = None,
    ) -> Image.Image:
        """Podmienia felgę w jednym kole; piksele poza `mask` zostają bit w bit."""
        wheel = mask_box(mask)
        if wheel is None:
            raise ValueError("Pusta maska koła")
        side = max(wheel.width, wheel.height)
        crop = crop_box_for(wheel, photo.size, margin=CROP_MARGIN)
        size = working_size(crop, resolution=self.resolution)
        crop_image = photo.crop(crop).resize(size, Image.Resampling.LANCZOS)
        crop_mask = mask.crop(crop).resize(size, Image.Resampling.BILINEAR)
        save_png(crop_image, out / "crop.png")
        save_png(crop_mask, out / "crop_mask.png")
        generated = self.editor.edit(
            [crop_image, self.rim],
            mask=crop_mask,
            prompt=self.prompt,
            seed=self.seed,
            size=size,
            resolution=self.resolution,
            on_step=on_step,
        ).convert("RGB")
        save_png(generated, out / "generated.png")
        feather = max(2, round(side * FEATHER))
        return composite(photo, generated, crop, mask, feather=feather)


def _save_mask(raw: Image.Image, params: EditParams, target: Path) -> MaskFiles | None:
    """Poszerza surową maskę SAM o `mask_grow` boku koła i zapisuje ją z nakładką.

    Zwraca None dla pustej maski (nic nie zapisuje)."""
    box = mask_box(raw)
    if box is None:
        return None
    side = max(box.width, box.height)
    mask = grow_mask(raw, pixels=round(side * params.mask_grow))
    target.mkdir(parents=True, exist_ok=True)
    grown = mask_box(mask)
    assert grown is not None
    return MaskFiles(
        mask=save_png(mask, target / "mask.png"),
        overlay=save_png(_overlay(mask), target / "overlay.png"),
        box=grown,
    )


def build_prompt(suffix: str = "") -> str:
    suffix = suffix.strip()
    return f"{PROMPT_TEMPLATE} {suffix}" if suffix else PROMPT_TEMPLATE


def _overlay(mask: Image.Image) -> Image.Image:
    inside = np.asarray(mask.convert("L")) > 0
    rgba = np.zeros((*inside.shape, 4), dtype=np.uint8)
    rgba[inside] = OVERLAY_RGBA
    return Image.fromarray(rgba)
