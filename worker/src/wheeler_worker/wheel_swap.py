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

from wheeler_shared import storage
from wheeler_shared.edit_params import EditParams
from wheeler_shared.failures import Failure, ProcessingFailed
from wheeler_shared.progress import Progress
from wheeler_worker.background import BackgroundRemover
from wheeler_worker.compositing import composite, crop_box_for, grow_mask, mask_box, working_size
from wheeler_worker.detector import WheelDetector
from wheeler_worker.photos import load_original, load_photo, read_metadata, save_png
from wheeler_worker.processing import (
    CutRimItem,
    CutRimResult,
    DetectItem,
    DetectResult,
    GenerateItem,
    GenerateResult,
    MaskFiles,
    RegenerateWheelItem,
    SegmentItem,
    SegmentResult,
)
from wheeler_worker.rim_cut import cut_out, load_reference, pick_rim_box, rim_region
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
# Rozmiar felgi ≠ „bez zmian” (ticket 12): bez „Keep the tire” i „fill the tire opening
# exactly”, bo felga ma urosnąć albo zmaleć kosztem boku opony przy tej samej średnicy koła.
# Cale są przybliżone (obecnego rozmiaru felgi nie znamy): 1″ ≈ 4% średnicy całego koła.
PLUS_SIZING_TEMPLATE = (
    "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. "
    "Keep the car body, the wheel arch, the brake caliper position, the lighting, "
    "the shadows and the camera angle of image 1 unchanged. Make the new rim about "
    "{inches} {direction} in diameter than the current rim (about {percent}% of the overall "
    "wheel diameter), with a correspondingly {sidewall} tire sidewall. Keep the overall outer "
    "diameter of the wheel (tire included) exactly as in image 1. The new rim must follow the "
    "perspective of the wheel in image 1. Photorealistic."
)
RIM_SIZE_PERCENT_PER_INCH = 4
CROP_MARGIN = 0.25  # margines wycinka: ułamek boku koła z każdej strony
FEATHER = 0.015  # pas wygładzania krawędzi (do wewnątrz maski): ułamek boku koła
OVERLAY_RGBA = (37, 99, 235, 110)  # półprzezroczysty niebieski


def wheel_dir(output_dir: Path, wheel: int | str) -> Path:
    """Katalog plików koła w katalogu zadania (układ: `wheeler_shared.storage.wheel_dir`)."""
    target = output_dir / storage.wheel_dir(wheel)
    target.mkdir(parents=True, exist_ok=True)
    return target


class WheelSwapProcessor:
    def __init__(
        self,
        *,
        detector: WheelDetector,
        segmenter: Segmenter,
        editor: RimEditor,
        remover: BackgroundRemover | None = None,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.editor = editor
        # Usuwanie tła felgi (biblioteka felg); bez niego `cut_rim` działa tylko z kliknięciem.
        self.remover = remover

    def detect(self, item: DetectItem) -> DetectResult:
        photo = load_photo(item.photo_path)
        params = EditParams.from_params(item.params)
        boxes = self.detector.detect(photo)
        raw_masks = self.segmenter.segment_boxes(photo, boxes) if boxes else []
        wheels: list[MaskFiles] = []
        for raw in raw_masks:
            target = item.output_dir / storage.wheel_dir(f"detected-{len(wheels) + 1}")
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
                raise ProcessingFailed(Failure.NOTHING_AT_CLICK, f"pusta maska w {point}")
            masks[wheel_id] = files
        return SegmentResult(masks=masks)

    def cut_rim(self, item: CutRimItem) -> CutRimResult:
        """Felga bez tła, przycięta do felgi.

        Detektor kół wskazuje koło (największe albo to pod kliknięciem), a BiRefNet usuwa tło
        z kadru wokół niego. Na zdjęciu felgi zamontowanej w aucie BiRefNet uruchomiony na
        całym zdjęciu zostawia karoserię (dominujący obiekt), więc kadr jest kluczowy. Bez
        wykrytego koła: BiRefNet na całym zdjęciu albo SAM 2 z kliknięcia.
        """
        photo = load_photo(item.original_path)
        foreground = self._rim_foreground(photo, item.click)
        cut = cut_out(photo, foreground)
        if cut is None:
            raise ProcessingFailed(Failure.RIM_NOT_FOUND, "pusta maska pierwszego planu")
        return CutRimResult(cut_path=save_png(cut, item.output_dir / "cut.png"))

    def _rim_foreground(self, photo: Image.Image, click: tuple[float, float] | None) -> Image.Image:
        box = pick_rim_box(self.detector.detect(photo), click)
        if box is None and click is not None:
            return self.segmenter.segment_click(photo, click)
        if self.remover is None:
            raise RuntimeError("Procesor nie ma modelu do usuwania tła")
        if box is None:
            return self.remover.foreground(photo)
        region = rim_region(box, photo.size)
        foreground = Image.new("L", photo.size, 0)
        foreground.paste(self.remover.foreground(photo.crop(region)), region[:2])
        return foreground

    def generate(self, item: GenerateItem) -> GenerateResult:
        params = EditParams.from_params(item.params)
        if params.seed is None:
            raise ValueError("Brak seeda w parametrach zadania")
        edit = self._prepare_edit(params, item.rim_path, seed=params.seed)
        result = load_original(item.photo_path)
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

    def regenerate_wheel(self, item: RegenerateWheelItem) -> GenerateResult:
        """Jedno koło z nowym seedem na bieżącym wyniku; zapis do nowego pliku `result-<n>.png`
        (poprzedni wynik zostaje, a nowa nazwa omija pamięć podręczną przeglądarki)."""
        params = EditParams.from_params(item.params)
        edit = self._prepare_edit(params, item.rim_path, seed=item.seed)
        item.on_progress(Progress(wheel=1, wheels=1))
        with Image.open(item.mask_path) as mask_file:
            mask = mask_file.convert("L")

        def on_step(step: int, steps: int) -> None:
            item.on_progress(Progress(wheel=1, wheels=1, step=step, steps=steps))

        result = edit.apply(
            load_original(item.result_path),
            mask,
            wheel_dir(item.output_dir, item.wheel_id),
            on_step,
        )
        target = save_png(
            result, _next_result_path(item.output_dir), metadata=read_metadata(item.photo_path)
        )
        return GenerateResult(result_path=target, width=result.width, height=result.height)

    def _prepare_edit(self, params: EditParams, rim_path: Path, *, seed: int) -> _WheelEdit:
        # Florence-2 i SAM 2 (~3 GB każdy) z mflux (szczyt ~27 GB) nie mieszczą się w 32 GB.
        self.detector.release()
        self.segmenter.release()
        if self.remover is not None:
            self.remover.release()
        return _WheelEdit(
            editor=self.editor,
            # Felga z biblioteki ma przezroczyste tło: model widzi ją na jednolitym tle.
            rim=load_reference(rim_path),
            prompt=build_prompt(params.prompt_suffix, rim_size=params.rim_size),
            seed=seed,
            resolution=params.resolution,
        )

    def release(self) -> None:
        self.detector.release()
        self.segmenter.release()
        self.editor.release()
        if self.remover is not None:
            self.remover.release()


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
            raise ProcessingFailed(Failure.EMPTY_MASK)
        side = max(wheel.width, wheel.height)
        crop = crop_box_for(wheel, photo.size, margin=CROP_MARGIN)
        size = working_size(crop, resolution=self.resolution)
        # Model dostaje RGB; `photo` jest w trybie wyniku (`photos.load_original`).
        crop_image = photo.crop(crop).convert("RGB").resize(size, Image.Resampling.LANCZOS)
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


def _next_result_path(output_dir: Path) -> Path:
    number = 2
    while (output_dir / f"result-{number}.png").exists():
        number += 1
    return output_dir / f"result-{number}.png"


def build_prompt(suffix: str = "", *, rim_size: int = 0) -> str:
    """Prompt edycji: szablon (przy `rim_size` ≠ 0 plus-sizing o tyle cali) i dopisek."""
    template = PROMPT_TEMPLATE
    if rim_size:
        inches = abs(rim_size)
        template = PLUS_SIZING_TEMPLATE.format(
            inches=f"{inches} inch" if inches == 1 else f"{inches} inches",
            direction="larger" if rim_size > 0 else "smaller",
            percent=inches * RIM_SIZE_PERCENT_PER_INCH,
            sidewall="thinner" if rim_size > 0 else "thicker",
        )
    suffix = suffix.strip()
    return f"{template} {suffix}" if suffix else template


def _overlay(mask: Image.Image) -> Image.Image:
    inside = np.asarray(mask.convert("L")) > 0
    rgba = np.zeros((*inside.shape, 4), dtype=np.uint8)
    rgba[inside] = OVERLAY_RGBA
    return Image.fromarray(rgba)
