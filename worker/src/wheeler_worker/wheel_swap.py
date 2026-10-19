"""Podmiana felg: maski kół (Florence-2 + SAM 2) i edycja każdego koła osobno (mflux).

Pipeline (docs/DESIGN.md, „Pipeline edycji”):
1. `detect` (po wgraniu): boxy kół z detektora -> maski SAM z promptem box. `segment` (klik
   w miejsce bez maski): klik -> maska SAM całego koła. Obie maski są poszerzane o `mask_grow`
   razy bok koła; zapisana maska to dokładnie obszar, który generowanie może zmienić.
2. `generate`: dla każdego koła kwadratowy wycinek oryginału z marginesem 25%, przeskalowany
   do rozdzielczości roboczej; model dostaje `[wycinek, felga]` i maskę wycinka; wynik wraca
   do rozmiaru wycinka i jest wklejany w zdjęcie (przy „ponów to koło”: w bieżący wynik)
   z krawędzią wygładzaną do wewnątrz maski.
   Rozmiar felgi ≠ „bez zmian” (ticket 12): przed mflux SAM 2 wyznacza elipsę felgi każdego
   koła (`rim_geometry.measure_rim`), a wycinek dostaje radialny warp felgi do docelowego
   rozmiaru (`rim_geometry.warp_rim`); prompt jest ten sam.

Pliki w katalogu zadania: `wheels/<id>/mask.png`, `overlay.png` (segment; dla kół z detekcji
`wheels/detected-<n>/`, bo id koła powstaje dopiero przy zapisie) oraz `crop.png`,
`crop_mask.png`, `generated.png` (generate; przy zmianie rozmiaru felgi także diagnostyczne
`rim_ellipse.png` i `warped.png`, czyli wycinek, który dostał model), wynik `result.png`.
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
from wheeler_worker.compositing import (
    composite,
    crop_box_for,
    grow_wheel_mask,
    mask_box,
    ungrown_wheel_mask,
    working_size,
)
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
from wheeler_worker.rim_geometry import (
    RimGeometry,
    measure_rim,
    outline_rim,
    warp_rim,
)
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
        masks = {wheel_id: _load_mask(path) for wheel_id, path in item.masks.items()}
        geometries = self._measure_rims(
            item.photo_path, masks, rim_size=params.rim_size, mask_grow=params.mask_grow
        )
        edit = self._prepare_edit(params, item.rim_path, seed=params.seed)
        original = load_original(item.photo_path)
        result = original
        wheels = len(masks)
        # Koła po kolei, każde osobnym wywołaniem modelu, z tym samym seedem i promptem.
        # Raport przed kołem i po każdym kroku: tu odbiorca może przerwać pracę (anulowanie).
        for index, (wheel_id, mask) in enumerate(masks.items(), start=1):
            item.on_progress(Progress(wheel=index, wheels=wheels))

            def on_step(step: int, steps: int, wheel: int = index) -> None:
                item.on_progress(Progress(wheel=wheel, wheels=wheels, step=step, steps=steps))

            result = edit.apply(
                original,
                result,
                mask,
                wheel_dir(item.output_dir, wheel_id),
                rim_geometry=geometries.get(wheel_id),
                on_step=on_step,
            )
        target = save_png(
            result, item.output_dir / "result.png", metadata=read_metadata(item.photo_path)
        )
        return GenerateResult(result_path=target, width=result.width, height=result.height)

    def regenerate_wheel(self, item: RegenerateWheelItem) -> GenerateResult:
        """Jedno koło z nowym seedem, złożone na bieżącym wyniku; zapis do nowego pliku
        `result-<n>.png` (poprzedni wynik zostaje, a nowa nazwa omija pamięć podręczną
        przeglądarki). Wycinek dla modelu i pomiar felgi z oryginału, jak przy generowaniu
        (felga po zmianie rozmiaru nie jest przeskalowywana drugi raz)."""
        params = EditParams.from_params(item.params)
        mask = _load_mask(item.mask_path)
        geometries = self._measure_rims(
            item.photo_path,
            {item.wheel_id: mask},
            rim_size=params.rim_size,
            mask_grow=params.mask_grow,
        )
        edit = self._prepare_edit(params, item.rim_path, seed=item.seed)
        item.on_progress(Progress(wheel=1, wheels=1))

        def on_step(step: int, steps: int) -> None:
            item.on_progress(Progress(wheel=1, wheels=1, step=step, steps=steps))

        result = edit.apply(
            load_original(item.photo_path),
            load_original(item.result_path),
            mask,
            wheel_dir(item.output_dir, item.wheel_id),
            rim_geometry=geometries.get(item.wheel_id),
            on_step=on_step,
        )
        target = save_png(
            result, _next_result_path(item.output_dir), metadata=read_metadata(item.photo_path)
        )
        return GenerateResult(result_path=target, width=result.width, height=result.height)

    def _measure_rims(
        self,
        photo_path: Path,
        masks: dict[int, Image.Image],
        *,
        rim_size: int,
        mask_grow: float,
    ) -> dict[int, RimGeometry]:
        """Geometria felgi każdego koła (SAM 2 na oryginale; `masks` poszerzone o `mask_grow`),
        gdy rozmiar felgi się zmienia; przy „bez zmian” nic nie liczymy. Felgi, której nie da
        się wyznaczyć: błąd zadania."""
        if rim_size == 0:
            return {}
        photo = load_photo(photo_path)
        geometries: dict[int, RimGeometry] = {}
        for wheel_id, mask in masks.items():
            wheel = ungrown_wheel_mask(mask, mask_grow)
            geometry = measure_rim(wheel, self.segmenter.segment_rim(photo, wheel))
            if geometry is None:
                raise ProcessingFailed(
                    Failure.RIM_SIZE_UNAVAILABLE, f"koło {wheel_id}: brak elipsy felgi"
                )
            geometries[wheel_id] = geometry
        return geometries

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
            prompt=build_prompt(params.prompt_suffix),
            seed=seed,
            resolution=params.resolution,
            rim_size=params.rim_size,
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
    rim_size: int

    def apply(
        self,
        original: Image.Image,
        base: Image.Image,
        mask: Image.Image,
        out: Path,
        *,
        rim_geometry: RimGeometry | None = None,
        on_step: StepCallback | None = None,
    ) -> Image.Image:
        """Podmienia felgę w jednym kole: model dostaje wycinek `original` (z `rim_geometry`:
        felga przeskalowana o `rim_size`, tylko w `mask`), wynik jest składany na `base`;
        piksele poza `mask` zostają z `base` bit w bit."""
        wheel = mask_box(mask)
        if wheel is None:
            raise ProcessingFailed(Failure.EMPTY_MASK)
        side = max(wheel.width, wheel.height)
        crop = crop_box_for(wheel, original.size, margin=CROP_MARGIN)
        size = working_size(crop, resolution=self.resolution)
        # Model dostaje RGB; `original` jest w trybie wyniku (`photos.load_original`).
        source = original.crop(crop).convert("RGB")
        crop_image = source.resize(size, Image.Resampling.LANCZOS)
        crop_mask = mask.crop(crop).resize(size, Image.Resampling.BILINEAR)
        save_png(crop_image, out / "crop.png")
        save_png(crop_mask, out / "crop_mask.png")
        if rim_geometry is not None:
            local = rim_geometry.shifted(-crop.left, -crop.top)
            save_png(outline_rim(source, local), out / "rim_ellipse.png")
            warped = warp_rim(source, local, rim_size=self.rim_size)
            # Tylko w masce, którą model zmienia: bok opony (felga * k) potrafi z niej wyjść,
            # a poza maską kompozycja i tak zostawia oryginał (model widziałby szew).
            crop_image = Image.composite(
                warped.resize(size, Image.Resampling.LANCZOS), crop_image, crop_mask
            )
            save_png(crop_image, out / "warped.png")
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
        return composite(base, generated, crop, mask, feather=feather)


def _save_mask(raw: Image.Image, params: EditParams, target: Path) -> MaskFiles | None:
    """Poszerza surową maskę SAM o `mask_grow` boku koła i zapisuje ją z nakładką.

    Zwraca None dla pustej maski (nic nie zapisuje)."""
    mask = grow_wheel_mask(raw, params.mask_grow)
    grown = mask_box(mask)
    if grown is None:
        return None
    target.mkdir(parents=True, exist_ok=True)
    return MaskFiles(
        mask=save_png(mask, target / "mask.png"),
        overlay=save_png(_overlay(mask), target / "overlay.png"),
        box=grown,
    )


def _load_mask(path: Path) -> Image.Image:
    with Image.open(path) as mask_file:
        return mask_file.convert("L")


def _next_result_path(output_dir: Path) -> Path:
    number = 2
    while (output_dir / f"result-{number}.png").exists():
        number += 1
    return output_dir / f"result-{number}.png"


def build_prompt(suffix: str = "") -> str:
    """Prompt edycji: szablon i dopisek. Ten sam dla każdego rozmiaru felgi: polecenie
    rozmiaru model ignoruje, więc rozmiarem steruje warp wycinka (ticket 12)."""
    suffix = suffix.strip()
    return f"{PROMPT_TEMPLATE} {suffix}" if suffix else PROMPT_TEMPLATE


def _overlay(mask: Image.Image) -> Image.Image:
    inside = np.asarray(mask.convert("L")) > 0
    rgba = np.zeros((*inside.shape, 4), dtype=np.uint8)
    rgba[inside] = OVERLAY_RGBA
    return Image.fromarray(rgba)
