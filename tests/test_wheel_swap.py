"""Procesor podmiany felgi (`WheelSwapProcessor`) z atrapami SAM 2 i mflux."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor, disc_mask
from helpers import noise_file
from photo_files import display_profile
from PIL import Image

from wheeler_shared.progress import Progress
from wheeler_worker.processing import (
    DetectItem,
    GenerateItem,
    JobCancelled,
    RegenerateWheelItem,
    SegmentItem,
    build_processor,
)
from wheeler_worker.settings import WorkerSettings
from wheeler_worker.wheel_swap import WheelSwapProcessor


def test_click_gives_slightly_widened_wheel_mask_and_overlay(tmp_path: Path) -> None:
    photo = noise_file(tmp_path / "input.png")
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=50), editor=SolidEditor()
    )

    result = processor.segment(
        SegmentItem(
            job_id=1, photo_path=photo, output_dir=tmp_path, params={}, clicks={7: (150, 120)}
        )
    )

    files = result.masks[7]
    assert files.mask.is_relative_to(tmp_path) and files.overlay.is_relative_to(tmp_path)
    mask = np.asarray(Image.open(files.mask).convert("L")) > 0
    assert mask.shape == (240, 400)
    # Koło o boku 101 px poszerzone o 3% boku = 3 px.
    assert mask[120, 150 + 53] and mask[120 - 53, 150]
    assert not mask[120, 150 + 56]
    overlay = Image.open(files.overlay)
    assert overlay.mode == "RGBA" and overlay.size == (400, 240)
    alpha = np.asarray(overlay)[..., 3]
    assert (alpha[mask] > 0).all() and (alpha[~mask] == 0).all()


def _rim(path: Path) -> Path:
    Image.new("RGB", (300, 200), (220, 20, 20)).save(path)
    return path


def _segment_one(processor: WheelSwapProcessor, photo: Path, out: Path) -> Path:
    item = SegmentItem(
        job_id=1, photo_path=photo, output_dir=out, params={}, clicks={7: (150, 120)}
    )
    return processor.segment(item).masks[7].mask


def test_generated_png_has_input_resolution_and_changes_only_pixels_inside_mask(
    tmp_path: Path,
) -> None:
    photo = noise_file(tmp_path / "input.jpg", width=401, height=239)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=50), editor=SolidEditor()
    )
    mask_path = _segment_one(processor, photo, tmp_path)

    result = processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=_rim(tmp_path / "rim.png"),
            output_dir=tmp_path,
            params={"seed": 7},
            masks={7: mask_path},
        )
    )

    assert result.result_path.is_relative_to(tmp_path)
    with Image.open(result.result_path) as output:
        assert output.format == "PNG"
        assert output.size == (401, 239) == (result.width, result.height)
        pixels = np.asarray(output.convert("RGB"))
    with Image.open(photo) as original_file:
        original = np.asarray(original_file.convert("RGB"))
    outside = np.asarray(Image.open(mask_path).convert("L")) == 0
    assert np.array_equal(pixels[outside], original[outside])
    core = np.asarray(disc_mask((401, 239), (150, 120), 40)) > 0
    assert (pixels[core] == (10, 200, 30)).all()


def test_model_gets_wheel_crop_then_rim_with_mask_seed_and_default_prompt(tmp_path: Path) -> None:
    photo = noise_file(tmp_path / "input.png", width=400, height=240)
    editor = SolidEditor()
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=50), editor=editor
    )
    mask_path = _segment_one(processor, photo, tmp_path)

    processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=_rim(tmp_path / "rim.png"),
            output_dir=tmp_path,
            params={"seed": 1234, "prompt_suffix": "Chrome finish."},
            masks={7: mask_path},
        )
    )

    [call] = editor.calls
    crop, rim = call.images
    # Koło 107 px (po poszerzeniu) + 25% marginesu = kwadrat ~160 px, skalowany do 1024².
    assert call.size == (1024, 1024) == crop.size == call.mask.size
    assert rim.size == (300, 200) and rim.convert("RGB").getpixel((5, 5)) == (220, 20, 20)
    assert call.seed == 1234
    assert call.prompt.startswith(
        "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2."
    )
    assert call.prompt.endswith("Photorealistic. Chrome finish.")
    mask = np.asarray(call.mask.convert("L"))
    assert mask[512, 512] == 255 and mask[5, 5] == 0  # koło w środku wycinka, tło poza maską


def test_placeholder_processor_runs_whole_pipeline_without_models(tmp_path: Path) -> None:
    photo = noise_file(tmp_path / "input.png")
    processor = build_processor(WorkerSettings(processor="placeholder"))
    item = SegmentItem(
        job_id=1, photo_path=photo, output_dir=tmp_path, params={}, clicks={1: (200, 120)}
    )
    mask_path = processor.segment(item).masks[1].mask

    result = processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=_rim(tmp_path / "rim.png"),
            output_dir=tmp_path,
            params={"seed": 1},
            masks={1: mask_path},
        )
    )

    output = np.asarray(Image.open(result.result_path).convert("RGB"))
    original = np.asarray(Image.open(photo).convert("RGB"))
    inside = np.asarray(Image.open(mask_path).convert("L")) > 0
    assert output.shape == original.shape
    assert np.array_equal(output[~inside], original[~inside])
    assert not np.array_equal(output[inside], original[inside])


def test_detected_wheels_get_widened_masks_and_boxes_and_empty_masks_are_skipped(
    tmp_path: Path,
) -> None:
    class HalfBlindSegmenter(DiscSegmenter):
        """SAM nie znajduje niczego w drugim boxie."""

        def segment_boxes(
            self, image: Image.Image, boxes: list[tuple[float, float, float, float]]
        ) -> list[Image.Image]:
            masks = super().segment_boxes(image, boxes)
            masks[1] = Image.new("L", image.size, 0)
            return masks

    photo = noise_file(tmp_path / "input.png")
    processor = WheelSwapProcessor(
        detector=BoxDetector([(100, 70, 201, 171), (300, 10, 340, 50), (250, 150, 300, 200)]),
        segmenter=HalfBlindSegmenter(),
        editor=SolidEditor(),
    )

    result = processor.detect(
        DetectItem(job_id=1, photo_path=photo, output_dir=tmp_path, params={})
    )

    first, second = result.wheels
    mask = np.asarray(Image.open(first.mask).convert("L")) > 0
    # Koło o boku 101 px (dysk r=50 wokół (150, 120)) poszerzone o 3% boku = 3 px.
    assert mask[120, 150 + 53] and not mask[120, 150 + 56]
    assert first.box == (97, 67, 204, 174)
    assert first.mask.is_relative_to(tmp_path) and first.overlay.is_relative_to(tmp_path)
    assert first.mask != second.mask
    assert np.asarray(Image.open(second.mask).convert("L"))[175, 275] > 0


def test_detection_without_wheels_gives_empty_result(tmp_path: Path) -> None:
    photo = noise_file(tmp_path / "input.png")
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(), editor=SolidEditor()
    )
    item = DetectItem(job_id=1, photo_path=photo, output_dir=tmp_path, params={})
    assert processor.detect(item).wheels == []


def _two_wheels(processor: WheelSwapProcessor, photo: Path, out: Path) -> dict[int, Path]:
    item = SegmentItem(
        job_id=1, photo_path=photo, output_dir=out, params={}, clicks={7: (90, 120), 8: (300, 120)}
    )
    return {wheel_id: files.mask for wheel_id, files in processor.segment(item).masks.items()}


def test_generation_reports_progress_per_wheel_and_denoising_step(tmp_path: Path) -> None:
    photo = noise_file(tmp_path / "input.png")
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=30), editor=SolidEditor(steps=2)
    )
    reports: list[Progress] = []

    processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=_rim(tmp_path / "rim.png"),
            output_dir=tmp_path,
            params={"seed": 1},
            masks=_two_wheels(processor, photo, tmp_path),
            on_progress=reports.append,
        )
    )

    assert reports == [
        Progress(wheel=1, wheels=2),
        Progress(wheel=1, wheels=2, step=1, steps=2),
        Progress(wheel=1, wheels=2, step=2, steps=2),
        Progress(wheel=2, wheels=2),
        Progress(wheel=2, wheels=2, step=1, steps=2),
        Progress(wheel=2, wheels=2, step=2, steps=2),
    ]


def test_exception_from_progress_callback_stops_generation_before_next_wheel(
    tmp_path: Path,
) -> None:
    photo = noise_file(tmp_path / "input.png")
    editor = SolidEditor(steps=2)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=30), editor=editor
    )

    def cancel_after_first_wheel(progress: Progress) -> None:
        if progress.wheel == 2:
            raise JobCancelled

    with pytest.raises(JobCancelled):
        processor.generate(
            GenerateItem(
                job_id=1,
                photo_path=photo,
                rim_path=_rim(tmp_path / "rim.png"),
                output_dir=tmp_path,
                params={"seed": 1},
                masks=_two_wheels(processor, photo, tmp_path),
                on_progress=cancel_after_first_wheel,
            )
        )

    assert len(editor.calls) == 1 and editor.steps_run == 2
    assert not (tmp_path / "result.png").exists()


def _two_wheel_result(tmp_path: Path) -> tuple[Path, dict[int, Path], Path]:
    """Zdjęcie 400x240 z ICC, koła 3 i 4 wygenerowane na zielono: (zdjęcie, maski, wynik)."""
    photo = tmp_path / "input.png"
    rng = np.random.default_rng(5)
    Image.fromarray(rng.integers(0, 256, (240, 400, 3), dtype=np.uint8)).save(
        photo, icc_profile=display_profile()
    )
    segmenter = DiscSegmenter(radius=40)
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=segmenter, editor=SolidEditor()
    )
    masks = processor.segment(
        SegmentItem(
            job_id=1,
            photo_path=photo,
            output_dir=tmp_path,
            params={},
            clicks={3: (100, 120), 4: (300, 120)},
        )
    ).masks
    result = processor.generate(
        GenerateItem(
            job_id=1,
            photo_path=photo,
            rim_path=_rim(tmp_path / "rim.png"),
            output_dir=tmp_path,
            params={"seed": 7},
            masks={wheel_id: files.mask for wheel_id, files in masks.items()},
        )
    )
    return photo, {wheel_id: files.mask for wheel_id, files in masks.items()}, result.result_path


def test_regenerating_one_wheel_keeps_every_pixel_outside_its_mask_bit_for_bit(
    tmp_path: Path,
) -> None:
    photo, masks, previous = _two_wheel_result(tmp_path)
    previous_bytes = previous.read_bytes()
    editor = SolidEditor(color=(200, 0, 200))
    processor = WheelSwapProcessor(detector=BoxDetector(), segmenter=DiscSegmenter(), editor=editor)
    reports: list[Progress] = []

    result = processor.regenerate_wheel(
        RegenerateWheelItem(
            job_id=1,
            photo_path=photo,
            result_path=previous,
            rim_path=tmp_path / "rim.png",
            output_dir=tmp_path,
            params={"seed": 7, "prompt_suffix": "Matte."},
            wheel_id=4,
            mask_path=masks[4],
            seed=99,
            on_progress=reports.append,
        )
    )

    [call] = editor.calls  # tylko ponawiane koło, z nowym seedem i tym samym promptem
    assert call.seed == 99 and call.prompt.endswith("Photorealistic. Matte.")
    assert result.result_path != previous and result.result_path.is_relative_to(tmp_path)
    assert previous.read_bytes() == previous_bytes  # poprzedni wynik nietknięty
    with Image.open(result.result_path) as output:
        assert output.format == "PNG" and output.size == (400, 240)
        assert (result.width, result.height) == (400, 240)
        assert output.info.get("icc_profile") == display_profile()
        pixels = np.asarray(output.convert("RGB"))
    with Image.open(previous) as before:
        old = np.asarray(before.convert("RGB"))
    outside = np.asarray(Image.open(masks[4]).convert("L")) == 0
    assert np.array_equal(pixels[outside], old[outside])  # w tym całe koło nr 3
    assert (pixels[np.asarray(disc_mask((400, 240), (100, 120), 30)) > 0] == (10, 200, 30)).all()
    assert (pixels[np.asarray(disc_mask((400, 240), (300, 120), 30)) > 0] == (200, 0, 200)).all()
    assert reports[0] == Progress(wheel=1, wheels=1)
    assert reports[-1] == Progress(wheel=1, wheels=1, step=1, steps=1)
