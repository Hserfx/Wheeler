"""Rysunki do raportu: porównanie detektorów i maski SAM 2.1 (bez modelu, czyta out/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "spike" / "samples"
OUT = Path(__file__).resolve().parents[1] / "out"
FIG = ROOT / "docs" / "spike" / "img"
CARS = ["car_side_golf.jpg", "car_34front_sorento.jpg", "car_occluded_vios.jpg"]
VARIANTS = [
    ("gdino-tiny", "gdino", "Grounding DINO tiny"),
    ("gdino-base", "gdino", "Grounding DINO base"),
    ("florence2-base", "grounding", "Florence-2 base (grounding)"),
    ("florence2-large", "grounding", "Florence-2 large (grounding)"),
]
TILE = 520


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def detect_grid() -> None:
    reports = {name: json.loads((OUT / "detect" / f"{name}.json").read_text()) for name, _, _ in VARIANTS}
    tiles = []
    for car in CARS:
        base = Image.open(SAMPLES / car).convert("RGB")
        scale = TILE / base.width
        for name, mode, title in VARIANTS:
            im = base.resize((TILE, round(base.height * scale)))
            d = ImageDraw.Draw(im)
            run = reports[name]["runs"][mode][car]
            for det in run["detections"]:
                d.rectangle([v * scale for v in det["box"]], outline=(255, 0, 255), width=3)
            d.rectangle((0, 0, TILE, 22), fill=(0, 0, 0))
            d.text((4, 3), f"{title} · {run['warm_s']} s", fill=(255, 255, 0), font=font(15))
            tiles.append(im)
    row_h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (TILE * len(VARIANTS), row_h * len(CARS)), (30, 30, 30))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % len(VARIANTS)) * TILE, (i // len(VARIANTS)) * row_h))
    sheet.save(FIG / "detectors.jpg", quality=80, optimize=True)


def mask_grid() -> None:
    tiles = []
    for car in CARS:
        stem = Path(car).stem
        base = Image.open(SAMPLES / car).convert("RGB")
        overlay = np.asarray(base).astype(np.float32)
        i = 0
        while (OUT / "masks" / f"{stem}_{i}.png").exists():
            m = np.asarray(Image.open(OUT / "masks" / f"{stem}_{i}.png")) > 0
            overlay[m] = overlay[m] * 0.45 + np.array([255, 0, 255]) * 0.55
            i += 1
        im = Image.fromarray(overlay.astype(np.uint8))
        height = 420
        tiles.append(im.resize((round(base.width * height / base.height), height)))
    width = sum(t.width for t in tiles)
    sheet = Image.new("RGB", (width, max(t.height for t in tiles)), (30, 30, 30))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width
    sheet.save(FIG / "sam_masks.jpg", quality=80, optimize=True)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    detect_grid()
    mask_grid()
