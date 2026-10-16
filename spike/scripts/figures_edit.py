"""Rysunki i analiza edycji (bez modelu): porównanie konfiguracji, kompozycja, test siatki 992/1024/1056.

    uv run python scripts/figures_edit.py
Czyta out/edit/*, zapisuje docs/spike/img/*.jpg i out/edit/analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bench_mflux import CASES, OUT, SAMPLES, mask_box  # noqa: E402
from wheeler_spike.wheel_crop import Box, composite, crop_box_for, grow_mask

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "spike" / "img"
EDIT = OUT / "edit"
TILE = 300


def font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def label(im: Image.Image, text: str) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width, 20), fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 0), font=font(14))
    return im


def grid(rows: list[list[Image.Image]], path: Path) -> None:
    w = max(sum(t.width for t in r) for r in rows)
    h = sum(max(t.height for t in r) for r in rows)
    sheet = Image.new("RGB", (w, h), (30, 30, 30))
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y))
            x += t.width
        y += max(t.height for t in r)
    sheet.save(path, quality=82, optimize=True)


def tile(path: Path, text: str) -> Image.Image:
    if not path.exists():
        return label(Image.new("RGB", (TILE, TILE), (60, 60, 60)), text + " (brak)")
    im = Image.open(path).convert("RGB")
    im.thumbnail((TILE, TILE), Image.Resampling.LANCZOS)
    return label(im, text)


def secs(tag: str, case: str) -> str:
    rep = EDIT / tag / "report.json"
    if not rep.exists():
        return ""
    data = json.loads(rep.read_text())["cases"].get(case, {})
    total = data.get("total_s") or data.get("generate_s")
    return f" · {total} s" if total else ""


def configs_figure() -> None:
    columns = [
        ("q8_1024_mask", "mflux q8 1024"),
        ("q4_1024_mask", "mflux q4 1024"),
        ("q8_1536_mask", "mflux q8 1536"),
        ("q4_1536_mask", "mflux q4 1536"),
        ("diffusers_bf16_1024", "diffusers bf16 1024"),
    ]
    rows = []
    for name, _, _, rim in CASES:
        row = [tile(EDIT / "q8_1024_mask" / f"{name}_input.png", "wycinek"), tile(SAMPLES / rim, "felga")]
        row += [tile(EDIT / tag / f"{name}_generated.png", title + secs(tag, name)) for tag, title in columns]
        rows.append(row)
    grid(rows, FIG / "configs.jpg")


def variants_figure() -> None:
    columns = [
        ("q8_1024_mask", "[wycinek, felga] + maska"),
        ("q8_1024_nomask", "[wycinek, felga] bez maski"),
        ("q8_1024_rimfirst", "[felga, wycinek] bez maski"),
        ("q8_1024_mask_bf16", "jak 1., compute bf16"),
    ]
    rows = []
    for name, *_ in CASES:
        rows.append([tile(EDIT / tag / f"{name}_generated.png", title) for tag, title in columns])
    grid(rows, FIG / "variants.jpg")


def wheel_mask(car: str, index: int) -> tuple[Image.Image, Box, Image.Image]:
    original = Image.open(SAMPLES / car).convert("RGB")
    sam = Image.open(OUT / "masks" / f"{Path(car).stem}_{index}.png")
    wheel = mask_box(sam)
    mask = grow_mask(sam, pixels=max(2, round(max(wheel.width, wheel.height) * 0.03)))
    return original, crop_box_for(wheel, original.size, margin=0.25), mask


def composite_figure(analysis: dict) -> None:
    rows = []
    for name, car, index, _ in CASES:
        original, crop, mask = wheel_mask(car, index)
        outside = np.asarray(mask) == 0
        row = []
        for tag, title in (("q8_1024_mask", "mflux q8 1024"), ("diffusers_bf16_1024", "diffusers bf16")):
            gen = EDIT / tag / f"{name}_generated.png"
            if not gen.exists():
                continue
            side = max(mask_box(mask).width, mask_box(mask).height)
            final = composite(original, Image.open(gen), crop, mask, feather=max(2, round(side * 0.015)))
            identical = bool(np.array_equal(np.asarray(final)[outside], np.asarray(original)[outside]))
            analysis.setdefault("outside_mask_identical", {})[f"{tag}/{name}"] = identical
            view = final.copy()
            view.thumbnail((640, 640), Image.Resampling.LANCZOS)
            row.append(label(view, f"wynik: {title}"))
        before = original.copy()
        before.thumbnail((640, 640), Image.Resampling.LANCZOS)
        rows.append([label(before, "oryginał"), *row])
    grid(rows, FIG / "composite.jpg")


def tile_coherence(path: Path, period: int = 16) -> float:
    """Siła wzoru powtarzającego się co `period` px (kafel latentu Qwen-Image-2.1 = 16 px).

    Obraz górnoprzepustowy składamy modulo `period` (średnia po wszystkich kaflach). Treść obrazu się
    uśrednia, a artefakt wyrównany do kafli sumuje się. Wynik = std wzoru / std oczekiwane dla szumu
    (≈1 bez siatki; syntetyczna siatka o amplitudzie 1 poziomu jasności daje wyraźnie więcej).
    """
    gray = Image.open(path).convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    high = arr - np.asarray(gray.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32)
    h, w = (arr.shape[0] // period) * period, (arr.shape[1] // period) * period
    tiles = high[:h, :w].reshape(h // period, period, w // period, period)
    pattern = tiles.mean(axis=(0, 2))
    expected = high[:h, :w].std() / np.sqrt((h // period) * (w // period))
    return round(float(pattern.std() / expected), 2)


def grid_figure(analysis: dict) -> None:
    rows = []
    for name, *_ in CASES:
        row = []
        for res in (992, 1024, 1056):
            path = EDIT / f"q8_{res}_mask" / f"{name}_generated.png"
            if not path.exists():
                continue
            score = tile_coherence(path)
            control = tile_coherence(path.with_name(f"{name}_input.png"))
            analysis.setdefault("grid_tile16", {})[f"{res}/{name}"] = {"generated": score, "input_photo": control}
            im = Image.open(path).convert("RGB")
            # 160x160 px z obszaru opony/felgi (dolna-lewa ćwiartka koła), powiększenie 2x bez interpolacji
            cx, cy = im.width // 2, im.height // 2
            box = (cx - im.width // 5 - 80, cy + im.height // 8 - 80, cx - im.width // 5 + 80, cy + im.height // 8 + 80)
            zoom = im.crop(box).resize((320, 320), Image.Resampling.NEAREST)
            row.append(label(zoom, f"{res} px · wzór 16 px: {score} (zdjęcie: {control})"))
        if row:
            rows.append(row)
    if rows:
        grid(rows, FIG / "grid_992_1024_1056.jpg")


def synthetic_grid_check(analysis: dict) -> None:
    """Kalibracja miary: ta sama generacja z dodaną siatką 16 px o amplitudzie 1 i 2 poziomów jasności."""
    src = EDIT / "q8_1024_mask" / "golf_front_generated.png"
    a = np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    lines = ((xx % 16) == 0) | ((yy % 16) == 0)
    tmp = EDIT / "_grid_check.png"
    for amp in (1, 2):
        Image.fromarray(np.clip(np.rint(a + amp * lines[..., None]), 0, 255).astype(np.uint8)).save(tmp)
        analysis.setdefault("grid_tile16_synthetic", {})[f"amp{amp}"] = tile_coherence(tmp)
    tmp.unlink()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    analysis: dict = {}
    synthetic_grid_check(analysis)
    configs_figure()
    variants_figure()
    composite_figure(analysis)
    grid_figure(analysis)
    (EDIT / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
