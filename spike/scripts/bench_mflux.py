"""Pomiar mflux Qwen-Image-2.1-Turbo: podmiana felgi na wycinku wokół koła.

Jeden proces = jedna kombinacja (kwantyzacja, rozdzielczość, wariant), żeby pomiar ładowania i
szczytowej pamięci był czysty. Uruchamiać przez gpu-run.sh:

    gpu-run.sh uv run python scripts/bench_mflux.py --bits 8 --res 1024
    gpu-run.sh uv run python scripts/bench_mflux.py --bits 8 --res 1024 --variant rimfirst

Wymaga masek z scripts/sam_masks.py (out/masks). Wynik: out/edit/<tag>/...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mflux.models.common.config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from PIL import Image

from wheeler_spike.memwatch import PeakMemory
from wheeler_spike.wheel_crop import Box, composite, crop_box_for, grow_mask, working_size

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "spike" / "samples"
OUT = Path(__file__).resolve().parents[1] / "out"
WEIGHTS = Path.home() / ".cache" / "wheeler"

# (nazwa, zdjęcie auta, indeks maski SAM, felga)
CASES = [
    ("golf_front", "car_side_golf.jpg", 1, "rim_multispoke_silver.jpg"),
    ("sorento_front", "car_34front_sorento.jpg", 0, "rim_5twin_black.jpg"),
    ("vios_front", "car_occluded_vios.jpg", 0, "rim_mesh_silver_angle.jpg"),
]

# Szablon promptu: obraz 1 = wycinek koła (źródło edycji, do niego odnosi się maska), obraz 2 = felga.
PROMPT_WHEEL_FIRST = (
    "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. "
    "Keep the tire, the car body, the wheel arch, the brake caliper position, the lighting, the shadows "
    "and the camera angle of image 1 unchanged. The new rim must follow the perspective of the wheel in image 1 "
    "and fill the tire opening exactly. Photorealistic."
)
PROMPT_RIM_FIRST = (
    "Replace the rim of the car wheel in image 2 with the alloy rim design shown in image 1. "
    "Keep the tire, the car body, the wheel arch, the brake caliper position, the lighting, the shadows "
    "and the camera angle of image 2 unchanged. The new rim must follow the perspective of the wheel in image 2 "
    "and fill the tire opening exactly. Photorealistic."
)


def mask_box(mask: Image.Image) -> Box:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    return Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def gb(n: int) -> float:
    return round(n / 1e9, 2)


STEP_TIMES: list[float] = []
_transformer_call = QwenImage21Transformer.__call__


def _timed_transformer(self, *a, **k):
    t0 = time.perf_counter()
    r = _transformer_call(self, *a, **k)
    mx.eval(r)
    STEP_TIMES.append(time.perf_counter() - t0)
    return r


QwenImage21Transformer.__call__ = _timed_transformer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, required=True)
    p.add_argument("--res", type=int, required=True)
    p.add_argument("--variant", default="mask", choices=["mask", "nomask", "rimfirst"])
    p.add_argument("--cases", default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None)
    p.add_argument("--bf16", action="store_true", help="bez compute_precision=float16")
    args = p.parse_args()
    tag = args.tag or f"q{args.bits}_{args.res}_{args.variant}" + ("_bf16" if args.bf16 else "")
    out = OUT / "edit" / tag
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASES if args.cases == "all" or c[0] in args.cases.split(",")]

    report: dict = {"tag": tag, "bits": args.bits, "res": args.res, "variant": args.variant,
                    "compute": "bf16" if args.bf16 else "fp16", "cases": {}}
    # Bufory zwolnione przez MLX trafiają do cache; bez limitu realne zużycie rośnie ponad "peak active".
    mx.set_cache_limit(2 * 1024**3)
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        model = QwenImage21Edit(
            model_path=str(WEIGHTS / f"qwen-image-2.1-turbo-mflux-q{args.bits}"),
            model_config=ModelConfig.qwen_image_21_turbo(),
            compute_precision=None if args.bf16 else mx.float16,
        )
        model.tiling_config = TilingConfig()  # kafelkowany dekoder VAE: -9 GB szczytu przy 1024 px
        mx.eval(model.parameters())  # mflux ładuje leniwie; wymuszamy wczytanie wag do pamięci
        report["load_s"] = round(time.perf_counter() - t0, 1)
        report["after_load_active_gb"] = gb(mx.get_active_memory())
        report["after_load_peak_gb"] = gb(mx.get_peak_memory())
        print("loaded", report, flush=True)

        for name, car, mask_index, rim in cases:
            original = Image.open(SAMPLES / car).convert("RGB")
            sam = Image.open(OUT / "masks" / f"{Path(car).stem}_{mask_index}.png")
            wheel = mask_box(sam)
            side = max(wheel.width, wheel.height)
            mask = grow_mask(sam, pixels=max(2, round(side * 0.03)))
            crop = crop_box_for(wheel, original.size, margin=0.25)
            size = working_size(crop, resolution=args.res)
            crop_img = original.crop(crop).resize(size, Image.Resampling.LANCZOS)
            crop_mask = mask.crop(crop).resize(size, Image.Resampling.BILINEAR)
            rim_img = Image.open(SAMPLES / rim).convert("RGB")
            if args.variant == "rimfirst":
                images, prompt, mflux_mask = [rim_img, crop_img], PROMPT_RIM_FIRST, None
            else:
                images, prompt = [crop_img, rim_img], PROMPT_WHEEL_FIRST
                mflux_mask = crop_mask if args.variant == "mask" else None

            mx.reset_peak_memory()
            STEP_TIMES.clear()
            t0 = time.perf_counter()
            result = model.generate_image(
                seed=args.seed,
                prompt=prompt,
                image_paths=images,
                width=size[0],
                height=size[1],
                output_resolution=args.res,
                mask_image=mflux_mask,
            )
            total = time.perf_counter() - t0
            generated = result.image.convert("RGB")
            final = composite(original, generated, crop, mask, feather=max(2, round(side * 0.015)))

            crop_img.save(out / f"{name}_input.png")
            generated.save(out / f"{name}_generated.png")
            final.save(out / f"{name}_final.png")
            report["cases"][name] = {
                "crop": list(crop),
                "working_size": list(size),
                "total_s": round(total, 1),
                "denoise_s": round(sum(STEP_TIMES), 1),
                "first_step_s": round(STEP_TIMES[0], 1),
                "next_steps_avg_s": round(sum(STEP_TIMES[1:]) / max(1, len(STEP_TIMES) - 1), 2),
                "peak_gb": gb(mx.get_peak_memory()),
            }
            print(name, report["cases"][name], flush=True)
    report["process"] = mem.report()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
