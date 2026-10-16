"""Diagnostyka: czas i szczytowa pamięć MLX w poszczególnych fazach edycji mflux (jedno koło).

    gpu-run.sh uv run python scripts/profile_phases.py --bits 8 --res 1024 [--no-kv-cache] [--tiling]
"""

from __future__ import annotations

import argparse
import functools
import time

import mlx.core as mx
from mflux.models.common.config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.model.qwen21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from PIL import Image

from bench_mflux import OUT, PROMPT_WHEEL_FIRST, SAMPLES, WEIGHTS, mask_box  # noqa: E402
from wheeler_spike.wheel_crop import crop_box_for, grow_mask, working_size


def traced(label, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        mx.eval(r)
        print(f"{label:>14}: {time.perf_counter() - t0:6.2f} s  peak {mx.get_peak_memory() / 1e9:5.2f} GB  "
              f"active {mx.get_active_memory() / 1e9:5.2f} GB", flush=True)
        return r

    return wrapper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--no-kv-cache", action="store_true")
    p.add_argument("--rim-res", type=int, default=None, help="osobny budżet pikseli dla felgi (patch)")
    p.add_argument("--tiling", action="store_true")
    p.add_argument("--cache-limit-gb", type=float, default=None)
    p.add_argument("--fp16", action="store_true", help="compute_precision=float16 (PR #820)")
    args = p.parse_args()

    if args.cache_limit_gb is not None:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    model = QwenImage21Edit(
        model_path=str(WEIGHTS / f"qwen-image-2.1-turbo-mflux-q{args.bits}"),
        model_config=ModelConfig.qwen_image_21_turbo(),
        compute_precision=mx.float16 if args.fp16 else None,
    )
    if args.tiling:
        model.tiling_config = TilingConfig()
    mx.eval(model.parameters())
    print(f"weights active {mx.get_active_memory() / 1e9:.2f} GB")
    model._encode_prompt = traced("encode_prompt", model._encode_prompt)
    QwenImage21VAE.encode = traced("vae.encode", QwenImage21VAE.encode)
    QwenImage21Transformer.__call__ = traced("transformer", QwenImage21Transformer.__call__)
    VAEUtil.decode = staticmethod(traced("vae.decode", VAEUtil.decode))

    original = Image.open(SAMPLES / "car_side_golf.jpg").convert("RGB")
    sam = Image.open(OUT / "masks" / "car_side_golf_1.png")
    wheel = mask_box(sam)
    crop = crop_box_for(wheel, original.size)
    size = working_size(crop, args.res)
    crop_img = original.crop(crop).resize(size, Image.Resampling.LANCZOS)
    mask = grow_mask(sam, 8).crop(crop).resize(size)
    rim = Image.open(SAMPLES / "rim_multispoke_silver.jpg").convert("RGB")
    t0 = time.perf_counter()
    model.generate_image(
        seed=42, prompt=PROMPT_WHEEL_FIRST, image_paths=[crop_img, rim], width=size[0], height=size[1],
        output_resolution=args.res, mask_image=mask, use_kv_cache=not args.no_kv_cache,
    )
    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
