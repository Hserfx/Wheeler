"""Punkt odniesienia: ta sama edycja (wycinek koła + felga) w diffusers bf16 na MPS.

Dwuetapowe ładowanie i obejście F.pad jak w ../image-gen-model-test/generate.py (tamten katalog
nie jest modyfikowany; używamy tylko jego venv z diffusers z gita):

    gpu-run.sh /Users/hugo/marcel/image-gen-model-test/.venv/bin/python scripts/diffusers_baseline.py

Wejścia (wycinki) bierze z przebiegu mflux out/edit/q8_1024_mask/<case>_input.png, żeby edycja była identyczna.
Wynik: out/edit/diffusers_bf16_1024/<case>_generated.png + report.json (kompozycję robi figures_edit.py).
"""

from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline
from diffusers.models.autoencoders import autoencoder_kl_qwenimage21
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "spike" / "samples"
OUT = Path(__file__).resolve().parents[1] / "out" / "edit"
MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
RES = 1024
SEED = 42
CASES = [
    ("golf_front", "rim_multispoke_silver.jpg"),
    ("sorento_front", "rim_5twin_black.jpg"),
    ("vios_front", "rim_mesh_silver_angle.jpg"),
]
# Ten sam szablon co w bench_mflux.py (obraz 1 = wycinek, obraz 2 = felga).
PROMPT = (
    "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. "
    "Keep the tire, the car body, the wheel arch, the brake caliper position, the lighting, the shadows "
    "and the camera angle of image 1 unchanged. The new rim must follow the perspective of the wheel in image 1 "
    "and fill the tire opening exactly. Photorealistic."
)

_avg_down_forward = autoencoder_kl_qwenimage21.QwenImage21AvgDown3D.forward


def _avg_down_forward_mps_safe(self, x):  # diffusers#14902
    pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
    if pad_t:
        x = torch.cat([x.new_zeros(x.shape[0], x.shape[1], pad_t, *x.shape[3:]), x], dim=2)
    return _avg_down_forward(self, x)


autoencoder_kl_qwenimage21.QwenImage21AvgDown3D.forward = _avg_down_forward_mps_safe


class Peak:
    def __init__(self):
        self.value = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.value = max(self.value, torch.mps.driver_allocated_memory())
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def gb(n):
    return round(n / 1e9, 2)


def main() -> None:
    out = OUT / "diffusers_bf16_1024"
    out.mkdir(parents=True, exist_ok=True)
    src = OUT / "q8_1024_mask"
    report: dict = {"cases": {}}
    inputs = {
        name: [Image.open(src / f"{name}_input.png").convert("RGBA"), Image.open(SAMPLES / rim).convert("RGBA")]
        for name, rim in CASES
    }
    with Peak() as peak:
        t0 = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, dtype=torch.bfloat16).to("mps")
        report["stage1_load_s"] = round(time.perf_counter() - t0, 1)
        encoded = {}
        for name, images in inputs.items():
            t0 = time.perf_counter()
            condition = []
            for img in images:
                w, h, _ = calculate_dimensions(RES * RES, img.width / img.height)
                condition.append(pipe.image_processor.resize(img, width=w, height=h))
            with torch.no_grad():
                enc = pipe.encode_prompt(PROMPT, image=condition, device="mps")
            encoded[name] = tuple(t.cpu() if t is not None else None for t in enc)
            report["cases"][name] = {"encode_s": round(time.perf_counter() - t0, 1)}
        report["stage1_peak_gb"] = gb(peak.value)
        del pipe
        gc.collect()
        torch.mps.empty_cache()

        peak.value = 0
        t0 = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, text_encoder=None, dtype=torch.bfloat16).to("mps")
        report["stage2_load_s"] = round(time.perf_counter() - t0, 1)
        for name, images in inputs.items():
            enc = tuple(t.to("mps") if t is not None else None for t in encoded[name])
            pipe.encode_prompt = lambda *a, _enc=enc, **k: _enc
            t0 = time.perf_counter()
            with torch.no_grad():
                image = pipe(
                    prompt=PROMPT,
                    image=images,
                    width=images[0].width,
                    height=images[0].height,
                    output_resolution=RES,
                    generator=torch.Generator("cpu").manual_seed(SEED),
                ).images[0]
            report["cases"][name]["generate_s"] = round(time.perf_counter() - t0, 1)
            image.convert("RGB").save(out / f"{name}_generated.png")
            print(name, report["cases"][name], flush=True)
        report["stage2_peak_gb"] = gb(peak.value)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
