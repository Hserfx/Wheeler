"""Konwersja Qwen-Image-2.1-Turbo (bf16 z cache HF) do zapisanego checkpointu mflux q8 / q4.

Uruchamiać przez gpu-run.sh:
    gpu-run.sh uv run python scripts/convert_weights.py 8
    gpu-run.sh uv run python scripts/convert_weights.py 4

Wynik: ~/.cache/wheeler/qwen-image-2.1-turbo-mflux-q{bits}
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

bits = int(sys.argv[1])
dest = Path.home() / ".cache" / "wheeler" / f"qwen-image-2.1-turbo-mflux-q{bits}"
dest.parent.mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
model = QwenImage21Edit(quantize=bits, model_config=ModelConfig.qwen_image_21_turbo())
t_load = time.perf_counter() - t0
t0 = time.perf_counter()
model.save_model(str(dest))
t_save = time.perf_counter() - t0

size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
result = {
    "bits": bits,
    "path": str(dest),
    "load_quantize_s": round(t_load, 1),
    "save_s": round(t_save, 1),
    "size_gb": round(size / 1e9, 2),
    "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 2),
}
print(json.dumps(result))
