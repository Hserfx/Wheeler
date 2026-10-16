"""Konfiguracja workera (zmienne środowiskowe `WHEELER_*`)."""

from __future__ import annotations

from pathlib import Path

from wheeler_shared.settings import Settings
from wheeler_worker.rim_editor import DEFAULT_MODEL_PATH


class WorkerSettings(Settings):
    poll_interval: float = 1.0
    # Nazwa procesora z `wheeler_worker.processing.PROCESSORS`.
    processor: str = "wheel-swap"
    # Wagi Qwen-Image-2.1-Turbo skonwertowane do mflux (q4: `…-mflux-q4`, gdy brakuje pamięci).
    mflux_model_path: Path = DEFAULT_MODEL_PATH
