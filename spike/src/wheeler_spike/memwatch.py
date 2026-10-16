"""Próbkowanie szczytowej pamięci w tle (spike, nie kod produkcyjny)."""

from __future__ import annotations

import threading
import time

import psutil


class PeakMemory:
    """Co `interval` s zapisuje maksimum RSS procesu oraz pamięci sterownika MPS / MLX (jeśli dostępne)."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss = 0
        self.peak_mps = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
        try:
            import torch

            if torch.backends.mps.is_available():
                self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def report(self) -> dict[str, float]:
        return {"peak_rss_gb": round(self.peak_rss / 1e9, 2), "peak_mps_gb": round(self.peak_mps / 1e9, 2)}
