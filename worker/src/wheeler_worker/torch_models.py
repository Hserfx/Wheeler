"""Wspólne dla modeli PyTorch w workerze (Florence-2, SAM 2, BiRefNet): wybór urządzenia
i zwalnianie pamięci. `torch` importujemy leniwie, więc testy i atrapa go nie ładują."""

from __future__ import annotations


def torch_device() -> str:
    """`mps` (GPU Apple Silicon), gdy dostępne, inaczej `cpu`."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def free_torch_memory() -> None:
    """Po porzuceniu referencji do modelu: odzyskuje pamięć procesu i cache MPS."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
