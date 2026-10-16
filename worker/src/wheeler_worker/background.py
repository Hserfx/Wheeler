"""Usuwanie tła ze zdjęcia felgi: interfejs i adapter BiRefNet (torch, MPS).

Ciężkie biblioteki importujemy leniwie, przy pierwszym użyciu modelu (jak `segmenter.py`).
Wagi: `uv run hf download ZhengPeng7/BiRefNet --revision <BIREFNET_REVISION>`
(ok. 0,9 GB w cache HF); model ma kod w repozytorium (`trust_remote_code`), stąd przypięta
rewizja i zależności `timm`, `kornia`, `einops`.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver


class BackgroundRemover(Protocol):
    def foreground(self, image: Image.Image, /) -> Image.Image:
        """Maska L (0..255, miękka krawędź) pierwszego planu w rozmiarze `image`."""
        ...

    def release(self) -> None:
        """Zwalnia model z pamięci (następne użycie załaduje go ponownie)."""
        ...


BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BIREFNET_SIZE = 1024  # model liczy na kwadracie 1024 px
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class BiRefNetRemover:
    """BiRefNet (dichotomous image segmentation) w fp16 na MPS: ~1 GB, ~0,5 s na zdjęcie."""

    def __init__(
        self,
        repo: str = BIREFNET_REPO,
        revision: str = BIREFNET_REVISION,
        observer: LoadObserver = NO_OBSERVER,
    ) -> None:
        self.repo = repo
        self.revision = revision
        self.observer = observer
        self._model: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self.observer.loading():
            import torch
            from transformers import AutoModelForImageSegmentation

            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            model: Any = AutoModelForImageSegmentation.from_pretrained(
                self.repo, revision=self.revision, trust_remote_code=True, local_files_only=True
            )
            model = model.to(self._device).eval()
            self._model = model.half() if self._device == "mps" else model

    def release(self) -> None:
        if self._model is None:
            return
        import gc

        import torch

        self._model = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def foreground(self, image: Image.Image) -> Image.Image:
        import numpy as np
        import torch

        self._load()
        model = self._model
        square = image.convert("RGB").resize(
            (BIREFNET_SIZE, BIREFNET_SIZE), Image.Resampling.BILINEAR
        )
        pixels = torch.from_numpy(np.array(square)).permute(2, 0, 1)[None].float() / 255
        mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_STD).view(1, 3, 1, 1)
        batch = ((pixels - mean) / std).to(self._device, dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            prediction = model(batch)[-1].sigmoid().float().cpu()[0, 0].numpy()
        alpha = Image.fromarray((prediction * 255).round().astype(np.uint8))
        return alpha.resize(image.size, Image.Resampling.BILINEAR)
