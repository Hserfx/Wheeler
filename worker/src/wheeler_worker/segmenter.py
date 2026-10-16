"""Maski kół (z kliknięcia albo z boxów detektora): interfejs i adapter SAM 2.1 (transformers, MPS).

Ciężkie biblioteki (torch, transformers) importujemy leniwie, przy pierwszym użyciu modelu,
więc testy i atrapa (`placeholder`) ich nie ładują.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image


class Segmenter(Protocol):
    def segment_click(self, image: Image.Image, point: tuple[float, float], /) -> Image.Image:
        """Maska L (0/255) całego koła (opona + felga) w rozmiarze `image`, bez poszerzania."""
        ...

    def segment_boxes(
        self, image: Image.Image, boxes: list[tuple[float, float, float, float]], /
    ) -> list[Image.Image]:
        """Maska L (0/255) całego koła dla każdego boxu (x0, y0, x1, y1), w kolejności boxów."""
        ...

    def release(self) -> None:
        """Zwalnia model z pamięci (następne użycie załaduje go ponownie)."""
        ...


SAM_REPO = "facebook/sam2.1-hiera-large"
SAM_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"


class Sam2Segmenter:
    """SAM 2.1 z promptem punktowym, dwustopniowo (spike 01, `sam2.md`).

    Pojedynczy klik daje zwykle samą felgę albo logo, więc: klik -> maska o najwyższym IoU ->
    kwadrat `grow` razy wokół niej -> drugi prompt box + ten sam punkt -> całe koło.
    Model ładuje się przy pierwszym kliknięciu i zostaje w pamięci (~3 GB) do `release()`.
    """

    def __init__(self, repo: str = SAM_REPO, revision: str = SAM_REVISION, grow: float = 1.5):
        self.repo = repo
        self.revision = revision
        self.grow = grow
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        kwargs: dict[str, Any] = {"revision": self.revision, "local_files_only": True}
        self._processor = Sam2Processor.from_pretrained(self.repo, **kwargs)
        model: Any = Sam2Model.from_pretrained(self.repo, **kwargs)
        self._model = model.to(self._device).eval()

    def release(self) -> None:
        """Zwalnia model (np. przed mflux, żeby oba nie zajmowały pamięci naraz)."""
        if self._model is None:
            return
        import gc

        import torch

        self._model = self._processor = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _embed(self, rgb: Image.Image) -> Any:
        self._load()
        embedded = self._processor(images=rgb, return_tensors="pt").to(self._device)
        return self._model.get_image_embeddings(embedded["pixel_values"])

    def segment_boxes(
        self, image: Image.Image, boxes: list[tuple[float, float, float, float]]
    ) -> list[Image.Image]:
        """Wszystkie boxy jednym wywołaniem dekodera (embedding obrazu liczony raz)."""
        import numpy as np
        import torch

        if not boxes:
            return []
        rgb = image.convert("RGB")
        with torch.no_grad():
            embeddings = self._embed(rgb)
            prompt = [[[float(v) for v in box] for box in boxes]]
            inputs = self._processor(images=rgb, input_boxes=prompt, return_tensors="pt").to(
                self._device
            )
            out = self._model(
                image_embeddings=embeddings,
                input_boxes=inputs["input_boxes"],
                multimask_output=False,
            )
            masks = self._processor.post_process_masks(
                out.pred_masks.cpu(), inputs["original_sizes"].cpu()
            )[0]
        return [
            Image.fromarray(masks[i, 0].numpy().astype(np.uint8) * 255) for i in range(len(boxes))
        ]

    def segment_click(self, image: Image.Image, point: tuple[float, float]) -> Image.Image:
        import numpy as np
        import torch

        rgb = image.convert("RGB")
        points, labels = [[[[float(point[0]), float(point[1])]]]], [[[1]]]
        with torch.no_grad():
            embeddings = self._embed(rgb)
            processor, model, device = self._processor, self._model, self._device

            first = processor(
                images=rgb, input_points=points, input_labels=labels, return_tensors="pt"
            ).to(device)
            out = model(
                image_embeddings=embeddings,
                input_points=first["input_points"],
                input_labels=first["input_labels"],
                multimask_output=True,
            )
            candidates = processor.post_process_masks(
                out.pred_masks.cpu(), first["original_sizes"].cpu()
            )[0]
            rim = candidates[0, int(out.iou_scores[0, 0].argmax())].numpy()
            ys, xs = np.nonzero(rim)
            if xs.size == 0:
                raise ValueError("SAM 2 nie znalazł obiektu w miejscu kliknięcia")
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            half = max(xs.max() - xs.min(), ys.max() - ys.min()) * self.grow / 2
            box = [[float(cx - half), float(cy - half), float(cx + half), float(cy + half)]]

            second = processor(
                images=rgb,
                input_boxes=[box],
                input_points=points,
                input_labels=labels,
                return_tensors="pt",
            ).to(device)
            out = model(
                image_embeddings=embeddings,
                input_boxes=second["input_boxes"],
                input_points=second["input_points"],
                input_labels=second["input_labels"],
                multimask_output=False,
            )
            wheel = processor.post_process_masks(
                out.pred_masks.cpu(), second["original_sizes"].cpu()
            )[0][0, 0].numpy()
        return Image.fromarray(wheel.astype(np.uint8) * 255)
