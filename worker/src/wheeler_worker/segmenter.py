"""Maski kół (z kliknięcia albo z boxów detektora): interfejs i adapter SAM 2.1 (transformers, MPS).

Ciężkie biblioteki (torch, transformers) importujemy leniwie, przy pierwszym użyciu modelu,
więc testy i atrapa (`placeholder`) ich nie ładują.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from wheeler_shared.failures import Failure, ProcessingFailed
from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver
from wheeler_worker.rim_geometry import RimCandidate, wheel_ellipse
from wheeler_worker.torch_models import free_torch_memory, torch_device


class Segmenter(Protocol):
    def segment_click(self, image: Image.Image, point: tuple[float, float], /) -> Image.Image:
        """Maska L (0/255) całego koła (opona + felga) w rozmiarze `image`, bez poszerzania."""
        ...

    def segment_boxes(
        self, image: Image.Image, boxes: list[tuple[float, float, float, float]], /
    ) -> list[Image.Image]:
        """Maska L (0/255) całego koła dla każdego boxu (x0, y0, x1, y1), w kolejności boxów."""
        ...

    def segment_rim(self, image: Image.Image, wheel_mask: Image.Image, /) -> list[RimCandidate]:
        """Kandydaci na maskę samej felgi koła `wheel_mask` (L, maska SAM bez poszerzenia):
        maski L (0/255) w rozmiarze `image` z oceną modelu; wybiera z nich
        `rim_geometry.measure_rim`. Pusta lista: nic nie znaleziono."""
        ...

    def release(self) -> None:
        """Zwalnia model z pamięci (następne użycie załaduje go ponownie)."""
        ...


SAM_REPO = "facebook/sam2.1-hiera-large"
SAM_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
# Felga (ticket 12): punkty negatywne na oponie, na tym promieniu elipsy wpisanej w bbox koła;
# box felgi = ten ułamek bboxa koła (rozpoznanie `prototype/plus-sizing-warp`, `rim_ellipse/`).
RIM_NEGATIVE_RADIUS = 0.92
RIM_BOX_FRACTION = 0.8


class Sam2Segmenter:
    """SAM 2.1 z promptem punktowym, dwustopniowo (spike 01, `sam2.md`).

    Pojedynczy klik daje zwykle samą felgę albo logo, więc: klik -> maska o najwyższym IoU ->
    kwadrat `grow` razy wokół niej -> drugi prompt box + ten sam punkt -> całe koło.
    Model ładuje się przy pierwszym kliknięciu i zostaje w pamięci (~3 GB) do `release()`.
    """

    def __init__(
        self,
        repo: str = SAM_REPO,
        revision: str = SAM_REVISION,
        grow: float = 1.5,
        observer: LoadObserver = NO_OBSERVER,
    ):
        self.repo = repo
        self.observer = observer
        self.revision = revision
        self.grow = grow
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self.observer.loading():
            from transformers import Sam2Model, Sam2Processor

            self._device = torch_device()
            kwargs: dict[str, Any] = {"revision": self.revision, "local_files_only": True}
            self._processor = Sam2Processor.from_pretrained(self.repo, **kwargs)
            model: Any = Sam2Model.from_pretrained(self.repo, **kwargs)
            self._model = model.to(self._device).eval()

    def release(self) -> None:
        """Zwalnia model (np. przed mflux, żeby oba nie zajmowały pamięci naraz)."""
        if self._model is None:
            return
        self._model = self._processor = None
        free_torch_memory()

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

    def segment_rim(self, image: Image.Image, wheel_mask: Image.Image) -> list[RimCandidate]:
        """Felga z dwóch promptów multimask (po 3 kandydatów): punkt w centroidzie maski koła
        z 8 punktami negatywnymi na oponie oraz box 0,8 bboxa koła z tym samym punktem.
        Sam punkt bez negatywów daje zwykle tylko logo albo piastę."""
        import numpy as np
        import torch

        wheel = wheel_ellipse(wheel_mask)
        if wheel is None:
            return []
        ys, xs = np.nonzero(np.asarray(wheel_mask.convert("L")) > 0)
        rgb = image.convert("RGB")
        center = [float(xs.mean()), float(ys.mean())]
        angles = np.deg2rad(np.arange(0, 360, 45))
        negative_xs, negative_ys = wheel.from_unit(
            RIM_NEGATIVE_RADIUS * np.cos(angles), RIM_NEGATIVE_RADIUS * np.sin(angles)
        )
        negatives = [[float(x), float(y)] for x, y in zip(negative_xs, negative_ys, strict=True)]
        half_width = RIM_BOX_FRACTION * wheel.width / 2
        half_height = RIM_BOX_FRACTION * wheel.height / 2
        box = [
            wheel.cx - half_width,
            wheel.cy - half_height,
            wheel.cx + half_width,
            wheel.cy + half_height,
        ]
        prompts: list[dict[str, Any]] = [
            {
                "input_points": [[[center, *negatives]]],
                "input_labels": [[[1] + [0] * len(negatives)]],
            },
            {"input_points": [[[center]]], "input_labels": [[[1]]], "input_boxes": [[box]]},
        ]
        candidates: list[RimCandidate] = []
        with torch.no_grad():
            embeddings = self._embed(rgb)
            for prompt in prompts:
                inputs = self._processor(images=rgb, return_tensors="pt", **prompt).to(self._device)
                out = self._model(
                    image_embeddings=embeddings,
                    multimask_output=True,
                    **{key: inputs[key] for key in prompt},
                )
                masks = self._processor.post_process_masks(
                    out.pred_masks.cpu(), inputs["original_sizes"].cpu()
                )[0][0]
                scores = out.iou_scores[0, 0].float().cpu().numpy()
                candidates.extend(
                    (Image.fromarray(masks[i].numpy().astype(np.uint8) * 255), float(scores[i]))
                    for i in range(masks.shape[0])
                )
        return candidates

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
                raise ProcessingFailed(Failure.NOTHING_AT_CLICK, f"SAM 2: nic w {point}")
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
