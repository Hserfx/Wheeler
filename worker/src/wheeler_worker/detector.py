"""Wykrywanie kół na zdjęciu auta: interfejs i adapter Florence-2-large (transformers, MPS).

Detektor daje tylko przybliżone prostokąty kół; precyzyjny kształt wyznacza SAM 2 z promptem
box (`Segmenter.segment_boxes`). Ciężkie biblioteki importujemy leniwie, jak w `segmenter.py`.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver
from wheeler_worker.torch_models import free_torch_memory, torch_device

BoxF = tuple[float, float, float, float]  # x0, y0, x1, y1 w pikselach zdjęcia


class WheelDetector(Protocol):
    def detect(self, image: Image.Image, /) -> list[BoxF]:
        """Prostokąty wszystkich widocznych kół (może być pusta lista)."""
        ...

    def release(self) -> None:
        """Zwalnia model z pamięci (następne użycie załaduje go ponownie)."""
        ...


FLORENCE_REPO = "florence-community/Florence-2-large"
FLORENCE_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


class FlorenceWheelDetector:
    """Florence-2-large, phrase grounding frazy `wheel` (wybór spike'a 01, `detector.md`).

    Wynik nie ma `score` (to generacja tekstu), więc nie ma progów. Wykrywa też koła skuterów
    i rowerów; użytkownik usuwa zbędne maski kliknięciem. Model (~3 GB) ładuje się przy
    pierwszej detekcji i zostaje w pamięci do `release()`.
    """

    def __init__(
        self,
        repo: str = FLORENCE_REPO,
        revision: str = FLORENCE_REVISION,
        phrase: str = "wheel",
        observer: LoadObserver = NO_OBSERVER,
    ) -> None:
        self.repo = repo
        self.observer = observer
        self.revision = revision
        self.phrase = phrase
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        self._dtype: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self.observer.loading():
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration

            self._device = torch_device()
            self._dtype = torch.float16 if self._device == "mps" else torch.float32
            kwargs: dict[str, Any] = {"revision": self.revision, "local_files_only": True}
            auto_processor: Any = AutoProcessor  # bez typów w transformers
            self._processor = auto_processor.from_pretrained(self.repo, **kwargs)
            model: Any = Florence2ForConditionalGeneration.from_pretrained(
                self.repo, dtype=self._dtype, **kwargs
            )
            self._model = model.to(self._device).eval()

    def release(self) -> None:
        if self._model is None:
            return
        self._model = self._processor = None
        free_torch_memory()

    def detect(self, image: Image.Image) -> list[BoxF]:
        import torch

        self._load()
        rgb = image.convert("RGB")
        processor, model = self._processor, self._model
        with torch.no_grad():
            inputs = processor(
                text=GROUNDING_TASK + self.phrase, images=rgb, return_tensors="pt"
            ).to(self._device, self._dtype)
            ids = model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(raw, task=GROUNDING_TASK, image_size=rgb.size)
        boxes = parsed[GROUNDING_TASK]["bboxes"]
        return [(float(x0), float(y0), float(x1), float(y1)) for x0, y0, x1, y1 in boxes]
