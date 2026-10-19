"""Edycja wycinka koła modelem: interfejs i adapter mflux Qwen-Image-2.1-Turbo.

MLX i mflux importujemy leniwie, przy pierwszym generowaniu, więc testy i atrapa
(`placeholder`) ich nie ładują. Ustawienia domyślne: spike 01 (`docs/DESIGN.md`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver

DEFAULT_MODEL_PATH = Path.home() / ".cache" / "wheeler" / "qwen-image-2.1-turbo-mflux-q8"

# Po każdym kroku denoisingu: (ukończone kroki, liczba kroków). Wyjątek z callbacku przerywa
# edycję (wychodzi z `edit`), więc anulowanie działa najpóźniej po bieżącym kroku.
StepCallback = Callable[[int, int], None]


class RimEditor(Protocol):
    def edit(
        self,
        images: list[Image.Image],
        *,
        mask: Image.Image,
        prompt: str,
        seed: int,
        size: tuple[int, int],
        resolution: int,
        on_step: StepCallback | None = None,
    ) -> Image.Image:
        """Edytuje `images[0]` (wycinek) według promptu; `images[1:]` to referencje (felga).

        `mask` (L, rozmiar `size`, biały = przemaluj) dotyczy pierwszego obrazu. Wynik ma
        rozmiar `size` (wielokrotności 32). `resolution` to budżet pikseli każdej referencji.
        `on_step` jest wołany po każdym kroku denoisingu (`StepCallback`).
        """
        ...

    def release(self) -> None:
        """Zwalnia model z pamięci (następne użycie załaduje go ponownie)."""
        ...


class MfluxRimEditor:
    """mflux `QwenImage21Edit` (Turbo: 8 kroków, guidance 1), q8, fp16 compute, kafelkowy VAE.

    Wagi ładują się przy pierwszym wywołaniu (~4 s) i zostają w pamięci do `release()`.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        cache_limit_gb: float = 2.0,
        observer: LoadObserver = NO_OBSERVER,
    ):
        self.model_path = model_path
        self.observer = observer
        self.cache_limit_gb = cache_limit_gb
        self._model: Any = None
        self._steps = _StepRelay()

    def _load(self) -> Any:
        if self._model is None:
            with self.observer.loading():
                import mlx.core as mx
                from mflux.models.common.config import ModelConfig
                from mflux.models.common.vae.tiling_config import TilingConfig
                from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

                # Bez limitu cache MLX trzyma zwolnione bufory i realne zużycie rośnie.
                mx.set_cache_limit(int(self.cache_limit_gb * 1024**3))
                model = QwenImage21Edit(
                    model_path=str(self.model_path),
                    model_config=ModelConfig.qwen_image_21_turbo(),
                    compute_precision=mx.float16,
                )
                # Kafelkowy dekoder VAE: szczyt pamięci niższy o 9-14 GB przy 1024 px.
                model.tiling_config = TilingConfig()
                mx.eval(model.parameters())  # ładowanie jest leniwe; wymuszamy je tutaj
                # mflux nie pozwala wyrejestrować callbacku, więc rejestrujemy jeden przekaźnik.
                model.callbacks.register(self._steps)
                self._model = model
        return self._model

    def release(self) -> None:
        if self._model is None:
            return
        import gc

        import mlx.core as mx

        self._model = None
        gc.collect()
        mx.clear_cache()

    def edit(
        self,
        images: list[Image.Image],
        *,
        mask: Image.Image,
        prompt: str,
        seed: int,
        size: tuple[int, int],
        resolution: int,
        on_step: StepCallback | None = None,
    ) -> Image.Image:
        model = self._load()
        self._steps.target = on_step
        try:
            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                image_paths=list(images),
                width=size[0],
                height=size[1],
                output_resolution=resolution,
                mask_image=mask,
            )
        finally:
            self._steps.target = None
        image: Image.Image = result.image
        return image.convert("RGB")


class _StepRelay:
    """Callback mflux (`InLoopCallback`) przekazujący kroki do `on_step` bieżącej edycji.

    mflux woła `call_in_loop` po każdym kroku (`t` = indeks kroku od `init_time_step`)
    i nie łapie wyjątków innych niż `KeyboardInterrupt`, więc wyjątek z `on_step` przerywa
    `generate_image`.
    """

    def __init__(self) -> None:
        self.target: StepCallback | None = None

    def call_in_loop(
        self, t: int, seed: int, prompt: str, latents: Any, config: Any, time_steps: Any
    ) -> None:
        if self.target is None:
            return
        first = config.init_time_step
        self.target(t - first + 1, config.num_inference_steps - first)
