"""Parametry edycji zapisywane w `Job.params` i ich wartości domyślne (wynik spike'a 01).

Jedno źródło prawdy dla API (formularz „Zaawansowane”, ticket 09) i workera. Brak klucza
w `Job.params` oznacza wartość domyślną; `seed` jest losowany przy starcie generowania.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Klucze w `Job.params`.
SEED = "seed"
RESOLUTION = "resolution"
MASK_GROW = "mask_grow"
PROMPT_SUFFIX = "prompt_suffix"


@dataclass(frozen=True)
class EditParams:
    seed: int | None = None
    # Rozdzielczość robocza modelu: pole wycinka ≈ resolution², boki wielokrotnością 32 px.
    resolution: int = 1024
    # Poszerzenie maski SAM jako ułamek boku koła (0.03 = 3%).
    mask_grow: float = 0.03
    # Dopisek użytkownika doklejany na końcu szablonu promptu.
    prompt_suffix: str = ""

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> EditParams:
        defaults = cls()
        seed = params.get(SEED)
        return cls(
            seed=None if seed is None else int(seed),
            resolution=int(params.get(RESOLUTION, defaults.resolution)),
            mask_grow=float(params.get(MASK_GROW, defaults.mask_grow)),
            prompt_suffix=str(params.get(PROMPT_SUFFIX, defaults.prompt_suffix)),
        )
