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
RIM_SIZE = "rim_size"

# Dozwolone zakresy (sekcja „Zaawansowane”, ticket 09; wyniki spike'a 01).
SEED_MAX = 2**31 - 1
# Boki wycinka są wielokrotnością 32 px. Poniżej ~1024 px Turbo traci detal (siatka pikseli);
# spike 01 sprawdził 992 i 1056 px bez różnic względem 1024, więc 992 px to dolna granica
# (DESIGN.md, „Pipeline edycji”). 1536 px przy q8 przekracza 32 GB (swap,
# 35,7 GB); 1024 px ma szczyt 26,9 GB, a pamięć rośnie z polem wycinka, więc górna granica
# 1152 px (pole 1,27 razy większe, szacunkowo ~29 GB) zostawia zapas przed swapem.
RESOLUTION_STEP = 32
RESOLUTION_MIN = 992
RESOLUTION_MAX = 1152
# Poszerzenie maski: 0 = dokładnie maska SAM; powyżej ~10% boku koła wycinek obejmuje już
# spory kawałek nadkola i karoserii, które model wygeneruje na nowo.
MASK_GROW_MIN = 0.0
MASK_GROW_MAX = 0.10
PROMPT_SUFFIX_MAX_LENGTH = 300
# Rozmiar felgi względem obecnej w calach (plus-sizing, ticket 12), wybierany przed „Generuj”.
RIM_SIZE_MIN = -2
RIM_SIZE_MAX = 2


@dataclass(frozen=True)
class EditParams:
    seed: int | None = None
    # Rozdzielczość robocza modelu: pole wycinka ≈ resolution², boki wielokrotnością 32 px.
    resolution: int = 1024
    # Poszerzenie maski SAM jako ułamek boku koła (0.03 = 3%).
    mask_grow: float = 0.03
    # Dopisek użytkownika doklejany na końcu szablonu promptu.
    prompt_suffix: str = ""
    # Średnica nowej felgi względem obecnej w calach (0 = bez zmian); zewnętrzna średnica koła
    # zostaje, więc zmienia się profil opony.
    rim_size: int = 0

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> EditParams:
        defaults = cls()
        seed = params.get(SEED)
        return cls(
            seed=None if seed is None else int(seed),
            resolution=int(params.get(RESOLUTION, defaults.resolution)),
            mask_grow=float(params.get(MASK_GROW, defaults.mask_grow)),
            prompt_suffix=str(params.get(PROMPT_SUFFIX, defaults.prompt_suffix)),
            rim_size=int(params.get(RIM_SIZE, defaults.rim_size)),
        )


def parse_rim_size(text: str) -> int:
    """Stopień rozmiaru felgi z formularza („-2”…„2”, „+1”); spoza zakresu: `ValueError`."""
    rim_size = int(text.strip())
    if not RIM_SIZE_MIN <= rim_size <= RIM_SIZE_MAX:
        raise ValueError(text)
    return rim_size
