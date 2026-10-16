"""Sekcja „Zaawansowane” formularza na stronie głównej: pola tekstowe <-> `Job.params`.

Poszerzenie maski użytkownik podaje w procentach boku koła, w `Job.params` jest ułamkiem
(jak w `EditParams`). Wartości domyślne i zakresy pochodzą z `wheeler_shared.edit_params`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wheeler_api.texts import TEXTS
from wheeler_shared import edit_params
from wheeler_shared.edit_params import EditParams


class InvalidParam(ValueError):
    """Wartość pola spoza dozwolonego zakresu; `field` wskazuje pole formularza."""

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


def percent(fraction: float) -> str:
    """Ułamek jako procent bez zbędnych zer (0.03 -> „3”, 0.025 -> „2.5”)."""
    return f"{round(fraction * 100, 6):g}"


def error_message(error: InvalidParam) -> str:
    """Komunikat dla użytkownika z dozwolonym zakresem pola."""
    return TEXTS[f"error_param_{error.field}"].format(
        seed_max=edit_params.SEED_MAX,
        min=edit_params.RESOLUTION_MIN,
        max=edit_params.RESOLUTION_MAX,
        step=edit_params.RESOLUTION_STEP,
        grow_min=percent(edit_params.MASK_GROW_MIN),
        grow_max=percent(edit_params.MASK_GROW_MAX),
        length=edit_params.PROMPT_SUFFIX_MAX_LENGTH,
    )


def display_value(key: str, value: object) -> str | None:
    """Wartość parametru w szczegółach zadania z jednostką; `None` = bez specjalnego formatu."""
    if key == edit_params.RESOLUTION and isinstance(value, int):
        return f"{value} px"
    if key == edit_params.MASK_GROW and isinstance(value, int | float):
        return f"{percent(value)}%"
    if key == edit_params.PROMPT_SUFFIX and value == "":
        return TEXTS["param_empty"]
    return None


def default_values() -> dict[str, str]:
    """Wartości pól sekcji przy pierwszym wyświetleniu (pusty seed = losowy)."""
    defaults = EditParams()
    return {
        edit_params.SEED: "",
        edit_params.RESOLUTION: str(defaults.resolution),
        edit_params.MASK_GROW: percent(defaults.mask_grow),
        edit_params.PROMPT_SUFFIX: defaults.prompt_suffix,
    }


def parse(values: Mapping[str, str]) -> dict[str, Any]:
    """Pola sekcji -> `Job.params` z użytymi wartościami (pusty seed zostaje do wylosowania).

    Puste pole (albo brak pola) oznacza wartość domyślną. Zły zakres: `InvalidParam`.
    """
    params: dict[str, Any] = {
        edit_params.RESOLUTION: _resolution(values.get(edit_params.RESOLUTION, "")),
        edit_params.MASK_GROW: _mask_grow(values.get(edit_params.MASK_GROW, "")),
        edit_params.PROMPT_SUFFIX: _prompt_suffix(values.get(edit_params.PROMPT_SUFFIX, "")),
    }
    seed = _seed(values.get(edit_params.SEED, ""))
    if seed is not None:
        params[edit_params.SEED] = seed
    return params


def _integer(text: str, field: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise InvalidParam(field) from None


def _seed(text: str) -> int | None:
    if not text.strip():
        return None
    seed = _integer(text, edit_params.SEED)
    if not 0 <= seed <= edit_params.SEED_MAX:
        raise InvalidParam(edit_params.SEED)
    return seed


def _resolution(text: str) -> int:
    if not text.strip():
        return EditParams().resolution
    resolution = _integer(text, edit_params.RESOLUTION)
    if (
        not edit_params.RESOLUTION_MIN <= resolution <= edit_params.RESOLUTION_MAX
        or resolution % edit_params.RESOLUTION_STEP
    ):
        raise InvalidParam(edit_params.RESOLUTION)
    return resolution


def _mask_grow(text: str) -> float:
    if not text.strip():
        return EditParams().mask_grow
    try:
        # Przecinek dziesiętny też jest poprawny („2,5”).
        fraction = round(float(text.strip().replace(",", ".")) / 100, 6)
    except ValueError:
        raise InvalidParam(edit_params.MASK_GROW) from None
    if not edit_params.MASK_GROW_MIN <= fraction <= edit_params.MASK_GROW_MAX:
        raise InvalidParam(edit_params.MASK_GROW)
    return fraction


def _prompt_suffix(text: str) -> str:
    suffix = " ".join(text.split())
    if len(suffix) > edit_params.PROMPT_SUFFIX_MAX_LENGTH:
        raise InvalidParam(edit_params.PROMPT_SUFFIX)
    return suffix
