"""Punkt rozszerzenia: przetwarzanie pojedynczego zadania.

`Processor` to dowolna funkcja `WorkItem -> WorkResult`. Runner (`runner.py`) nie wie nic
o modelu: pobiera zadanie z kolejki, woła procesor i zapisuje wynik w bazie. Testy podają
własny procesor (atrapę), a aplikacja wybiera go po nazwie przez `build_processor`
(zmienna `WHEELER_PROCESSOR`).

Ticket 03 dopisuje prawdziwy procesor (mflux + SAM 2) do `PROCESSORS` i zmienia domyślny.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageOps


@dataclass(frozen=True)
class WorkItem:
    job_id: int
    input_path: Path  # bezwzględna ścieżka zdjęcia wejściowego
    output_dir: Path  # bezwzględny katalog zadania na wolumenie danych (istnieje)
    params: Mapping[str, Any]


@dataclass(frozen=True)
class WorkResult:
    result_path: Path  # bezwzględna ścieżka wyniku; musi leżeć na wolumenie danych
    width: int
    height: int


class Processor(Protocol):
    def __call__(self, item: WorkItem, /) -> WorkResult: ...


def placeholder_processor(item: WorkItem) -> WorkResult:
    """Atrapa bez AI: obraz w odcieniach szarości, zapisany jako PNG w tej samej rozdzielczości."""
    with Image.open(item.input_path) as source:
        image = ImageOps.exif_transpose(source)
        result = ImageOps.grayscale(image).convert("RGB")
    target = item.output_dir / "result.png"
    result.save(target, format="PNG")
    return WorkResult(result_path=target, width=result.width, height=result.height)


PROCESSORS: dict[str, Callable[[], Processor]] = {
    "placeholder": lambda: placeholder_processor,
}


def build_processor(name: str) -> Processor:
    try:
        factory = PROCESSORS[name]
    except KeyError:
        known = ", ".join(sorted(PROCESSORS))
        raise ValueError(f"Nieznany procesor {name!r}; dostępne: {known}") from None
    return factory()
