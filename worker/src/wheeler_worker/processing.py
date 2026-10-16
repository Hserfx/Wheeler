"""Punkt rozszerzenia: przetwarzanie zadań przez worker.

`Processor` ma po jednej metodzie na rodzaj zadania (`wheeler_shared.models.TaskKind`).
Metody pracują wyłącznie na plikach (wejście w katalogu zadania, wynik do `output_dir`)
i nie znają bazy. Runner (`runner.py`) pobiera zadanie z kolejki, buduje `*Item`, woła
właściwą metodę i zapisuje wynik w bazie.

Procesory wybiera się po nazwie (`WHEELER_PROCESSOR`, `build_processor`):
- `wheel-swap` (domyślny): SAM 2 + mflux (`wheel_swap.WheelSwapProcessor`),
- `placeholder`: te same kroki z atrapami modeli, bez GPU (`placeholder.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wheeler_worker.settings import WorkerSettings


@dataclass(frozen=True)
class SegmentItem:
    """Kliknięcia w koła -> maski. Klucz `clicks`: id koła (`Wheel.id`)."""

    job_id: int
    photo_path: Path  # bezwzględna ścieżka zdjęcia auta
    output_dir: Path  # bezwzględny katalog zadania na wolumenie danych (istnieje)
    params: Mapping[str, Any]
    clicks: Mapping[int, tuple[float, float]]  # punkt w pikselach zdjęcia (po obrocie EXIF)


@dataclass(frozen=True)
class MaskFiles:
    mask: Path  # L 0/255, rozmiar zdjęcia: dokładnie obszar, który generowanie może zmienić
    overlay: Path  # RGBA do wyświetlenia na zdjęciu (przezroczysty poza maską)


@dataclass(frozen=True)
class SegmentResult:
    masks: Mapping[int, MaskFiles]  # id koła -> pliki maski


@dataclass(frozen=True)
class GenerateItem:
    """Podmiana felg we wszystkich kołach (`masks`: id koła -> maska z `SegmentResult`)."""

    job_id: int
    photo_path: Path
    rim_path: Path  # zdjęcie felgi referencyjnej
    output_dir: Path
    params: Mapping[str, Any]  # zawiera `seed` (ustawiany przy starcie generowania)
    masks: Mapping[int, Path]


@dataclass(frozen=True)
class GenerateResult:
    result_path: Path  # PNG w rozdzielczości zdjęcia; musi leżeć na wolumenie danych
    width: int
    height: int


class Processor(Protocol):
    def segment(self, item: SegmentItem, /) -> SegmentResult: ...

    def generate(self, item: GenerateItem, /) -> GenerateResult: ...


def _wheel_swap(settings: WorkerSettings) -> Processor:
    from wheeler_worker.rim_editor import MfluxRimEditor
    from wheeler_worker.segmenter import Sam2Segmenter
    from wheeler_worker.wheel_swap import WheelSwapProcessor

    return WheelSwapProcessor(
        segmenter=Sam2Segmenter(), editor=MfluxRimEditor(model_path=settings.mflux_model_path)
    )


def _placeholder(settings: WorkerSettings) -> Processor:
    from wheeler_worker.placeholder import placeholder_processor

    return placeholder_processor()


PROCESSORS: dict[str, Callable[[WorkerSettings], Processor]] = {
    "wheel-swap": _wheel_swap,
    "placeholder": _placeholder,
}


def build_processor(settings: WorkerSettings) -> Processor:
    try:
        factory = PROCESSORS[settings.processor]
    except KeyError:
        known = ", ".join(sorted(PROCESSORS))
        raise ValueError(f"Nieznany procesor {settings.processor!r}; dostępne: {known}") from None
    return factory(settings)
