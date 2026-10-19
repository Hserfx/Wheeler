"""Punkt rozszerzenia: przetwarzanie zadań przez worker.

`Processor` ma po jednej metodzie na rodzaj zadania (`wheeler_shared.models.TaskKind`)
oraz `cut_rim` dla kolejki felg (`wheeler_shared.rim_library`).
Metody pracują wyłącznie na plikach (wejście w katalogu zadania, wynik do `output_dir`)
i nie znają bazy. Runner (`runner.py`) pobiera zadanie z kolejki, buduje `*Item`, woła
właściwą metodę i zapisuje wynik w bazie.

Procesory wybiera się po nazwie (`WHEELER_PROCESSOR`, `build_processor`):
- `wheel-swap` (domyślny): Florence-2 + SAM 2 + mflux + BiRefNet (`wheel_swap.WheelSwapProcessor`),
- `placeholder`: te same kroki z atrapami modeli, bez GPU (`placeholder.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wheeler_shared.progress import Progress
from wheeler_worker.model_lifecycle import NO_OBSERVER, LoadObserver
from wheeler_worker.settings import WorkerSettings

# Odbiorca postępu generowania. Może rzucić wyjątek (np. `JobCancelled`), żeby przerwać pracę:
# procesor go nie łapie, więc generowanie kończy się po bieżącym kroku albo przed kolejnym kołem.
ProgressCallback = Callable[[Progress], None]


class JobCancelled(Exception):
    """Użytkownik anulował zadanie; rzucane przez `ProgressCallback`."""


def _ignore_progress(progress: Progress, /) -> None:
    pass


@dataclass(frozen=True)
class DetectItem:
    """Zdjęcie auta -> maski wszystkich widocznych kół."""

    job_id: int
    photo_path: Path  # bezwzględna ścieżka zdjęcia auta
    output_dir: Path  # bezwzględny katalog zadania na wolumenie danych (istnieje)
    params: Mapping[str, Any]


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
    box: tuple[int, int, int, int]  # prostokąt obejmujący maskę (left, top, right, bottom)


@dataclass(frozen=True)
class DetectResult:
    wheels: Sequence[MaskFiles]  # wykryte koła (pusta lista = nie znaleziono żadnego)


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
    # Wołany przed każdym kołem (krok 0) i po każdym kroku denoisingu.
    on_progress: ProgressCallback = _ignore_progress


@dataclass(frozen=True)
class RegenerateWheelItem:
    """„Ponów to koło”: jedno koło od nowa z nowym seedem, złożone na bieżącym wyniku.

    Piksele poza maską tego koła (w tym pozostałe koła) zostają takie jak w `result_path`.
    """

    job_id: int
    photo_path: Path  # zdjęcie wejściowe (źródło ICC/EXIF wyniku)
    result_path: Path  # bieżący wynik, na którym składamy nowe koło (nie jest nadpisywany)
    rim_path: Path
    output_dir: Path
    params: Mapping[str, Any]  # prompt i rozdzielczość jak przy generowaniu; seed jest w `seed`
    wheel_id: int
    mask_path: Path
    seed: int
    on_progress: ProgressCallback = _ignore_progress


@dataclass(frozen=True)
class GenerateResult:
    result_path: Path  # PNG w rozdzielczości zdjęcia; musi leżeć na wolumenie danych
    width: int
    height: int


@dataclass(frozen=True)
class CutRimItem:
    """Usunięcie tła ze zdjęcia felgi (biblioteka felg, `wheeler_shared.rim_library`)."""

    rim_id: int
    original_path: Path  # bezwzględna ścieżka zdjęcia felgi
    output_dir: Path  # bezwzględny katalog felgi na wolumenie danych (istnieje)
    # Klik w felgę (piksele po obrocie EXIF): wskazuje koło, gdy na zdjęciu jest ich kilka.
    click: tuple[float, float] | None = None


@dataclass(frozen=True)
class CutRimResult:
    cut_path: Path  # RGBA PNG przycięty do felgi, przezroczyste tło; na wolumenie danych


class Processor(Protocol):
    def detect(self, item: DetectItem, /) -> DetectResult: ...

    def segment(self, item: SegmentItem, /) -> SegmentResult: ...

    def generate(self, item: GenerateItem, /) -> GenerateResult: ...

    def regenerate_wheel(self, item: RegenerateWheelItem, /) -> GenerateResult: ...

    def cut_rim(self, item: CutRimItem, /) -> CutRimResult: ...

    def release(self) -> None:
        """Zwalnia wszystkie modele z pamięci (po bezczynności; następne zadanie je załaduje)."""
        ...


def _wheel_swap(settings: WorkerSettings, observer: LoadObserver) -> Processor:
    from wheeler_worker.background import BiRefNetRemover
    from wheeler_worker.detector import FlorenceWheelDetector
    from wheeler_worker.rim_editor import MfluxRimEditor
    from wheeler_worker.segmenter import Sam2Segmenter
    from wheeler_worker.wheel_swap import WheelSwapProcessor

    return WheelSwapProcessor(
        detector=FlorenceWheelDetector(observer=observer),
        segmenter=Sam2Segmenter(observer=observer),
        editor=MfluxRimEditor(model_path=settings.mflux_model_path, observer=observer),
        remover=BiRefNetRemover(observer=observer),
    )


def _placeholder(settings: WorkerSettings, observer: LoadObserver) -> Processor:
    from wheeler_worker.placeholder import placeholder_processor

    return placeholder_processor(observer)


# Fabryka dostaje `observer`: modele zgłaszają przez niego ładowanie (`model_lifecycle.py`).
PROCESSORS: dict[str, Callable[[WorkerSettings, LoadObserver], Processor]] = {
    "wheel-swap": _wheel_swap,
    "placeholder": _placeholder,
}


def build_processor(settings: WorkerSettings, observer: LoadObserver = NO_OBSERVER) -> Processor:
    try:
        factory = PROCESSORS[settings.processor]
    except KeyError:
        known = ", ".join(sorted(PROCESSORS))
        raise ValueError(f"Nieznany procesor {settings.processor!r}; dostępne: {known}") from None
    return factory(settings, observer)
