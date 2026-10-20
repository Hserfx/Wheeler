"""Przyczyny niepowodzenia pracy workera, zapisywane w `Job.error` i `Rim.error` jako kody.

W bazie leży kod (`Failure`), a tekst dla użytkownika powstaje dopiero przy renderowaniu
(`wheeler_api.texts.failure_text`), więc wszystkie komunikaty są w `texts.py`. Szczegóły
techniczne (wyjątki bibliotek) trafiają wyłącznie do logu workera.
"""

from __future__ import annotations

import enum


class Failure(enum.StrEnum):
    INTERNAL = "internal"  # nieoczekiwany wyjątek (np. modelu); szczegóły w logu workera
    INTERRUPTED = "interrupted"  # awaria albo restart workera w trakcie pracy
    LEGACY_JOB = "legacy_job"  # zadanie z wersji bez felgi i masek (migracja 0002)
    NOTHING_AT_CLICK = "nothing_at_click"  # SAM 2 nic nie znalazł w klikniętym miejscu
    EMPTY_MASK = "empty_mask"  # maska koła do generowania jest pusta
    RIM_NOT_FOUND = "rim_not_found"  # na zdjęciu felgi nie znaleziono felgi
    NO_RIM = "no_rim"  # generowanie bez wybranej felgi
    # Zmiana rozmiaru felgi (tickety 12, 15): SAM 2 nie wyznaczył felgi na którymś kole, maska
    # karoserii przy nim jest zła albo koło zasłania coś obcego (`body_mask`).
    RIM_SIZE_UNAVAILABLE = "rim_size_unavailable"
    # Tryb pikseli, którego wynik nie odda bit w bit (`photo_modes`; wgranie go odrzuca).
    UNSUPPORTED_PHOTO = "unsupported_photo"


class ProcessingFailed(Exception):
    """Przewidziane niepowodzenie pracy workera; runner zapisuje `failure` w bazie.

    `detail` jest tylko dla logu (po angielsku albo polsku, dowolnie techniczny).
    """

    def __init__(self, failure: Failure, detail: str = "") -> None:
        super().__init__(detail or failure.value)
        self.failure = failure
