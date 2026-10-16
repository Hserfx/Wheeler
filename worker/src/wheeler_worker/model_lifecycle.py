"""Cykl życia modeli w pamięci: uśpiony -> ładowanie -> gotowy -> (bezczynność) -> uśpiony.

Modele ładują się leniwie, przy pierwszym zadaniu, które ich potrzebuje: adapter modelu
otacza swoje ładowanie `with lifecycle.loading():` (protokół `LoadObserver`). Pętla workera
woła `mark_used()` po każdym zadaniu i `release_if_idle(processor.release)`, gdy kolejka
jest pusta. Każdą zmianę stanu `ModelLifecycle` przekazuje do `report` (w workerze:
`db_reporter`, zapis w bazie dla UI). Czas pochodzi z `clock` (w testach: wstrzykiwany).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.worker_state import ModelState, set_model_state

log = logging.getLogger(__name__)

Clock = Callable[[], float]
Report = Callable[[ModelState], None]


class LoadObserver(Protocol):
    def loading(self) -> AbstractContextManager[None]:
        """Otacza ładowanie wag modelu."""
        ...


class _NoObserver:
    def loading(self) -> AbstractContextManager[None]:
        return nullcontext()


NO_OBSERVER: LoadObserver = _NoObserver()


class ModelLifecycle:
    def __init__(
        self, *, idle_timeout: float, report: Report, clock: Clock = time.monotonic
    ) -> None:
        self.idle_timeout = idle_timeout
        self._report = report
        self._clock = clock
        self._state = ModelState.ASLEEP
        self._last_used = clock()
        report(self._state)

    @property
    def state(self) -> ModelState:
        return self._state

    @contextmanager
    def loading(self) -> Iterator[None]:
        """Otacza ładowanie wag jednego modelu; po sukcesie modele są gotowe."""
        before = self._state
        self._set(ModelState.LOADING)
        try:
            yield
        except BaseException:
            self._set(before)
            raise
        self._last_used = self._clock()
        self._set(ModelState.READY)

    def mark_used(self) -> None:
        """Koniec zadania: od teraz liczy się czas bezczynności."""
        self._last_used = self._clock()

    def release_if_idle(self, release: Callable[[], None]) -> bool:
        """Zwalnia modele (`release`), jeśli od ostatniego zadania minął `idle_timeout`."""
        if self._state is not ModelState.READY:
            return False
        if self._clock() - self._last_used < self.idle_timeout:
            return False
        self.release(release)
        return True

    def release(self, release: Callable[[], None]) -> None:
        """Zwalnia modele od razu (np. przy zatrzymaniu workera)."""
        release()
        self._set(ModelState.ASLEEP)

    def _set(self, state: ModelState) -> None:
        if state is not self._state:
            self._state = state
            self._report(state)


def db_reporter(session_factory: sessionmaker[Session]) -> Report:
    """Zapisuje stan modelu w bazie (dla UI). Błąd zapisu nie przerywa pracy workera."""

    def report(state: ModelState) -> None:
        try:
            with session_factory.begin() as session:
                set_model_state(session, state)
        except Exception:
            log.exception("Nie udało się zapisać stanu modelu (%s)", state)
        else:
            log.info("Stan modelu: %s", state)

    return report
