"""Biblioteka felg: dodanie (usunięcie tła przez worker), akceptacja, wybór i usuwanie.

Cykl życia felgi (`Rim.status`):

    queued --worker: BiRefNet--> review --„Dodaj do biblioteki”--> ready
    review/failed --klik w felgę--> queued (worker: SAM 2 z kliknięcia) --> review
    running --awaria/restart workera--> failed (`fail_interrupted_rims`, bez ponowień)
    review/failed/ready --„Odrzuć”/„Usuń”--> (wiersz i pliki usunięte)

Usuwanie tła działa tylko w natywnym workerze, więc felgi mają własną kolejkę w tabeli
`rims` (`claim_next_rim`, `FOR UPDATE SKIP LOCKED`), obsługiwaną przed zadaniami edycji.
Funkcje (poza `delete_rim`) nie zatwierdzają transakcji, jak `queue.py`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.db import get_or_raise, utcnow
from wheeler_shared.failures import Failure
from wheeler_shared.models import Rim, RimStatus
from wheeler_shared.storage import Storage


class RimError(Exception):
    """Działanie niedozwolone w obecnym stanie felgi."""


class RimBusy(RimError):
    """Worker właśnie wycina tło tej felgi."""


class RimNotReady(RimError):
    """Felga nie jest (jeszcze) w bibliotece, nie można jej użyć."""


@dataclass(frozen=True)
class ClaimedRim:
    """Odłączona od sesji migawka felgi do wycięcia przez worker."""

    id: int
    original_path: str
    click: tuple[float, float] | None


def add_rim(session: Session, *, original_path: str, name: str | None = None) -> int:
    """Nowa felga (oryginał już zapisany); worker usunie tło. Zwraca id."""
    rim = Rim(original_path=original_path, name=(name or "").strip() or None)
    _enqueue(rim)
    session.add(rim)
    session.flush()
    return rim.id


def claim_next_rim(session: Session) -> ClaimedRim | None:
    """Pobiera następną felgę do wycięcia i oznacza ją jako `running`."""
    rim = session.scalars(
        select(Rim)
        .where(Rim.status == RimStatus.QUEUED)
        .order_by(Rim.queued_at, Rim.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if rim is None:
        return None
    rim.status = RimStatus.RUNNING
    click = None if rim.click_x is None or rim.click_y is None else (rim.click_x, rim.click_y)
    return ClaimedRim(id=rim.id, original_path=rim.original_path, click=click)


def complete_rim_cut(session: Session, rim_id: int, *, cut_path: str) -> None:
    rim = session.get(Rim, rim_id)
    if rim is None:  # usunięta w trakcie wycinania (nie powinno się zdarzyć: `RimBusy`)
        return
    rim.cut_path = cut_path
    rim.status = RimStatus.REVIEW


def fail_rim_cut(session: Session, rim_id: int, *, error: Failure) -> None:
    rim = session.get(Rim, rim_id)
    if rim is None:
        return
    rim.error = error
    rim.status = RimStatus.FAILED


def fail_interrupted_rims(session: Session) -> list[int]:
    """Przy starcie workera: felgi `running` zostały przerwane przez awarię albo restart.

    Dostają `failed` (bez automatycznych ponowień, jak zadania w `queue.fail_interrupted_jobs`):
    użytkownik może kliknąć felgę, żeby wyciąć ją ponownie, albo ją odrzucić.
    """
    rims = session.scalars(
        select(Rim).where(Rim.status == RimStatus.RUNNING).order_by(Rim.id).with_for_update()
    ).all()
    for rim in rims:
        rim.error = Failure.INTERRUPTED
        rim.status = RimStatus.FAILED
    return [rim.id for rim in rims]


def accept_rim(session: Session, rim_id: int) -> None:
    """Akceptacja podglądu: felga trafia do biblioteki."""
    rim = get_or_raise(session, Rim, rim_id, lock=True)
    if rim.status != RimStatus.REVIEW:
        raise RimError(rim.status)
    rim.status = RimStatus.READY


def refine_rim(session: Session, rim_id: int, point: tuple[float, float]) -> None:
    """Klik w felgę na oryginale: worker wytnie ją ponownie, tym razem z kliknięcia."""
    rim = get_or_raise(session, Rim, rim_id, lock=True)
    if rim.status not in (RimStatus.REVIEW, RimStatus.FAILED):
        raise RimError(rim.status)
    rim.click_x, rim.click_y = float(point[0]), float(point[1])
    _enqueue(rim)


def delete_rim(session_factory: sessionmaker[Session], storage: Storage, rim_id: int) -> None:
    """Usuwa felgę (z biblioteki albo odrzucony podgląd) razem z jej katalogiem plików.

    Zadania mają własne kopie wyciętej felgi, więc ich historia zostaje nienaruszona.
    Nieistniejąca felga jest ignorowana; wycinaną właśnie przez worker odrzuca `RimBusy`.
    """
    with session_factory.begin() as session:
        rim = session.get(Rim, rim_id, with_for_update=True)
        if rim is None:
            return
        if rim.status == RimStatus.RUNNING:
            raise RimBusy
        rim_dir = storage.rim_dir_of(rim.original_path)
        session.delete(rim)
    if rim_dir is not None:
        storage.remove_tree(rim_dir)


def use_rim(session: Session, storage: Storage, rim_id: int, *, job_dir: str) -> str:
    """Wybór felgi z biblioteki dla zadania: kopiuje wyciętą felgę do katalogu zadania
    (`<job_dir>/rim.png`), liczy użycie i zwraca ścieżkę kopii. `RimNotReady`, gdy felgi
    nie ma w bibliotece. Blokada wiersza wyklucza równoległe usunięcie felgi."""
    rim = session.get(Rim, rim_id, with_for_update=True, populate_existing=True)
    if rim is None or rim.status != RimStatus.READY or rim.cut_path is None:
        raise RimNotReady(rim_id)
    target = f"{job_dir}/rim.png"
    _copy(storage, rim.cut_path, target)
    rim.use_count += 1
    rim.last_used_at = utcnow()
    return target


def pending_rims(session: Session) -> list[Rim]:
    """Felgi, które jeszcze nie są w bibliotece (wycinane, do akceptacji, nieudane)."""
    return list(
        session.scalars(
            select(Rim).where(Rim.status != RimStatus.READY).order_by(Rim.created_at, Rim.id)
        )
    )


def library(session: Session) -> list[Rim]:
    """Felgi w bibliotece, od ostatnio użytej; nowa, jeszcze nieużyta felga liczy się
    od chwili dodania."""
    recency = func.coalesce(Rim.last_used_at, Rim.created_at)
    return list(
        session.scalars(
            select(Rim).where(Rim.status == RimStatus.READY).order_by(recency.desc(), Rim.id.desc())
        )
    )


def _enqueue(rim: Rim) -> None:
    rim.status = RimStatus.QUEUED
    rim.queued_at = utcnow()
    rim.error = None


def _copy(storage: Storage, source: str, target: str) -> None:
    shutil.copyfile(storage.path(source), storage.path(target))
