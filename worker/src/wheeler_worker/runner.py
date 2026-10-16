"""Pętla workera: jedno zadanie naraz, pobierane z kolejki przez `FOR UPDATE SKIP LOCKED`.

Runner nie wie nic o modelu. Dla rodzaju pracy (`TaskKind`) wybiera obsługę z `HANDLERS`:
obsługa buduje `*Item` z migawki zadania, woła metodę procesora (poza transakcją, może trwać
minuty) i zwraca funkcję, która zapisze wynik w bazie w osobnej, krótkiej transakcji.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.models import TaskKind
from wheeler_shared.queue import (
    ClaimedJob,
    claim_next_job,
    complete_job,
    complete_segmentation,
    fail_job,
)
from wheeler_shared.rim_library import claim_next_rim, complete_rim_cut, fail_rim_cut
from wheeler_shared.storage import Storage
from wheeler_worker.processing import CutRimItem, GenerateItem, Processor, SegmentItem

log = logging.getLogger(__name__)

SaveResult = Callable[[Session], None]
Handler = Callable[[ClaimedJob, Storage, Processor], SaveResult]


def _segment(job: ClaimedJob, storage: Storage, processor: Processor) -> SaveResult:
    photo = storage.path(job.input_path)
    clicks = {wheel.id: wheel.click for wheel in job.wheels if wheel.click and not wheel.mask_path}
    result = processor.segment(
        SegmentItem(
            job_id=job.id,
            photo_path=photo,
            output_dir=photo.parent,
            params=job.params,
            clicks=clicks,
        )
    )
    masks = {
        wheel_id: (storage.relative(files.mask), storage.relative(files.overlay))
        for wheel_id, files in result.masks.items()
    }
    return lambda session: complete_segmentation(session, job.id, masks=masks)


def _generate(job: ClaimedJob, storage: Storage, processor: Processor) -> SaveResult:
    if job.rim_path is None:
        raise ValueError("Zadanie nie ma felgi referencyjnej")
    photo = storage.path(job.input_path)
    masks = {wheel.id: storage.path(wheel.mask_path) for wheel in job.wheels if wheel.mask_path}
    result = processor.generate(
        GenerateItem(
            job_id=job.id,
            photo_path=photo,
            rim_path=storage.path(job.rim_path),
            output_dir=photo.parent,
            params=job.params,
            masks=masks,
        )
    )
    result_path = storage.relative(result.result_path)
    return lambda session: complete_job(
        session, job.id, result_path=result_path, width=result.width, height=result.height
    )


HANDLERS: dict[TaskKind, Handler] = {
    TaskKind.SEGMENT: _segment,
    TaskKind.GENERATE: _generate,
}


def run_once(
    session_factory: sessionmaker[Session], storage: Storage, processor: Processor
) -> bool:
    """Przetwarza następne oczekujące zadanie. Zwraca False, gdy kolejka jest pusta.

    Felgi do wycięcia (biblioteka, ~1 s, użytkownik czeka na podgląd) idą przed zadaniami.
    """
    if _cut_next_rim(session_factory, storage, processor):
        return True
    with session_factory.begin() as session:
        job = claim_next_job(session)
    if job is None:
        return False

    log.info("Zadanie %s (%s): start", job.id, job.task)
    try:
        save = HANDLERS[job.task](job, storage, processor)
    except Exception as exc:
        log.exception("Zadanie %s (%s): błąd", job.id, job.task)
        with session_factory.begin() as session:
            fail_job(session, job.id, error=f"{type(exc).__name__}: {exc}")
        return True

    with session_factory.begin() as session:
        save(session)
    log.info("Zadanie %s (%s): gotowe", job.id, job.task)
    return True


def _cut_next_rim(
    session_factory: sessionmaker[Session], storage: Storage, processor: Processor
) -> bool:
    with session_factory.begin() as session:
        rim = claim_next_rim(session)
    if rim is None:
        return False

    log.info("Felga %s: usuwanie tła", rim.id)
    original = storage.path(rim.original_path)
    try:
        result = processor.cut_rim(
            CutRimItem(
                rim_id=rim.id, original_path=original, output_dir=original.parent, click=rim.click
            )
        )
        cut_path = storage.relative(result.cut_path)
    except Exception as exc:
        log.exception("Felga %s: błąd", rim.id)
        with session_factory.begin() as session:
            fail_rim_cut(session, rim.id, error=f"{type(exc).__name__}: {exc}")
        return True

    with session_factory.begin() as session:
        complete_rim_cut(session, rim.id, cut_path=cut_path)
    return True


def run_forever(
    session_factory: sessionmaker[Session],
    storage: Storage,
    processor: Processor,
    *,
    poll_interval: float,
    stop: threading.Event,
) -> None:
    """Przetwarza zadania aż do ustawienia `stop`. Pusta kolejka → czeka `poll_interval` s."""
    while not stop.is_set():
        try:
            processed = run_once(session_factory, storage, processor)
        except Exception:
            # Np. chwilowy brak bazy; nie kończymy procesu, tylko próbujemy ponownie.
            log.exception("Błąd pętli workera")
            processed = False
        if not processed:
            stop.wait(poll_interval)
