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
from wheeler_shared.progress import Progress
from wheeler_shared.queue import (
    ClaimedJob,
    SavedMask,
    claim_next_job,
    complete_detection,
    complete_job,
    complete_segmentation,
    fail_interrupted_jobs,
    fail_job,
    mark_cancelled,
    report_progress,
)
from wheeler_shared.rim_library import claim_next_rim, complete_rim_cut, fail_rim_cut
from wheeler_shared.storage import Storage
from wheeler_worker.model_lifecycle import ModelLifecycle
from wheeler_worker.processing import (
    CutRimItem,
    DetectItem,
    GenerateItem,
    JobCancelled,
    MaskFiles,
    Processor,
    ProgressCallback,
    SegmentItem,
)

log = logging.getLogger(__name__)

SaveResult = Callable[[Session], None]
# Obsługa dostaje też `report`: zapisuje postęp w bazie i rzuca `JobCancelled`, gdy użytkownik
# anulował zadanie (procesor woła go między krokami, zob. `GenerateItem.on_progress`).
Handler = Callable[[ClaimedJob, Storage, Processor, ProgressCallback], SaveResult]


def _saved(storage: Storage, files: MaskFiles) -> SavedMask:
    return SavedMask(
        mask_path=storage.relative(files.mask),
        overlay_path=storage.relative(files.overlay),
        box=files.box,
    )


def _detect(
    job: ClaimedJob, storage: Storage, processor: Processor, report: ProgressCallback
) -> SaveResult:
    photo = storage.path(job.input_path)
    result = processor.detect(
        DetectItem(job_id=job.id, photo_path=photo, output_dir=photo.parent, params=job.params)
    )
    wheels = [_saved(storage, files) for files in result.wheels]
    return lambda session: complete_detection(session, job.id, wheels=wheels)


def _segment(
    job: ClaimedJob, storage: Storage, processor: Processor, report: ProgressCallback
) -> SaveResult:
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
    masks = {wheel_id: _saved(storage, files) for wheel_id, files in result.masks.items()}
    return lambda session: complete_segmentation(session, job.id, masks=masks)


def _generate(
    job: ClaimedJob, storage: Storage, processor: Processor, report: ProgressCallback
) -> SaveResult:
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
            on_progress=report,
        )
    )
    result_path = storage.relative(result.result_path)
    return lambda session: complete_job(
        session, job.id, result_path=result_path, width=result.width, height=result.height
    )


HANDLERS: dict[TaskKind, Handler] = {
    TaskKind.DETECT: _detect,
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
        save = HANDLERS[job.task](job, storage, processor, _reporter(session_factory, job.id))
    except JobCancelled:
        log.info("Zadanie %s (%s): anulowane", job.id, job.task)
        with session_factory.begin() as session:
            mark_cancelled(session, job.id)
        return True
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


def _reporter(session_factory: sessionmaker[Session], job_id: int) -> ProgressCallback:
    """Postęp zadania do bazy (osobna krótka transakcja na każdy raport, ~raz na krok)."""

    def report(progress: Progress) -> None:
        with session_factory.begin() as session:
            cancel_requested = report_progress(session, job_id, progress)
        if cancel_requested:
            raise JobCancelled

    return report


def recover_interrupted_jobs(session_factory: sessionmaker[Session]) -> list[int]:
    """Wołane przy starcie workera: zadania przerwane przez awarię/restart -> `failed`."""
    with session_factory.begin() as session:
        job_ids = fail_interrupted_jobs(session)
    if job_ids:
        log.warning("Zadania przerwane przez awarię lub restart workera: %s", job_ids)
    return job_ids


def run_forever(
    session_factory: sessionmaker[Session],
    storage: Storage,
    processor: Processor,
    *,
    poll_interval: float,
    stop: threading.Event,
    lifecycle: ModelLifecycle | None = None,
) -> None:
    """Przetwarza zadania aż do ustawienia `stop`. Pusta kolejka → czeka `poll_interval` s.

    Z `lifecycle`: po każdym zadaniu liczy od nowa czas bezczynności, a przy pustej kolejce
    zwalnia modele procesora, gdy minie `lifecycle.idle_timeout`.
    """
    while not stop.is_set():
        try:
            processed = run_once(session_factory, storage, processor)
        except Exception:
            # Np. chwilowy brak bazy; nie kończymy procesu, tylko próbujemy ponownie.
            log.exception("Błąd pętli workera")
            processed = False
        if lifecycle is not None:
            _track_idle(lifecycle, processor, processed=processed)
        if not processed:
            stop.wait(poll_interval)


def _track_idle(lifecycle: ModelLifecycle, processor: Processor, *, processed: bool) -> None:
    if processed:
        lifecycle.mark_used()
        return
    try:
        if lifecycle.release_if_idle(processor.release):
            log.info("Modele zwolnione po %.0f s bezczynności", lifecycle.idle_timeout)
    except Exception:
        log.exception("Błąd zwalniania modeli")
