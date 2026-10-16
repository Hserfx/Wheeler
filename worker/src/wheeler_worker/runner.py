"""Pętla workera: jedno zadanie naraz, pobierane z kolejki przez `FOR UPDATE SKIP LOCKED`."""

from __future__ import annotations

import logging
import threading

from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.queue import claim_next_job, complete_job, fail_job
from wheeler_shared.storage import Storage
from wheeler_worker.processing import Processor, WorkItem

log = logging.getLogger(__name__)


def run_once(
    session_factory: sessionmaker[Session], storage: Storage, processor: Processor
) -> bool:
    """Przetwarza najstarsze oczekujące zadanie. Zwraca False, gdy kolejka jest pusta."""
    with session_factory.begin() as session:
        job = claim_next_job(session)
    if job is None:
        return False

    log.info("Zadanie %s: start", job.id)
    try:
        input_path = storage.path(job.input_path)
        item = WorkItem(
            job_id=job.id,
            input_path=input_path,
            output_dir=input_path.parent,
            params=job.params,
        )
        result = processor(item)
        result_path = storage.relative(result.result_path)
    except Exception as exc:
        log.exception("Zadanie %s: błąd", job.id)
        with session_factory.begin() as session:
            fail_job(session, job.id, error=f"{type(exc).__name__}: {exc}")
        return True

    with session_factory.begin() as session:
        complete_job(
            session, job.id, result_path=result_path, width=result.width, height=result.height
        )
    log.info("Zadanie %s: gotowe (%s)", job.id, result_path)
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
