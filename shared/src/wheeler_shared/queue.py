"""Kolejka zadań w PostgreSQL.

Funkcje nie zatwierdzają transakcji: robi to wywołujący (`with session_factory.begin()`).
`claim_next_job` blokuje wiersz przez `FOR UPDATE SKIP LOCKED`, więc kilka workerów
(albo kilka procesów) nigdy nie dostanie tego samego zadania.

Kolejność: najpierw segmentacja (interaktywna, ~1 s), potem generowanie; w obrębie rodzaju
FIFO po `queued_at` (moment wstawienia do kolejki, nie utworzenia zadania).
Zlecanie pracy z UI (klik, „Generuj”) jest w `editing.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import ColumnElement, case, func, literal, select, tuple_, update
from sqlalchemy.orm import Session

from wheeler_shared.models import Image, ImageRole, Job, JobStatus, TaskKind
from wheeler_shared.progress import Progress


@dataclass(frozen=True)
class ClaimedWheel:
    id: int
    click: tuple[float, float] | None
    mask_path: str | None


@dataclass(frozen=True)
class ClaimedJob:
    """Odłączona od sesji migawka zadania przekazywana do przetwarzania."""

    id: int
    task: TaskKind
    input_path: str
    rim_path: str | None
    params: dict[str, Any]
    wheels: tuple[ClaimedWheel, ...]


def create_job(
    session: Session, *, input_path: str, rim_path: str, params: Mapping[str, Any]
) -> int:
    """Nowe zadanie (szkic) ze zdjęciem auta i felgą (ścieżki względne); zwraca jego id."""
    job = Job(status=JobStatus.DRAFT, params=dict(params))
    job.images.append(Image(role=ImageRole.INPUT, path=input_path))
    job.images.append(Image(role=ImageRole.RIM, path=rim_path))
    session.add(job)
    session.flush()
    return job.id


def enqueue(job: Job, task: TaskKind) -> None:
    """Wstawia zadanie do kolejki z pracą `task` (koniec kolejki, czyści poprzedni błąd)."""
    job.status = JobStatus.QUEUED
    job.task = task
    job.queued_at = _now()
    job.error = None
    job.progress = None
    job.cancel_requested = False


def _generate_last() -> ColumnElement[int]:
    return case((Job.task == TaskKind.GENERATE, 1), else_=0)


def claim_next_job(session: Session) -> ClaimedJob | None:
    """Pobiera następne oczekujące zadanie i oznacza je jako `running`."""
    generate_last = _generate_last()
    job = session.scalars(
        select(Job)
        .where(Job.status == JobStatus.QUEUED, Job.task.is_not(None))
        .order_by(generate_last, Job.queued_at, Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if job is None:
        return None
    assert job.task is not None
    job.status = JobStatus.RUNNING
    job.started_at = _now()
    input_image = job.image(ImageRole.INPUT)
    if input_image is None:
        raise RuntimeError(f"Zadanie {job.id} nie ma obrazu wejściowego")
    rim = job.image(ImageRole.RIM)
    return ClaimedJob(
        id=job.id,
        task=job.task,
        input_path=input_image.path,
        rim_path=rim.path if rim else None,
        params=dict(job.params),
        wheels=tuple(
            ClaimedWheel(
                id=wheel.id,
                click=None
                if wheel.click_x is None or wheel.click_y is None
                else (wheel.click_x, wheel.click_y),
                mask_path=wheel.mask_path,
            )
            for wheel in job.wheels
        ),
    )


def queue_position(session: Session, job: Job) -> int | None:
    """Pozycja zadania w kolejce (1 = następne do pobrania), w kolejności `claim_next_job`."""
    if job.status != JobStatus.QUEUED or job.task is None or job.queued_at is None:
        return None
    mine = (1 if job.task == TaskKind.GENERATE else 0, job.queued_at, job.id)
    ahead = session.scalar(
        select(func.count())
        .select_from(Job)
        .where(
            Job.status == JobStatus.QUEUED,
            Job.task.is_not(None),
            tuple_(_generate_last(), Job.queued_at, Job.id) < tuple_(*map(literal, mine)),
        )
    )
    return int(ahead or 0) + 1


def report_progress(session: Session, job_id: int, progress: Progress) -> bool:
    """Zapisuje postęp pracy workera; zwraca True, jeśli użytkownik poprosił o anulowanie."""
    cancel_requested = session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(progress=progress.to_json())
        .returning(Job.cancel_requested)
    ).scalar_one()
    return bool(cancel_requested)


def complete_segmentation(
    session: Session, job_id: int, *, masks: Mapping[int, tuple[str, str]]
) -> None:
    """Zapisuje maski kół (`id -> (maska, nakładka)`) i oddaje zadanie użytkownikowi."""
    job = session.get_one(Job, job_id)
    for wheel in job.wheels:
        if wheel.id in masks:
            wheel.mask_path, wheel.overlay_path = masks[wheel.id]
    job.status = JobStatus.DRAFT


def complete_job(
    session: Session,
    job_id: int,
    *,
    result_path: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Kończy generowanie: zapisuje wynik i oznacza zadanie jako `done`."""
    job = session.get_one(Job, job_id)
    job.images.append(Image(role=ImageRole.RESULT, path=result_path, width=width, height=height))
    job.status = JobStatus.DONE
    job.finished_at = _now()


def fail_job(session: Session, job_id: int, *, error: str) -> None:
    """Błąd pracy workera. Nieudana segmentacja wraca do szkicu (można kliknąć ponownie),
    nieudane generowanie kończy zadanie statusem `failed`."""
    job = session.get_one(Job, job_id)
    job.error = error
    if job.task == TaskKind.SEGMENT:
        job.status = JobStatus.DRAFT
        for wheel in [w for w in job.wheels if w.mask_path is None]:
            job.wheels.remove(wheel)
        return
    job.status = JobStatus.FAILED
    job.finished_at = _now()


INTERRUPTED_ERROR = "Przerwane: worker został zatrzymany albo uległ awarii w trakcie zadania."


def fail_interrupted_jobs(session: Session) -> list[int]:
    """Przy starcie workera: zadania `running` zostały przerwane przez awarię albo restart.

    Nie wracają do kolejki (bez automatycznych ponowień): generowanie dostaje `failed`
    (albo `cancelled`, jeśli użytkownik zdążył je anulować), wyznaczanie maski wraca do
    szkicu jak po błędzie (`fail_job`). Zakłada jeden proces workera (jedno GPU).
    """
    jobs = session.scalars(
        select(Job).where(Job.status == JobStatus.RUNNING).order_by(Job.id).with_for_update()
    ).all()
    for job in jobs:
        if job.cancel_requested:
            mark_cancelled(session, job.id)
        else:
            fail_job(session, job.id, error=INTERRUPTED_ERROR)
    return [job.id for job in jobs]


def mark_cancelled(session: Session, job_id: int) -> None:
    """Worker przerwał zadanie na prośbę użytkownika (`Job.cancel_requested`)."""
    job = session.get_one(Job, job_id)
    job.status = JobStatus.CANCELLED
    job.finished_at = _now()


def _now() -> datetime:
    return datetime.now(UTC)
