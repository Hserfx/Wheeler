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

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from wheeler_shared.models import Image, ImageRole, Job, JobStatus, TaskKind


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


def claim_next_job(session: Session) -> ClaimedJob | None:
    """Pobiera następne oczekujące zadanie i oznacza je jako `running`."""
    generate_last = case((Job.task == TaskKind.GENERATE, 1), else_=0)
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


def _now() -> datetime:
    return datetime.now(UTC)
