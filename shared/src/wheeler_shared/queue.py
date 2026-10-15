"""Kolejka zadań w PostgreSQL.

Funkcje nie zatwierdzają transakcji: robi to wywołujący (`with session_factory.begin()`).
`claim_next_job` blokuje wiersz przez `FOR UPDATE SKIP LOCKED`, więc kilka workerów
(albo kilka procesów) nigdy nie dostanie tego samego zadania. Kolejność: FIFO po `id`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from wheeler_shared.models import Image, ImageRole, Job, JobStatus


@dataclass(frozen=True)
class ClaimedJob:
    """Odłączona od sesji migawka zadania przekazywana do przetwarzania."""

    id: int
    input_path: str
    params: dict[str, Any]


def enqueue_job(session: Session, *, input_path: str, params: dict[str, Any]) -> int:
    """Dodaje zadanie ze zdjęciem wejściowym (ścieżka względna) i zwraca jego id."""
    job = Job(status=JobStatus.QUEUED, params=dict(params))
    job.images.append(Image(role=ImageRole.INPUT, path=input_path))
    session.add(job)
    session.flush()
    return job.id


def claim_next_job(session: Session) -> ClaimedJob | None:
    """Pobiera najstarsze oczekujące zadanie i oznacza je jako `running`."""
    job = session.scalars(
        select(Job)
        .where(Job.status == JobStatus.QUEUED)
        .order_by(Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if job is None:
        return None
    job.status = JobStatus.RUNNING
    job.started_at = _now()
    input_image = job.image(ImageRole.INPUT)
    if input_image is None:
        raise RuntimeError(f"Zadanie {job.id} nie ma obrazu wejściowego")
    return ClaimedJob(id=job.id, input_path=input_image.path, params=dict(job.params))


def complete_job(
    session: Session,
    job_id: int,
    *,
    result_path: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    job = session.get_one(Job, job_id)
    job.images.append(Image(role=ImageRole.RESULT, path=result_path, width=width, height=height))
    job.status = JobStatus.DONE
    job.finished_at = _now()


def fail_job(session: Session, job_id: int, *, error: str) -> None:
    job = session.get_one(Job, job_id)
    job.status = JobStatus.FAILED
    job.error = error
    job.finished_at = _now()


def _now() -> datetime:
    return datetime.now(UTC)
