"""Działania użytkownika na szkicu zadania: wybór koła kliknięciem i start generowania.

Każde działanie, które wymaga modelu (SAM 2, mflux), wstawia zadanie do kolejki z właściwym
`TaskKind`; resztę robi worker. Funkcje nie zatwierdzają transakcji (jak `queue.py`).
"""

from __future__ import annotations

from sqlalchemy.orm import Session, selectinload

from wheeler_shared import edit_params
from wheeler_shared.models import Job, JobStatus, TaskKind, Wheel
from wheeler_shared.queue import enqueue


class EditingError(Exception):
    """Działanie niedozwolone w obecnym stanie zadania."""


class JobBusy(EditingError):
    """Zadanie nie jest szkicem (worker nad nim pracuje albo jest już zakończone)."""


class NoWheelSelected(EditingError):
    """Generowanie bez żadnej gotowej maski koła."""


def select_wheel(session: Session, job_id: int, point: tuple[float, float]) -> int:
    """Klik w koło: zastępuje wybrane koło nowym i zleca workerowi wyznaczenie maski.

    Ticket 03 obsługuje jedno koło, więc kolejny klik zastępuje poprzedni wybór.
    Zwraca id nowego koła. Pliki starych masek zostają w katalogu zadania.
    """
    job = _lock_draft(session, job_id)
    job.wheels.clear()
    wheel = Wheel(click_x=float(point[0]), click_y=float(point[1]))
    job.wheels.append(wheel)
    enqueue(job, TaskKind.SEGMENT)
    session.flush()
    return wheel.id


def start_generation(session: Session, job_id: int, *, seed: int) -> None:
    """„Generuj”: zleca podmianę felg we wszystkich kołach z maską.

    `seed` (np. losowy) jest zapisywany w parametrach, chyba że zadanie już go ma.
    """
    job = _lock_draft(session, job_id)
    if not job.masked_wheels or len(job.masked_wheels) != len(job.wheels):
        raise NoWheelSelected
    if job.params.get(edit_params.SEED) is None:
        job.params = {**job.params, edit_params.SEED: seed}
    enqueue(job, TaskKind.GENERATE)


def _lock_draft(session: Session, job_id: int) -> Job:
    job = session.get(
        Job,
        job_id,
        with_for_update=True,
        options=[selectinload(Job.wheels)],
        populate_existing=True,
    )
    if job is None:
        raise LookupError(job_id)
    if job.status != JobStatus.DRAFT:
        raise JobBusy
    return job
