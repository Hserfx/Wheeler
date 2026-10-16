"""Działania użytkownika na szkicu zadania: detekcja kół, edycja masek kliknięciem i start
generowania.

Każde działanie, które wymaga modelu (SAM 2, mflux), wstawia zadanie do kolejki z właściwym
`TaskKind`; resztę robi worker. Funkcje nie zatwierdzają transakcji (jak `queue.py`).
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session, selectinload

from wheeler_shared import edit_params
from wheeler_shared.models import Job, JobStatus, TaskKind, Wheel, WheelSource
from wheeler_shared.queue import enqueue


class EditingError(Exception):
    """Działanie niedozwolone w obecnym stanie zadania."""


class JobBusy(EditingError):
    """Zadanie nie jest szkicem (worker nad nim pracuje albo jest już zakończone)."""


class NoWheelSelected(EditingError):
    """Generowanie bez żadnej gotowej maski koła."""


Point = tuple[float, float]
# Czy maska (ścieżka względna pliku L 0/255) obejmuje punkt? Odczyt pliku robi wywołujący (API).
MaskContains = Callable[[str, Point], bool]


def click_photo(
    session: Session, job_id: int, point: Point, *, mask_contains: MaskContains
) -> None:
    """Klik w zdjęcie szkicu: w masce koła usuwa to koło (od razu, bez workera), w pustym
    miejscu dodaje nowe koło i zleca workerowi wyznaczenie jego maski (SAM 2).

    Przy nakładających się maskach usuwane jest koło dodane najpóźniej (rysowane na wierzchu).
    Pliki usuniętych masek zostają w katalogu zadania do jego usunięcia.
    """
    job = _lock_draft(session, job_id)
    hit = next(
        (
            wheel
            for wheel in reversed(job.masked_wheels)
            if _wheel_contains(wheel, point, mask_contains)
        ),
        None,
    )
    if hit is not None:
        job.wheels.remove(hit)
        job.error = None
        return
    job.wheels.append(
        Wheel(source=WheelSource.CLICK, click_x=float(point[0]), click_y=float(point[1]))
    )
    enqueue(job, TaskKind.SEGMENT)


def _wheel_contains(wheel: Wheel, point: Point, mask_contains: MaskContains) -> bool:
    assert wheel.mask_path is not None
    box = wheel.box
    if box is not None:
        left, top, right, bottom = box
        if not (left <= point[0] < right and top <= point[1] < bottom):
            return False
    return mask_contains(wheel.mask_path, point)


def request_detection(session: Session, job_id: int) -> None:
    """Po wgraniu zdjęcia: zleca workerowi wykrycie wszystkich kół (detektor + SAM 2)."""
    enqueue(_lock_draft(session, job_id), TaskKind.DETECT)


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
