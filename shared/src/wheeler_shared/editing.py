"""Działania użytkownika na zadaniu: detekcja kół, edycja masek kliknięciem, start generowania
i anulowanie.

Każde działanie, które wymaga modelu (SAM 2, mflux), wstawia zadanie do kolejki z właściwym
`TaskKind`; resztę robi worker. Funkcje nie zatwierdzają transakcji (jak `queue.py`).
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session, selectinload

from wheeler_shared import edit_params
from wheeler_shared.db import get_or_raise
from wheeler_shared.models import (
    Image,
    ImageRole,
    Job,
    JobStatus,
    Point,
    TaskKind,
    Wheel,
    WheelSource,
)
from wheeler_shared.queue import enqueue, mark_cancelled
from wheeler_shared.rim_library import use_rim
from wheeler_shared.storage import Storage


class EditingError(Exception):
    """Działanie niedozwolone w obecnym stanie zadania."""


class JobBusy(EditingError):
    """Zadanie nie jest szkicem (worker nad nim pracuje albo jest już zakończone)."""


class NoWheelSelected(EditingError):
    """Generowanie bez żadnej gotowej maski koła."""


class NoRimSelected(EditingError):
    """Generowanie bez wybranej felgi."""


class NotCancellable(EditingError):
    """Anulować można tylko generowanie (albo ponowienie koła) w kolejce lub w trakcie."""


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
    wheel = Wheel(source=WheelSource.CLICK)
    wheel.click = point
    job.wheels.append(wheel)
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


def can_choose_rim(job: Job) -> bool:
    """Felgę wybiera się przed „Generuj”: w szkicu albo w trakcie detekcji/maski z kliknięcia."""
    if job.status == JobStatus.DRAFT:
        return True
    return job.status.busy and job.task is not None and not job.task.generates


def choose_rim(session: Session, storage: Storage, job_id: int, rim_id: int) -> None:
    """Wybór felgi z biblioteki dla szkicu: kopia wyciętej felgi w katalogu zadania
    (`rim_library.use_rim`, `RimNotReady` dla felgi spoza biblioteki) zastępuje poprzednią."""
    job = _lock_draft(session, job_id)
    photo = job.image(ImageRole.INPUT)
    job_dir = storage.job_dir_of(photo.path) if photo is not None else None
    if job_dir is None:
        raise RuntimeError(f"Zadanie {job_id} nie ma katalogu z danymi")
    path = use_rim(session, storage, rim_id, job_dir=job_dir)
    rim = job.image(ImageRole.RIM)
    if rim is None:
        job.images.append(Image(role=ImageRole.RIM, path=path))
    else:
        rim.path, rim.width, rim.height = path, None, None


def start_generation(
    session: Session, job_id: int, *, seed: int, rim_size: int | None = None
) -> None:
    """„Generuj”: zleca podmianę felg we wszystkich kołach z maską (`NoRimSelected` bez felgi).

    `seed` (np. losowy) jest zapisywany w parametrach, chyba że zadanie już go ma.
    `rim_size` (stopień z `edit_params.parse_rim_size`) zastępuje zapisany; `None` zostawia
    parametry bez zmian (brak klucza = „bez zmian”).
    """
    job = _lock_draft(session, job_id)
    if not job.masked_wheels or len(job.masked_wheels) != len(job.wheels):
        raise NoWheelSelected
    if job.image(ImageRole.RIM) is None:
        raise NoRimSelected
    if job.params.get(edit_params.SEED) is None:
        job.params = {**job.params, edit_params.SEED: seed}
    if rim_size is not None:
        job.params = {**job.params, edit_params.RIM_SIZE: rim_size}
    enqueue(job, TaskKind.GENERATE)


def cancel_job(session: Session, job_id: int) -> None:
    """„Anuluj” generowanie. Zadanie w kolejce jest anulowane od razu; zadanie w trakcie
    dostaje prośbę o anulowanie, którą worker sprawdza po każdym kroku modelu i przed kołem
    (`queue.report_progress`), a potem oznacza je jako `cancelled` (`queue.mark_cancelled`).
    """
    job = get_or_raise(session, Job, job_id, lock=True)
    if not is_cancellable(job):
        raise NotCancellable
    if job.status == JobStatus.QUEUED:
        mark_cancelled(session, job.id)
    else:
        job.cancel_requested = True


def is_cancellable(job: Job) -> bool:
    """Anulować można generowanie (albo ponowienie koła) w kolejce lub w trakcie."""
    return job.status.busy and job.task is not None and job.task.generates


def regenerate_wheel(session: Session, job_id: int, wheel_id: int, *, seed: int) -> None:
    """„Ponów to koło”: zleca wygenerowanie jednego koła gotowego wyniku od nowa z innym
    seedem. Worker składa je na bieżącym wyniku, więc reszta obrazu się nie zmienia.

    `seed` (np. losowy) zastępuje `seed + 1`, gdy jest równy obecnemu seedowi koła.
    """
    job = get_or_raise(
        session,
        Job,
        job_id,
        lock=True,
        options=[selectinload(Job.wheels), selectinload(Job.images)],
    )
    wheel = next((w for w in job.masked_wheels if w.id == wheel_id), None)
    if wheel is None:
        raise LookupError(wheel_id)
    if job.status != JobStatus.DONE or job.image(ImageRole.RESULT) is None:
        raise JobBusy
    if seed == wheel_seed(job, wheel):
        seed = (seed + 1) % (edit_params.SEED_MAX + 1)
    wheel.pending_seed = seed
    enqueue(job, TaskKind.REGENERATE_WHEEL)


def wheel_seed(job: Job, wheel: Wheel) -> int | None:
    """Seed, z którym koło jest w bieżącym wyniku (własny po „ponów to koło” albo zadania)."""
    if wheel.seed is not None:
        return wheel.seed
    seed = job.params.get(edit_params.SEED)
    return None if seed is None else int(seed)


def _lock_draft(session: Session, job_id: int) -> Job:
    job = get_or_raise(session, Job, job_id, lock=True, options=[selectinload(Job.wheels)])
    if job.status != JobStatus.DRAFT:
        raise JobBusy
    return job
