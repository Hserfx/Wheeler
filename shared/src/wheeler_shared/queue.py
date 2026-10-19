"""Kolejka zadań w PostgreSQL.

Funkcje nie zatwierdzają transakcji: robi to wywołujący (`with session_factory.begin()`).
`claim_next_job` blokuje wiersz przez `FOR UPDATE SKIP LOCKED`, więc kilka workerów
(albo kilka procesów) nigdy nie dostanie tego samego zadania.

Kolejność: najpierw praca interaktywna (detekcja kół, segmentacja; ~1 s), potem generowanie;
w obrębie rodzaju FIFO po `queued_at` (moment wstawienia do kolejki, nie utworzenia zadania).
Zlecanie pracy z UI (klik, „Generuj”) jest w `editing.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, and_, case, func, literal, or_, select, tuple_, update
from sqlalchemy.orm import Session

from wheeler_shared.db import utcnow
from wheeler_shared.failures import Failure
from wheeler_shared.models import (
    GENERATING_TASKS,
    Image,
    ImageRole,
    Job,
    JobStatus,
    Rim,
    RimStatus,
    TaskKind,
    Wheel,
    WheelSource,
)
from wheeler_shared.progress import Progress


@dataclass(frozen=True)
class ClaimedWheel:
    id: int
    click: tuple[float, float] | None
    mask_path: str | None
    pending_seed: int | None = None  # „ponów to koło”: seed nowej wersji tego koła


@dataclass(frozen=True)
class ClaimedJob:
    """Odłączona od sesji migawka zadania przekazywana do przetwarzania."""

    id: int
    task: TaskKind
    input_path: str
    rim_path: str | None
    params: dict[str, Any]
    wheels: tuple[ClaimedWheel, ...]
    result_path: str | None = None  # bieżący wynik (podstawa „ponów to koło”)


@dataclass(frozen=True)
class SavedMask:
    """Maska koła zapisana przez worker (ścieżki względne) i prostokąt, który obejmuje."""

    mask_path: str
    overlay_path: str
    box: tuple[int, int, int, int]  # left, top, right, bottom (prawy/dolny wyłącznie)


def create_job(
    session: Session, *, input_path: str, rim_path: str | None, params: Mapping[str, Any]
) -> int:
    """Nowe zadanie (szkic) ze zdjęciem auta i opcjonalnie felgą (ścieżki względne); zwraca id.

    Felgę można wybrać później, przed „Generuj” (`editing.choose_rim`)."""
    job = Job(status=JobStatus.DRAFT, params=dict(params))
    job.images.append(Image(role=ImageRole.INPUT, path=input_path))
    if rim_path is not None:
        job.images.append(Image(role=ImageRole.RIM, path=rim_path))
    session.add(job)
    session.flush()
    return job.id


def enqueue(job: Job, task: TaskKind) -> None:
    """Wstawia zadanie do kolejki z pracą `task` (koniec kolejki, czyści poprzedni błąd)."""
    job.status = JobStatus.QUEUED
    job.task = task
    job.queued_at = utcnow()
    job.error = None
    job.progress = None
    job.cancel_requested = False


def _generate_last() -> ColumnElement[int]:
    return case((Job.task.in_(GENERATING_TASKS), 1), else_=0)


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
    job.started_at = utcnow()
    input_image = job.image(ImageRole.INPUT)
    if input_image is None:
        raise RuntimeError(f"Zadanie {job.id} nie ma obrazu wejściowego")
    rim = job.image(ImageRole.RIM)
    result = job.image(ImageRole.RESULT)
    return ClaimedJob(
        id=job.id,
        task=job.task,
        input_path=input_image.path,
        rim_path=rim.path if rim else None,
        params=dict(job.params),
        wheels=tuple(
            ClaimedWheel(
                id=wheel.id,
                click=wheel.click,
                mask_path=wheel.mask_path,
                pending_seed=wheel.pending_seed,
            )
            for wheel in job.wheels
        ),
        result_path=result.path if result else None,
    )


def queue_position(session: Session, job: Job) -> int | None:
    """Pozycja zadania w kolejce workera (1 = ruszy następne, nic przed nim nie stoi).

    Liczy wszystko, co worker zrobi wcześniej: trwającą pracę (zadanie albo wycinaną felgę),
    felgi w kolejce (worker bierze je przed zadaniami, `runner.run_once`) i zadania przed nim
    w kolejności `claim_next_job`. Praca interaktywna zlecona później może je jeszcze wyprzedzić.
    """
    if job.status != JobStatus.QUEUED or job.task is None or job.queued_at is None:
        return None
    mine = (1 if job.task.generates else 0, job.queued_at, job.id)
    jobs_ahead = session.scalar(
        select(func.count())
        .select_from(Job)
        .where(
            or_(
                Job.status == JobStatus.RUNNING,
                and_(
                    Job.status == JobStatus.QUEUED,
                    Job.task.is_not(None),
                    tuple_(_generate_last(), Job.queued_at, Job.id) < tuple_(*map(literal, mine)),
                ),
            )
        )
    )
    rims_ahead = session.scalar(
        select(func.count())
        .select_from(Rim)
        .where(Rim.status.in_([RimStatus.QUEUED, RimStatus.RUNNING]))
    )
    return int(jobs_ahead or 0) + int(rims_ahead or 0) + 1


def report_progress(session: Session, job_id: int, progress: Progress) -> bool:
    """Zapisuje postęp pracy workera; zwraca True, jeśli użytkownik poprosił o anulowanie."""
    cancel_requested = session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(progress=progress.to_json())
        .returning(Job.cancel_requested)
    ).scalar_one()
    return bool(cancel_requested)


def complete_segmentation(session: Session, job_id: int, *, masks: Mapping[int, SavedMask]) -> None:
    """Zapisuje maski klikniętych kół (`id koła -> maska`) i oddaje zadanie użytkownikowi."""
    job = session.get_one(Job, job_id)
    for wheel in job.wheels:
        if wheel.id in masks:
            _set_mask(wheel, masks[wheel.id])
    job.status = JobStatus.DRAFT


def complete_detection(session: Session, job_id: int, *, wheels: Sequence[SavedMask]) -> None:
    """Zapisuje wykryte koła (nowe `Wheel` z maskami) i oddaje zadanie użytkownikowi."""
    job = session.get_one(Job, job_id)
    for saved in wheels:
        wheel = Wheel(source=WheelSource.DETECTED)
        _set_mask(wheel, saved)
        job.wheels.append(wheel)
    job.wheels_detected = len(wheels)
    job.status = JobStatus.DRAFT


def _set_mask(wheel: Wheel, saved: SavedMask) -> None:
    wheel.mask_path, wheel.overlay_path = saved.mask_path, saved.overlay_path
    wheel.box = saved.box


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
    job.finished_at = utcnow()


def complete_wheel_regeneration(
    session: Session,
    job_id: int,
    *,
    result_path: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Kończy „ponów to koło”: nowy plik wyniku zastępuje poprzedni, koło dostaje nowy seed."""
    job = session.get_one(Job, job_id)
    result = job.image(ImageRole.RESULT)
    if result is None:
        job.images.append(
            Image(role=ImageRole.RESULT, path=result_path, width=width, height=height)
        )
    else:
        result.path, result.width, result.height = result_path, width, height
    for wheel in job.wheels:
        if wheel.pending_seed is not None:
            wheel.seed, wheel.pending_seed = wheel.pending_seed, None
    job.status = JobStatus.DONE
    job.finished_at = utcnow()


def _back_to_result(job: Job) -> None:
    """Ponowienie koła nie doszło do skutku: zadanie wraca do poprzedniego wyniku."""
    for wheel in job.wheels:
        wheel.pending_seed = None
    job.status = JobStatus.DONE


def fail_job(session: Session, job_id: int, *, error: Failure) -> None:
    """Błąd pracy workera (`error`: kod przyczyny, tekst w UI z `texts.failure_text`).

    Nieudana detekcja albo segmentacja wraca do szkicu (można kliknąć koło), nieudane
    ponowienie koła wraca do poprzedniego wyniku (`done` z błędem), nieudane generowanie
    kończy zadanie statusem `failed`."""
    job = session.get_one(Job, job_id)
    job.error = error
    if job.task == TaskKind.REGENERATE_WHEEL:
        _back_to_result(job)
        return
    if job.task in (TaskKind.SEGMENT, TaskKind.DETECT):
        job.status = JobStatus.DRAFT
        for wheel in [w for w in job.wheels if w.mask_path is None]:
            job.wheels.remove(wheel)
        return
    job.status = JobStatus.FAILED
    job.finished_at = utcnow()


def fail_interrupted_jobs(session: Session) -> list[int]:
    """Przy starcie workera: zadania `running` zostały przerwane przez awarię albo restart.

    Nie wracają do kolejki (bez automatycznych ponowień): generowanie dostaje `failed`
    (albo `cancelled`, jeśli użytkownik zdążył je anulować), detekcja i wyznaczanie maski
    wracają do szkicu jak po błędzie (`fail_job`). Zakłada jeden proces workera (jedno GPU).
    """
    jobs = session.scalars(
        select(Job).where(Job.status == JobStatus.RUNNING).order_by(Job.id).with_for_update()
    ).all()
    for job in jobs:
        if job.cancel_requested:
            mark_cancelled(session, job.id)
        else:
            fail_job(session, job.id, error=Failure.INTERRUPTED)
    return [job.id for job in jobs]


def mark_cancelled(session: Session, job_id: int) -> None:
    """Zadanie anulowane przez użytkownika (w kolejce albo przez worker po `cancel_requested`).

    Anulowane ponowienie koła wraca do poprzedniego wyniku (`done`)."""
    job = session.get_one(Job, job_id)
    if job.task == TaskKind.REGENERATE_WHEEL:
        _back_to_result(job)
        return
    job.status = JobStatus.CANCELLED
    job.finished_at = utcnow()
