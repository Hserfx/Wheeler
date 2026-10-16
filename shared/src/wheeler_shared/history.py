"""Historia zadań: ręczne usuwanie razem z plikami na dysku.

Retencja jest bezterminowa: nic nie usuwa zadań automatycznie, robi to wyłącznie
użytkownik (pojedynczo lub zbiorczo, z potwierdzeniem w UI).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from wheeler_shared.models import Job, JobStatus
from wheeler_shared.storage import Storage


@dataclass(frozen=True)
class DeletionResult:
    deleted: list[int]
    # Zadania w trakcie przetwarzania: worker pisze do ich katalogu, więc ich nie usuwamy.
    skipped: list[int]


def is_deletable(job: Job) -> bool:
    return job.status != JobStatus.RUNNING


def delete_jobs(
    session_factory: sessionmaker[Session], storage: Storage, job_ids: Iterable[int]
) -> DeletionResult:
    """Usuwa zadania (wiersze w bazie i katalogi zadań na wolumenie danych).

    Kasowany jest cały katalog zadania (`jobs/<id>`), także pliki, których baza nie zna.
    Pliki spoza katalogu zadania (np. felgi z biblioteki) zostają. Zadanie `running` jest
    pomijane; `FOR UPDATE` czeka na transakcję workera, który właśnie je pobiera, i po
    niej ponownie sprawdza warunek, więc nie usuniemy zadania, które przed chwilą ruszyło.
    Nieistniejące id są ignorowane. Pliki kasujemy dopiero po zatwierdzeniu transakcji.
    """
    ids = sorted(set(job_ids))
    if not ids:
        return DeletionResult(deleted=[], skipped=[])

    with session_factory.begin() as session:
        jobs = session.scalars(
            select(Job)
            .where(Job.id.in_(ids), Job.status != JobStatus.RUNNING)
            .order_by(Job.id)
            .with_for_update(of=Job)
            .options(selectinload(Job.images))
        ).all()
        skipped = list(
            session.scalars(
                select(Job.id)
                .where(Job.id.in_(ids), Job.status == JobStatus.RUNNING)
                .order_by(Job.id)
            )
        )
        job_dirs = {
            job_dir
            for job in jobs
            for image in job.images
            if (job_dir := storage.job_dir_of(image.path)) is not None
        }
        deleted = [job.id for job in jobs]
        for job in jobs:
            session.delete(job)

    for job_dir in sorted(job_dirs):
        storage.remove_tree(job_dir)
    return DeletionResult(deleted=deleted, skipped=skipped)
