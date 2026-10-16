"""„Ponów”: nowe zadanie z tymi samymi danymi wejściowymi i parametrami co nieudane.

Nowe zadanie dostaje własne kopie plików (zdjęcie, felga z katalogu zadania, maski kół),
bo usunięcie oryginału z Historii kasuje cały jego katalog. Pliki spoza katalogu zadania
(np. felga z biblioteki) są wskazywane tą samą ścieżką. Parametry, w tym seed, są
kopiowane bez zmian. Gdy wszystkie koła mają maski, zadanie od razu trafia do kolejki;
inaczej czeka na wybór koła jako szkic.
"""

from __future__ import annotations

import shutil
from pathlib import PurePosixPath

from sqlalchemy.orm import Session, selectinload

from wheeler_shared.editing import EditingError
from wheeler_shared.models import Image, ImageRole, Job, JobStatus, TaskKind, Wheel
from wheeler_shared.queue import enqueue
from wheeler_shared.storage import Storage

RETRYABLE = (JobStatus.FAILED, JobStatus.CANCELLED)


class NotRetryable(EditingError):
    """Ponowić można tylko zadanie zakończone niepowodzeniem albo anulowane."""


def is_retryable(job: Job) -> bool:
    return job.status in RETRYABLE


def retry_job(session: Session, storage: Storage, job_id: int, *, job_dir: str) -> int:
    """Tworzy kopię zadania `job_id` w pustym katalogu `job_dir` (`Storage.new_job_dir`).

    Zwraca id nowego zadania. Nie zatwierdza transakcji; po błędzie wywołujący usuwa `job_dir`.
    """
    job = session.get(Job, job_id, options=[selectinload(Job.images), selectinload(Job.wheels)])
    if job is None:
        raise LookupError(job_id)
    if not is_retryable(job):
        raise NotRetryable

    def copy(relative: str, name: str | None = None) -> str:
        source_dir = storage.job_dir_of(relative)
        if source_dir is None:
            return relative
        target = f"{job_dir}/{name or PurePosixPath(relative).relative_to(source_dir)}"
        storage.path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(storage.path(relative), storage.path(target))
        return target

    retried = Job(status=JobStatus.DRAFT, params=dict(job.params))
    for image in job.images:
        if image.role in (ImageRole.INPUT, ImageRole.RIM):
            retried.images.append(
                Image(
                    role=image.role,
                    path=copy(image.path),
                    width=image.width,
                    height=image.height,
                )
            )
    masked = job.masked_wheels
    for wheel in masked:
        retried.wheels.append(Wheel(click_x=wheel.click_x, click_y=wheel.click_y))
    session.add(retried)
    session.flush()  # id nowych kół: ich pliki leżą w `wheels/<id>/`

    for old, new in zip(masked, retried.wheels, strict=True):
        assert old.mask_path is not None
        folder = f"wheels/{new.id}"
        new.mask_path = copy(old.mask_path, f"{folder}/{PurePosixPath(old.mask_path).name}")
        if old.overlay_path is not None:
            new.overlay_path = copy(
                old.overlay_path, f"{folder}/{PurePosixPath(old.overlay_path).name}"
            )
    if retried.wheels and retried.image(ImageRole.RIM) is not None:
        enqueue(retried, TaskKind.GENERATE)
    session.flush()
    return retried.id
