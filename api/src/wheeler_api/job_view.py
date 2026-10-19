"""Stan zadania dla szablonów: opis statusu, postęp, dostępne działania (teksty z `texts.py`)."""

from __future__ import annotations

from wheeler_api.texts import STATUS_LABELS, TEXTS
from wheeler_shared.editing import is_cancellable
from wheeler_shared.models import Job, JobStatus, TaskKind
from wheeler_shared.progress import Progress


def job_status_label(job: Job) -> str:
    """Status zadania dla użytkownika; praca workera nad maską ma własny opis."""
    if job.status.busy and job.task == TaskKind.SEGMENT:
        return TEXTS["status_segmenting"]
    if job.status.busy and job.task == TaskKind.DETECT:
        return TEXTS["status_detecting"]
    if job.status.busy and job.cancel_requested:
        return TEXTS["status_cancelling"]
    if job.status.busy and job.task == TaskKind.REGENERATE_WHEEL:
        return TEXTS["status_regenerating"]
    return STATUS_LABELS[job.status]


def job_progress(job: Job) -> Progress | None:
    """Postęp generowania, gdy worker nad nim pracuje."""
    if job.status != JobStatus.RUNNING or job.task is None or not job.task.generates:
        return None
    return Progress.from_json(job.progress)


def progress_label(progress: Progress) -> str:
    if progress.steps is None:
        return TEXTS["progress_wheel"].format(wheel=progress.wheel, wheels=progress.wheels)
    return TEXTS["progress_step"].format(
        wheel=progress.wheel, wheels=progress.wheels, step=progress.step, steps=progress.steps
    )


def can_cancel(job: Job) -> bool:
    """Przycisk „Anuluj”: generowanie w kolejce albo w trakcie, bez wcześniejszej prośby."""
    return is_cancellable(job) and not job.cancel_requested
