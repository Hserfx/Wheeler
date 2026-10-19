"""Strony i fragmenty HTMX."""

from __future__ import annotations

import secrets
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session, object_session, selectinload

from wheeler_api import advanced
from wheeler_api.deps import SessionDep, SessionFactoryDep, StorageDep, TemplatesDep
from wheeler_api.rim_picker import picker_context
from wheeler_api.texts import TEXTS
from wheeler_api.thumbnails import ensure_thumbnail
from wheeler_api.uploads import UploadRejected, save_image_upload
from wheeler_api.wheel_masks import mask_contains
from wheeler_shared import edit_params
from wheeler_shared.db import get_or_raise
from wheeler_shared.editing import (
    EditingError,
    JobBusy,
    NoRimSelected,
    NotCancellable,
    NoWheelSelected,
    can_choose_rim,
    cancel_job,
    choose_rim,
    click_photo,
    regenerate_wheel,
    request_detection,
    start_generation,
    wheel_seed,
)
from wheeler_shared.history import delete_jobs, is_deletable
from wheeler_shared.models import ImageRole, Job
from wheeler_shared.queue import create_job as create_draft_job
from wheeler_shared.queue import queue_position
from wheeler_shared.retry import NotRetryable, retry_job
from wheeler_shared.rim_library import RimNotReady, use_rim

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request, session: SessionDep, templates: TemplatesDep) -> Response:
    return _home_page(request, session, templates)


def _home_page(
    request: Request,
    session: Session,
    templates: Jinja2Templates,
    *,
    error: str | None = None,
    values: dict[str, str] | None = None,
    advanced_error: bool = False,
    status_code: int = 200,
) -> Response:
    """Strona główna (krok 1: zdjęcie auta); po błędzie pola „Zaawansowane” zostają."""
    context = {
        "error": error,
        "advanced": values or advanced.default_values(),
        "advanced_open": advanced_error,
        "limits": {
            "seed_max": edit_params.SEED_MAX,
            "resolution_min": edit_params.RESOLUTION_MIN,
            "resolution_max": edit_params.RESOLUTION_MAX,
            "resolution_step": edit_params.RESOLUTION_STEP,
            "mask_grow_min": advanced.percent(edit_params.MASK_GROW_MIN),
            "mask_grow_max": advanced.percent(edit_params.MASK_GROW_MAX),
            "prompt_suffix_max": edit_params.PROMPT_SUFFIX_MAX_LENGTH,
        },
    }
    return templates.TemplateResponse(request, "index.html", context, status_code=status_code)


@router.post("/jobs")
def create_job(
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    templates: TemplatesDep,
    photo: UploadFile | None = None,
    rim_id: Annotated[int | None, Form()] = None,
    seed: Annotated[str, Form()] = "",
    resolution: Annotated[str, Form()] = "",
    mask_grow: Annotated[str, Form()] = "",
    prompt_suffix: Annotated[str, Form()] = "",
) -> Response:
    """Krok 1: zdjęcie auta i parametry; detekcja kół rusza od razu. Felgę wybiera się na
    stronie zadania przed „Generuj”; `rim_id` pozwala wskazać ją już tutaj (opcjonalnie)."""
    values = {
        edit_params.SEED: seed,
        edit_params.RESOLUTION: resolution,
        edit_params.MASK_GROW: mask_grow,
        edit_params.PROMPT_SUFFIX: prompt_suffix,
    }
    entered = values if any(values.values()) else None
    if photo is None or not photo.filename:
        return _home_page(
            request,
            session,
            templates,
            error=TEXTS["error_no_photo"],
            values=entered,
            status_code=400,
        )
    try:
        params = advanced.parse(values)
    except advanced.InvalidParam as exc:
        return _home_page(
            request,
            session,
            templates,
            error=advanced.error_message(exc),
            values=values,
            advanced_error=True,
            status_code=400,
        )

    job_dir = storage.new_job_dir()
    try:
        input_path = save_image_upload(storage, photo, f"{job_dir}/input")
        with session.begin():
            # Zadanie dostaje własną kopię wyciętej felgi (usunięcie z biblioteki jej nie rusza).
            rim_path = (
                None if rim_id is None else use_rim(session, storage, rim_id, job_dir=job_dir)
            )
            job_id = create_draft_job(
                session, input_path=input_path, rim_path=rim_path, params=params
            )
            request_detection(session, job_id)
    except (UploadRejected, RimNotReady) as exc:
        storage.remove_tree(job_dir)
        message = exc.message if isinstance(exc, UploadRejected) else TEXTS["error_no_rim"]
        return _home_page(
            request, session, templates, error=message, values=entered, status_code=400
        )
    except BaseException:
        storage.remove_tree(job_dir)
        raise
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/wheels", response_class=HTMLResponse)
def click_wheel(
    job_id: int,
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    templates: TemplatesDep,
    x: Annotated[float, Form()],
    y: Annotated[float, Form()],
) -> Response:
    """Klik w zdjęcie (piksele zdjęcia): w masce usuwa koło, obok masek dodaje koło (SAM 2)."""
    status_code = 200
    try:
        with session.begin():
            click_photo(session, job_id, (x, y), mask_contains=mask_contains(storage))
    except LookupError:
        raise HTTPException(status_code=404) from None
    except JobBusy:
        status_code = 409
    job = _load_job(session, job_id)
    return templates.TemplateResponse(
        request, "_job_status.html", _job_context(job), status_code=status_code
    )


@router.post("/jobs/{job_id}/generate")
def generate(
    job_id: int,
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    templates: TemplatesDep,
    rim_id: Annotated[int | None, Form()] = None,
) -> Response:
    """„Generuj” z felgą wybraną na stronie zadania (`rim_id`) albo już przypisaną."""
    try:
        with session.begin():
            if rim_id is not None:
                choose_rim(session, storage, job_id, rim_id)
            start_generation(session, job_id, seed=_random_seed())
    except LookupError:
        raise HTTPException(status_code=404) from None
    except (NoRimSelected, RimNotReady):
        # Strona zadania z komunikatem: wybór felgi i kół zostaje na miejscu.
        job = _load_job(session, job_id)
        return _job_page(request, session, templates, job, TEXTS["error_no_rim"], 409)
    except EditingError as exc:
        message = TEXTS["error_no_wheel" if isinstance(exc, NoWheelSelected) else "error_job_busy"]
        return templates.TemplateResponse(
            request, "error.html", {"message": message}, status_code=409
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/wheels/{wheel_id}/regenerate")
def regenerate(
    job_id: int, wheel_id: int, request: Request, session: SessionDep, templates: TemplatesDep
) -> Response:
    """„Ponów to koło”: jedno koło gotowego wyniku od nowa z nowym (losowym) seedem."""
    try:
        with session.begin():
            regenerate_wheel(session, job_id, wheel_id, seed=_random_seed())
    except LookupError:
        raise HTTPException(status_code=404) from None
    except JobBusy:
        return templates.TemplateResponse(
            request, "error.html", {"message": TEXTS["error_not_regeneratable"]}, status_code=409
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
def cancel(job_id: int, request: Request, session: SessionDep, templates: TemplatesDep) -> Response:
    """„Anuluj” (HTMX): zwraca odświeżony stan zadania; 409, gdy nie ma czego anulować."""
    status_code = 200
    try:
        with session.begin():
            cancel_job(session, job_id)
    except LookupError:
        raise HTTPException(status_code=404) from None
    except NotCancellable:
        status_code = 409
    job = _load_job(session, job_id)
    return templates.TemplateResponse(
        request, "_job_status.html", _job_context(job), status_code=status_code
    )


@router.post("/jobs/{job_id}/retry")
def retry(
    job_id: int,
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    templates: TemplatesDep,
) -> Response:
    """„Ponów”: nowe zadanie z tymi samymi plikami, kołami i parametrami."""
    job_dir = storage.new_job_dir()
    try:
        with session.begin():
            new_id = retry_job(session, storage, job_id, job_dir=job_dir)
    except BaseException as exc:
        storage.remove_tree(job_dir)
        if isinstance(exc, LookupError):
            raise HTTPException(status_code=404) from None
        if isinstance(exc, NotRetryable):
            return templates.TemplateResponse(
                request, "error.html", {"message": TEXTS["error_not_retryable"]}, status_code=409
            )
        raise
    return RedirectResponse(f"/jobs/{new_id}", status_code=303)


def _random_seed() -> int:
    """Losowy seed z dozwolonego zakresu (0..`SEED_MAX`), gdy użytkownik go nie podał."""
    return secrets.randbelow(edit_params.SEED_MAX + 1)


def _load_job(session: Session, job_id: int) -> Job:
    try:
        return get_or_raise(
            session, Job, job_id, options=[selectinload(Job.images), selectinload(Job.wheels)]
        )
    except LookupError:
        raise HTTPException(status_code=404) from None


def _job_context(job: Job) -> dict[str, object]:
    session = object_session(job)
    return {
        "job": job,
        "queue_position": queue_position(session, job) if session is not None else None,
        "rim_image": job.image(ImageRole.RIM),
        "input_image": job.image(ImageRole.INPUT),
        "result_image": job.image(ImageRole.RESULT),
        "wheel_seed": wheel_seed,
    }


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(
    job_id: int, request: Request, session: SessionDep, templates: TemplatesDep
) -> Response:
    return _job_page(request, session, templates, _load_job(session, job_id))


def _job_page(
    request: Request,
    session: Session,
    templates: Jinja2Templates,
    job: Job,
    error: str | None = None,
    status_code: int = 200,
) -> Response:
    """Strona zadania; przed „Generuj” z wyborem felgi (krok 3, `editing.can_choose_rim`)."""
    context = _job_context(job)
    if can_choose_rim(job):
        context.update(picker_context(session, mode="select"))
    context["page_error"] = error
    return templates.TemplateResponse(request, "job.html", context, status_code=status_code)


@router.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status(
    job_id: int, request: Request, session: SessionDep, templates: TemplatesDep
) -> Response:
    job = _load_job(session, job_id)
    return templates.TemplateResponse(request, "_job_status.html", _job_context(job))


@router.get("/jobs/{job_id}/thumbnail")
def job_thumbnail(job_id: int, session: SessionDep, storage: StorageDep) -> Response:
    job = _load_job(session, job_id)
    image = job.image(ImageRole.RESULT) or job.image(ImageRole.INPUT)
    if image is None or not storage.path(image.path).is_file():
        raise HTTPException(status_code=404)
    try:
        thumbnail = ensure_thumbnail(storage.path(image.path))
    except OSError:
        # Np. format, którego Pillow nie czyta: przeglądarka spróbuje wyświetlić oryginał.
        return RedirectResponse(f"/files/{image.path}", status_code=307)
    return FileResponse(thumbnail, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/jobs/{job_id}/download")
def download_result(job_id: int, session: SessionDep, storage: StorageDep) -> Response:
    """Wynik jako załącznik PNG (rozdzielczość oryginału, z profilem ICC i EXIF bez GPS)."""
    job = _load_job(session, job_id)
    image = job.image(ImageRole.RESULT)
    if image is None or not storage.path(image.path).is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        storage.path(image.path),
        media_type="image/png",
        filename=f"wheeler-{job.id}.png",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/history", response_class=HTMLResponse)
def history(
    request: Request,
    session: SessionDep,
    templates: TemplatesDep,
    deleted: int | None = None,
    skipped: int | None = None,
) -> Response:
    jobs = session.scalars(select(Job).order_by(Job.created_at.desc(), Job.id.desc())).all()
    return templates.TemplateResponse(
        request,
        "history.html",
        {"jobs": jobs, "deleted": deleted, "skipped": skipped},
    )


@router.get("/history/delete", response_class=HTMLResponse)
def confirm_delete(
    request: Request,
    session: SessionDep,
    templates: TemplatesDep,
    ids: Annotated[list[int], Query()] = [],  # noqa: B006 (FastAPI kopiuje wartość domyślną)
) -> Response:
    jobs = session.scalars(
        select(Job).where(Job.id.in_(ids)).order_by(Job.created_at.desc(), Job.id.desc())
    ).all()
    if not jobs:
        return RedirectResponse("/history", status_code=303)
    return templates.TemplateResponse(
        request,
        "delete_confirm.html",
        {
            "deletable": [job for job in jobs if is_deletable(job)],
            "running": [job for job in jobs if not is_deletable(job)],
        },
    )


@router.post("/history/delete")
def delete(
    session_factory: SessionFactoryDep,
    storage: StorageDep,
    ids: Annotated[list[int], Form()] = [],  # noqa: B006
) -> Response:
    result = delete_jobs(session_factory, storage, ids)
    query = urlencode({"deleted": len(result.deleted), "skipped": len(result.skipped)})
    return RedirectResponse(f"/history?{query}", status_code=303)


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
