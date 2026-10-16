"""Strony i fragmenty HTMX."""

from __future__ import annotations

import secrets
import shutil
from collections.abc import Iterator
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_api.thumbnails import ensure_thumbnail
from wheeler_api.uploads import UploadRejected, save_image_upload
from wheeler_shared.editing import (
    EditingError,
    JobBusy,
    NoWheelSelected,
    select_wheel,
    start_generation,
)
from wheeler_shared.history import delete_jobs, is_deletable
from wheeler_shared.models import ImageRole, Job
from wheeler_shared.queue import create_job as create_draft_job
from wheeler_shared.storage import Storage

router = APIRouter()


def get_session_factory(request: Request) -> sessionmaker[Session]:
    factory: sessionmaker[Session] = request.app.state.session_factory
    return factory


def get_session(request: Request) -> Iterator[Session]:
    with get_session_factory(request)() as session:
        yield session


def get_storage(request: Request) -> Storage:
    storage: Storage = request.app.state.storage
    return storage


def get_templates(request: Request) -> Jinja2Templates:
    templates: Jinja2Templates = request.app.state.templates
    return templates


SessionDep = Annotated[Session, Depends(get_session)]
SessionFactoryDep = Annotated[sessionmaker[Session], Depends(get_session_factory)]
StorageDep = Annotated[Storage, Depends(get_storage)]
TemplatesDep = Annotated[Jinja2Templates, Depends(get_templates)]


@router.get("/", response_class=HTMLResponse)
def home(request: Request, templates: TemplatesDep) -> Response:
    return templates.TemplateResponse(request, "index.html", {"error": None})


@router.post("/jobs")
def create_job(
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    templates: TemplatesDep,
    photo: UploadFile | None = None,
    rim: UploadFile | None = None,
) -> Response:
    for field, upload in (("photo", photo), ("rim", rim)):
        if upload is None or not upload.filename:
            return templates.TemplateResponse(
                request, "index.html", {"error": TEXTS[f"error_no_{field}"]}, status_code=400
            )
    assert photo is not None and rim is not None

    job_dir = storage.new_job_dir()
    try:
        input_path = save_image_upload(storage, photo, f"{job_dir}/input")
        rim_path = save_image_upload(storage, rim, f"{job_dir}/rim")
        with session.begin():
            job_id = create_draft_job(session, input_path=input_path, rim_path=rim_path, params={})
    except UploadRejected as rejected:
        shutil.rmtree(storage.path(job_dir), ignore_errors=True)
        return templates.TemplateResponse(
            request, "index.html", {"error": rejected.message}, status_code=400
        )
    except BaseException:
        shutil.rmtree(storage.path(job_dir), ignore_errors=True)
        raise
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/wheels", response_class=HTMLResponse)
def click_wheel(
    job_id: int,
    request: Request,
    session: SessionDep,
    templates: TemplatesDep,
    x: Annotated[float, Form()],
    y: Annotated[float, Form()],
) -> Response:
    """Klik w koło na zdjęciu (piksele zdjęcia): worker wyznaczy maskę (SAM 2)."""
    status_code = 200
    try:
        with session.begin():
            select_wheel(session, job_id, (x, y))
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
    job_id: int, request: Request, session: SessionDep, templates: TemplatesDep
) -> Response:
    try:
        with session.begin():
            start_generation(session, job_id, seed=secrets.randbelow(2**31))
    except LookupError:
        raise HTTPException(status_code=404) from None
    except EditingError as exc:
        message = TEXTS["error_no_wheel" if isinstance(exc, NoWheelSelected) else "error_job_busy"]
        return templates.TemplateResponse(
            request, "error.html", {"message": message}, status_code=409
        )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


def _load_job(session: Session, job_id: int) -> Job:
    job = session.get(
        Job,
        job_id,
        options=[selectinload(Job.images), selectinload(Job.wheels)],
        populate_existing=True,
    )
    if job is None:
        raise HTTPException(status_code=404)
    return job


def _job_context(job: Job) -> dict[str, object]:
    return {
        "job": job,
        "rim_image": job.image(ImageRole.RIM),
        "input_image": job.image(ImageRole.INPUT),
        "result_image": job.image(ImageRole.RESULT),
    }


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(
    job_id: int, request: Request, session: SessionDep, templates: TemplatesDep
) -> Response:
    job = _load_job(session, job_id)
    return templates.TemplateResponse(request, "job.html", _job_context(job))


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
