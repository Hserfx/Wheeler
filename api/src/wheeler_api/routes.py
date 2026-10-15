"""Strony i fragmenty HTMX."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, selectinload, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.models import ImageRole, Job
from wheeler_shared.queue import enqueue_job
from wheeler_shared.storage import Storage

router = APIRouter()


def get_session(request: Request) -> Iterator[Session]:
    factory: sessionmaker[Session] = request.app.state.session_factory
    with factory() as session:
        yield session


def get_storage(request: Request) -> Storage:
    storage: Storage = request.app.state.storage
    return storage


def get_templates(request: Request) -> Jinja2Templates:
    templates: Jinja2Templates = request.app.state.templates
    return templates


SessionDep = Annotated[Session, Depends(get_session)]
StorageDep = Annotated[Storage, Depends(get_storage)]
TemplatesDep = Annotated[Jinja2Templates, Depends(get_templates)]

_SAFE_SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")


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
) -> Response:
    if photo is None or not photo.filename:
        return templates.TemplateResponse(
            request, "index.html", {"error": TEXTS["error_no_file"]}, status_code=400
        )

    suffix = PurePath(photo.filename).suffix.lower()
    if not _SAFE_SUFFIX.fullmatch(suffix):
        suffix = ".bin"
    job_dir = storage.new_job_dir()
    input_path = f"{job_dir}/input{suffix}"
    try:
        with storage.path(input_path).open("wb") as target:
            shutil.copyfileobj(photo.file, target)
        with session.begin():
            job_id = enqueue_job(session, input_path=input_path, params={})
    except BaseException:
        shutil.rmtree(storage.path(job_dir), ignore_errors=True)
        raise
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


def _load_job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id, options=[selectinload(Job.images)])
    if job is None:
        raise HTTPException(status_code=404)
    return job


def _job_context(job: Job) -> dict[str, object]:
    return {
        "job": job,
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


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
