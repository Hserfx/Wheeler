"""Stan modelu w pamięci workera (uśpiony / ładowanie / gotowy) jako fragment HTMX w nagłówku."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import object_session

from wheeler_api.deps import SessionDep, TemplatesDep
from wheeler_shared.models import Job, JobStatus
from wheeler_shared.worker_state import ModelState, get_model_state

router = APIRouter()


def model_loading_for(job: Job) -> bool:
    """Zadanie w toku czeka na załadowanie modelu (szablony: `_job_status.html`).

    Czyta stan przez sesję, z której pochodzi `job`, więc trasy nie muszą go przekazywać.
    """
    session = object_session(job)
    if session is None or job.status != JobStatus.RUNNING:
        return False
    return get_model_state(session) is ModelState.LOADING


@router.get("/model-state", response_class=HTMLResponse)
def model_state(request: Request, session: SessionDep, templates: TemplatesDep) -> Response:
    return templates.TemplateResponse(
        request, "_model_state.html", {"model_state": get_model_state(session)}
    )
