"""Biblioteka felg: strona `/rims` i fragment HTMX `#rim-picker` (też na stronie głównej).

Każde działanie w bibliotece zwraca ponownie wyrenderowany fragment. `mode` mówi, gdzie
fragment jest osadzony: `select` (strona główna, felgi do wyboru dla formularza zadania)
albo `manage` (strona biblioteki, z usuwaniem). `rim_id` to felga zaznaczona w formularzu.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session

from wheeler_api.rim_picker import Mode, picker_context
from wheeler_api.routes import (
    SessionDep,
    SessionFactoryDep,
    StorageDep,
    TemplatesDep,
)
from wheeler_api.texts import TEXTS
from wheeler_api.uploads import UploadRejected, save_image_upload
from wheeler_shared.rim_library import (
    RimBusy,
    RimError,
    accept_rim,
    add_rim,
    delete_rim,
    refine_rim,
)

router = APIRouter()


def _picker(
    request: Request,
    session: Session,
    templates: TemplatesDep,
    *,
    mode: Mode,
    selected: int | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> Response:
    context = picker_context(session, mode=mode, selected=selected, error=error)
    return templates.TemplateResponse(request, "_rim_picker.html", context, status_code=status_code)


@router.get("/rims", response_class=HTMLResponse)
def rims_page(request: Request, session: SessionDep, templates: TemplatesDep) -> Response:
    return templates.TemplateResponse(request, "rims.html", picker_context(session, mode="manage"))


@router.get("/rims/picker", response_class=HTMLResponse)
def rim_picker(
    request: Request,
    session: SessionDep,
    templates: TemplatesDep,
    mode: Mode = "select",
    rim_id: int | None = None,
) -> Response:
    return _picker(request, session, templates, mode=mode, selected=rim_id)


@router.post("/rims", response_class=HTMLResponse)
def create_rim(
    request: Request,
    session: SessionDep,
    storage: StorageDep,
    templates: TemplatesDep,
    rim: UploadFile | None = None,
    name: Annotated[str, Form()] = "",
    mode: Annotated[Mode, Form()] = "select",
    rim_id: Annotated[int | None, Form()] = None,
) -> Response:
    """Nowa felga: zapis oryginału, worker usunie tło i pokaże podgląd do akceptacji."""
    if rim is None or not rim.filename:
        return _picker(
            request,
            session,
            templates,
            mode=mode,
            selected=rim_id,
            error=TEXTS["error_rim_file"],
            status_code=400,
        )
    rim_dir = storage.new_rim_dir()
    try:
        original = save_image_upload(storage, rim, f"{rim_dir}/original")
        with session.begin():
            add_rim(session, original_path=original, name=name)
    except UploadRejected as rejected:
        storage.remove_tree(rim_dir)
        return _picker(
            request,
            session,
            templates,
            mode=mode,
            selected=rim_id,
            error=rejected.message,
            status_code=400,
        )
    except BaseException:
        storage.remove_tree(rim_dir)
        raise
    return _picker(request, session, templates, mode=mode, selected=rim_id)


@router.post("/rims/{rim}/accept", response_class=HTMLResponse)
def accept(
    rim: int,
    request: Request,
    session: SessionDep,
    templates: TemplatesDep,
    mode: Annotated[Mode, Form()] = "select",
) -> Response:
    """Akceptacja podglądu: felga trafia do biblioteki i od razu jest wybrana."""
    try:
        with session.begin():
            accept_rim(session, rim)
    except LookupError:
        raise HTTPException(status_code=404) from None
    except RimError:
        return _picker(
            request, session, templates, mode=mode, error=TEXTS["error_rim_busy"], status_code=409
        )
    return _picker(request, session, templates, mode=mode, selected=rim)


@router.post("/rims/{rim}/click", response_class=HTMLResponse)
def click_rim(
    rim: int,
    request: Request,
    session: SessionDep,
    templates: TemplatesDep,
    x: Annotated[float, Form()],
    y: Annotated[float, Form()],
    mode: Annotated[Mode, Form()] = "select",
    rim_id: Annotated[int | None, Form()] = None,
) -> Response:
    """Klik w felgę na oryginale: worker wytnie samo koło (SAM 2) zamiast całego obiektu."""
    try:
        with session.begin():
            refine_rim(session, rim, (x, y))
    except LookupError:
        raise HTTPException(status_code=404) from None
    except RimError:
        return _picker(
            request,
            session,
            templates,
            mode=mode,
            selected=rim_id,
            error=TEXTS["error_rim_busy"],
            status_code=409,
        )
    return _picker(request, session, templates, mode=mode, selected=rim_id)


@router.post("/rims/{rim}/delete", response_class=HTMLResponse)
def delete(
    rim: int,
    request: Request,
    session: SessionDep,
    session_factory: SessionFactoryDep,
    storage: StorageDep,
    templates: TemplatesDep,
    mode: Annotated[Mode, Form()] = "select",
    rim_id: Annotated[int | None, Form()] = None,
) -> Response:
    """Odrzucenie podglądu albo usunięcie z biblioteki (razem z plikami felgi)."""
    try:
        delete_rim(session_factory, storage, rim)
    except RimBusy:
        return _picker(
            request,
            session,
            templates,
            mode=mode,
            selected=rim_id,
            error=TEXTS["error_rim_busy"],
            status_code=409,
        )
    selected = None if rim_id == rim else rim_id
    return _picker(request, session, templates, mode=mode, selected=selected)
