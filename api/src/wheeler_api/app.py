"""Fabryka aplikacji FastAPI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from wheeler_api.routes import router
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import STATUS_LABELS, TEXTS
from wheeler_shared.db import make_engine, make_session_factory
from wheeler_shared.migrate import migrate
from wheeler_shared.storage import Storage

PACKAGE_DIR = Path(__file__).resolve().parent


def create_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    templates.env.globals["t"] = TEXTS
    templates.env.globals["status_labels"] = STATUS_LABELS
    return templates


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.data_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.run_migrations:
            migrate(settings.database_url)
        engine = make_engine(settings.database_url)
        app.state.session_factory = make_session_factory(engine)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="Wheeler", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.storage = storage
    app.state.templates = create_templates()

    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    app.mount("/files", StaticFiles(directory=storage.root, check_dir=False), name="files")
    app.include_router(router)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        message = TEXTS["error_not_found"] if exc.status_code == 404 else TEXTS["error_generic"]
        templates: Jinja2Templates = request.app.state.templates
        response: HTMLResponse = templates.TemplateResponse(
            request, "error.html", {"message": message}, status_code=exc.status_code
        )
        return response

    return app
