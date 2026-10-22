"""Nieudane generowanie wraca do szkicu (ticket 14).

Zdjęcie, koła z maskami, felga i parametry (w tym seed) zostają, strona zadania znów pokazuje
wybór felgi, rozmiaru i edycję kół, a komunikat błędu wisi do kolejnego „Generuj”.
Przez HTTP i `run_once` workera (prawdziwy Postgres, atrapy SAM 2 i mflux).
"""

from __future__ import annotations

import re

import pytest
from fake_models import BoxDetector, BrokenEditor, DiscSegmenter, NothingSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import (
    click_wheel,
    draft_with_wheel,
    drain,
    generate,
    image_src,
    job_page,
    job_status,
    params_block,
)
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import PARAM_LABELS, TEXTS
from wheeler_shared import edit_params
from wheeler_shared.failures import Failure
from wheeler_shared.models import Job, JobStatus
from wheeler_shared.queue import claim_next_job, fail_job
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


def _processor(editor: SolidEditor, *, rim_radius: int | None = None) -> WheelSwapProcessor:
    return WheelSwapProcessor(
        detector=BoxDetector(),
        segmenter=DiscSegmenter(radius=10, rim_radius=rim_radius),
        editor=editor,
    )


def test_failed_generation_returns_the_job_to_a_draft_with_the_error_and_its_data(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = _processor(BrokenEditor())
    job_id = draft_with_wheel(client, session_factory, storage, processor)
    before = job_page(client, job_id)

    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    page = job_page(client, job_id)
    assert 'data-status="draft"' in page and TEXTS["status_draft"] in page
    assert TEXTS["failure_internal"] in page
    assert "model się wyłożył" not in page
    assert "hx-trigger" not in job_status(client, job_id)  # worker skończył, nic nie odpytuje
    # Zdjęcie, maska koła i felga zostają; strona znów pozwala wybrać felgę, rozmiar i koła.
    assert image_src(page, "input") == image_src(before, "input")
    assert image_src(page, "mask-overlay") == image_src(before, "mask-overlay")
    assert page.count('data-role="mask-overlay"') == before.count('data-role="mask-overlay"') == 1
    assert 'data-role="rim"' in page
    assert 'data-role="wheel-picker"' in page
    assert 'data-role="rim-choice"' in page and 'data-role="rim-size"' in page
    assert 'id="job-form"' in page
    assert f'action="/jobs/{job_id}/retry"' not in page
    assert TEXTS["job_finished"] not in page
    # Seed zapisany przy pierwszej próbie jest w parametrach zadania.
    assert 'data-param="seed"' in page


def test_generating_again_from_the_draft_queues_the_job_with_the_first_seed(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    broken = BrokenEditor()
    job_id = draft_with_wheel(client, session_factory, storage, _processor(broken))
    generate(client, job_id)
    assert run_once(session_factory, storage, _processor(broken))

    generate(client, job_id)

    status = job_status(client, job_id)
    assert TEXTS["status_queued"] in status
    assert TEXTS["failure_internal"] not in status  # „Generuj” czyści komunikat błędu
    editor = SolidEditor()
    assert run_once(session_factory, storage, _processor(editor))
    assert TEXTS["status_done"] in job_status(client, job_id)
    [first], [second] = broken.calls, editor.calls
    assert second.seed == first.seed
    assert second.mask.tobytes() == first.mask.tobytes()
    assert second.images[1].tobytes() == first.images[1].tobytes()  # ta sama felga


def test_error_stays_visible_while_the_draft_is_edited(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = _processor(BrokenEditor())
    job_id = draft_with_wheel(client, session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    status = click_wheel(client, job_id)  # klik w maskę usuwa koło

    assert 'data-role="mask-overlay"' not in status
    assert TEXTS["failure_internal"] in status
    assert TEXTS["failure_internal"] in job_page(client, job_id)


def test_generation_error_survives_adding_and_removing_wheels_until_generate_clears_it(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = _processor(BrokenEditor())
    job_id = draft_with_wheel(client, session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    status = click_wheel(client, job_id, (10, 10))  # dodanie koła zleca workerowi maskę
    assert TEXTS["status_segmenting"] in status
    assert TEXTS["failure_internal"] in status
    assert drain(session_factory, storage, processor) == 1

    page = job_page(client, job_id)
    assert page.count('data-role="mask-overlay"') == 2
    assert TEXTS["failure_internal"] in page
    # Prefiks komunikatu zależy od pochodzenia błędu (generowanie), nie od ostatniej pracy workera.
    assert f"{TEXTS['job_error']}: {TEXTS['failure_internal']}" in page
    assert TEXTS["error_mask_failed"] not in page

    status = click_wheel(client, job_id, (10, 10))  # usunięcie dodanego koła
    assert status.count('data-role="mask-overlay"') == 1
    assert TEXTS["failure_internal"] in status
    assert TEXTS["failure_internal"] in job_page(client, job_id)

    generate(client, job_id)
    assert TEXTS["failure_internal"] not in job_status(client, job_id)


def test_failed_mask_after_a_generation_error_is_reported_as_a_mask_error(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = _processor(BrokenEditor())
    job_id = draft_with_wheel(client, session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)
    click_wheel(client, job_id, (10, 10))
    nothing = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=NothingSegmenter(), editor=SolidEditor()
    )
    assert drain(session_factory, storage, nothing) == 1

    page = job_page(client, job_id)

    assert TEXTS["error_mask_failed"] in page and TEXTS["failure_nothing_at_click"] in page
    assert TEXTS["failure_internal"] not in page  # nowy błąd zastąpił poprzedni


def _checked_rim_size(page: str) -> list[str]:
    section = re.search(r'<fieldset[^>]*data-role="rim-size".*?</fieldset>', page, re.S)
    assert section, page
    return re.findall(r'name="rim_size" value="([^"]+)"[^>]*checked', section.group(0))


def test_unmeasurable_rim_resets_rim_size_to_unchanged_and_says_what_to_do(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor = SolidEditor()
    processor = _processor(editor, rim_radius=None)  # „SAM 2” nie widzi felgi
    job_id = draft_with_wheel(client, session_factory, storage, processor)
    response = client.post(
        f"/jobs/{job_id}/generate", data={"rim_size": "2"}, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    assert run_once(session_factory, storage, processor)

    page = job_page(client, job_id)

    assert TEXTS["status_draft"] in page
    assert TEXTS["failure_rim_size_unavailable"] in page
    assert _checked_rim_size(page) == ["0"]  # „bez zmian”
    params = params_block(page)
    assert params and PARAM_LABELS[edit_params.RIM_SIZE] not in params
    assert 'data-param="seed"' in page
    assert editor.calls == []

    # „Generuj” bez zmiany rozmiaru teraz przechodzi (felgi nie trzeba wyznaczać).
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)
    assert TEXTS["status_done"] in job_status(client, job_id)
    assert len(editor.calls) == 1


def test_history_shows_the_job_after_failed_generation_as_a_draft(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = _processor(BrokenEditor())
    job_id = draft_with_wheel(client, session_factory, storage, processor)
    generate(client, job_id)
    assert run_once(session_factory, storage, processor)

    history = client.get("/history").text

    item = re.search(rf'<li[^>]*data-job-id="{job_id}".*?</li>', history, re.S)
    assert item, history
    assert re.search(r'data-status="draft">\s*' + re.escape(TEXTS["status_draft"]), item.group(0))


def test_existing_failed_job_keeps_its_retry(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    # Zadanie `failed` sprzed ticketu 14 (bez migracji): zapisane wprost w bazie.
    job_id = draft_with_wheel(client, session_factory, storage, _processor(SolidEditor()))
    with session_factory.begin() as session:
        job = session.get_one(Job, job_id)
        job.status, job.error = JobStatus.FAILED, Failure.LEGACY_JOB

    status = job_status(client, job_id)
    assert TEXTS["status_failed"] in status and TEXTS["failure_legacy_job"] in status
    assert f'action="/jobs/{job_id}/retry"' in status

    response = client.post(f"/jobs/{job_id}/retry", follow_redirects=False)
    assert response.status_code == 303, response.text
    new_id = int(response.headers["location"].rsplit("/", 1)[1])
    assert TEXTS["status_queued"] in job_status(client, new_id)
    assert TEXTS["status_failed"] in job_status(client, job_id)


@pytest.mark.parametrize(
    "failure",
    [
        Failure.RIM_SIZE_UNAVAILABLE,
        Failure.RIM_NOT_FOUND,
        Failure.EMPTY_MASK,
        Failure.NO_RIM,
        Failure.INTERNAL,
        Failure.INTERRUPTED,
    ],
)
def test_every_generation_failure_returns_the_job_to_a_draft(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage, failure: Failure
) -> None:
    job_id = draft_with_wheel(client, session_factory, storage, _processor(SolidEditor()))
    generate(client, job_id)
    with session_factory.begin() as session:
        claimed = claim_next_job(session)
        assert claimed is not None and claimed.id == job_id
        fail_job(session, job_id, error=failure)

    page = job_page(client, job_id)
    assert 'data-status="draft"' in page
    assert TEXTS[f"failure_{failure.value}"] in page
    assert TEXTS["job_finished"] not in page
    assert 'id="job-form"' in page and 'data-role="mask-overlay"' in page
