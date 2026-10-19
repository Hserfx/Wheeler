"""Kolejność kroków strony głównej (review, spec 3): zdjęcie auta bez felgi, od razu detekcja,
felga z biblioteki albo nowa (z podglądem) wybierana przed „Generuj”."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient
from helpers import (
    click_wheel,
    drain,
    fake_processor,
    fetch,
    image_src,
    library_rim,
    rim_png,
    rim_processor,
    upload_photo,
)
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage


def _with_mask(client: TestClient, session_factory: sessionmaker[Session], storage: Storage) -> int:
    job_id = upload_photo(client)
    drain(session_factory, storage)
    click_wheel(client, job_id)
    assert drain(session_factory, storage)
    return job_id


def test_car_photo_without_rim_starts_detection_and_offers_rim_choice(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload_photo(client)

    detecting = client.get(f"/jobs/{job_id}").text
    assert TEXTS["status_detecting"] in detecting  # detekcja rusza od razu
    assert 'id="rim-picker"' in detecting  # felgę można wybierać w trakcie detekcji
    assert drain(session_factory, storage) == 1
    page = client.get(f"/jobs/{job_id}").text
    assert 'data-role="wheel-picker"' in page and 'id="rim-picker"' in page


def test_generate_without_rim_is_refused_with_readable_message(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _with_mask(client, session_factory, storage)

    response = client.post(f"/jobs/{job_id}/generate", follow_redirects=False)

    assert response.status_code == 409
    assert TEXTS["error_no_rim"] in response.text
    assert 'id="rim-picker"' in response.text  # wybór felgi zostaje na miejscu
    assert TEXTS["status_draft"] in client.get(f"/jobs/{job_id}/status").text


def test_rim_chosen_on_job_page_is_used_for_generation(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim_id = library_rim(client)
    job_id = _with_mask(client, session_factory, storage)
    page = client.get(f"/jobs/{job_id}").text
    # Radio felgi należy do formularza „Generuj” (wspólne id formularza zadania).
    assert re.search(rf'name="rim_id" value="{rim_id}" form="job-form"', page)
    assert re.search(rf'<form id="job-form" action="/jobs/{job_id}/generate"', page)

    response = client.post(
        f"/jobs/{job_id}/generate", data={"rim_id": str(rim_id)}, follow_redirects=False
    )
    assert response.status_code == 303
    assert drain(session_factory, storage, fake_processor())

    done = client.get(f"/jobs/{job_id}").text
    assert TEXTS["status_done"] in done and 'id="rim-picker"' not in done
    cut = fetch(client, "/files/" + _cut(client))
    assert fetch(client, image_src(done, "rim")).size == cut.size  # kopia felgi w zadaniu
    assert TEXTS["rim_used"].format(count=1) in client.get("/rims").text


def _cut(client: TestClient) -> str:
    match = re.search(r'src="/files/([^"]+)"[^>]*data-role="library-rim"', client.get("/rims").text)
    assert match
    return match.group(1)


def test_new_rim_added_on_job_page_waits_for_review_then_can_be_chosen(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload_photo(client)
    response = client.post(
        "/rims", files={"rim": ("felga.png", rim_png(), "image/png")}, data={"mode": "select"}
    )
    assert response.status_code == 200 and TEXTS["rim_cutting"] in response.text
    drain(session_factory, storage, rim_processor())  # felga (podgląd) i detekcja kół
    [rim_id] = re.findall(r'data-rim-id="(\d+)"', client.get("/rims/picker").text)
    accepted = client.post(f"/rims/{rim_id}/accept", data={"mode": "select"}).text
    assert re.search(rf'name="rim_id" value="{rim_id}" form="job-form"[^>]*checked', accepted)
    assert 'id="rim-picker"' in client.get(f"/jobs/{job_id}").text
