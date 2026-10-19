"""Biblioteka felg (prawdziwy Postgres, atrapy modeli): dodanie felgi z usunięciem tła,
akceptacja podglądu, wybór z biblioteki przy nowym zdjęciu i usuwanie z biblioteki.
"""

from __future__ import annotations

import re

import numpy as np
from fake_models import (
    BoxDetector,
    CenterDiscRemover,
    DiscSegmenter,
    NothingRemover,
    NothingSegmenter,
    SolidEditor,
)
from fastapi.testclient import TestClient
from helpers import (
    click_wheel,
    drain,
    encode,
    fetch,
    finish_job,
    generate,
    image_src,
    library_rim,
    noise_photo,
    rim_processor,
    upload,
)
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import TEXTS
from wheeler_shared.rim_library import claim_next_rim
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor


def _post_rim(client: TestClient, rim: Image.Image, name: str = "") -> str:
    response = client.post(
        "/rims", files={"rim": ("felga.png", encode(rim), "image/png")}, data={"name": name}
    )
    assert response.status_code == 200, response.text
    return response.text


def test_added_rim_has_background_removed_by_worker_and_waits_for_review(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim = noise_photo(100, 80)

    cutting = _post_rim(client, rim, name="BBS CH-R")
    assert TEXTS["rim_cutting"] in cutting
    assert 'hx-trigger="every 1s"' in cutting
    # „Model” uznaje za felgę dysk o promieniu 32 px wokół środka zdjęcia (50, 40).
    assert run_once(session_factory, storage, rim_processor(CenterDiscRemover(radius=32)))

    review = client.get("/rims/picker").text
    assert "hx-trigger" not in review
    assert TEXTS["rim_review_hint"] in review
    rim_id = re.search(r'data-rim-id="(\d+)"', review)
    assert rim_id
    assert f'hx-post="/rims/{rim_id.group(1)}/accept"' in review
    cut = fetch(client, image_src(review, "rim-cut"))
    # Wycięta felga: przezroczyste tło, przycięta do felgi (bok 65 px, od (18, 8)).
    assert cut.format == "PNG" and cut.mode == "RGBA" and cut.size == (65, 65)
    alpha = np.asarray(cut)[..., 3]
    assert alpha[0, 0] == 0 and alpha[32, 32] == 255
    assert tuple(np.asarray(cut)[32, 32, :3]) == rim.getpixel((50, 40))


def _rim_id(html: str) -> int:
    match = re.search(r'data-rim-id="(\d+)"', html)
    assert match, html
    return int(match.group(1))


def _library_ids(html: str) -> list[int]:
    return [int(m) for m in re.findall(r'data-library-rim-id="(\d+)"', html)]


def test_accepted_rim_joins_library_and_is_selected_on_home_page(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _post_rim(client, noise_photo(100, 80), name="BBS CH-R")
    assert run_once(session_factory, storage, rim_processor(CenterDiscRemover(radius=32)))
    rim_id = _rim_id(client.get("/rims/picker").text)

    accepted = client.post(f"/rims/{rim_id}/accept", data={"mode": "select"})
    assert accepted.status_code == 200
    assert _library_ids(accepted.text) == [rim_id]
    assert re.search(rf'name="rim_id" value="{rim_id}"[^>]*checked', accepted.text)
    assert "data-rim-id" not in accepted.text  # podgląd zniknął

    home = client.get("/").text
    assert _library_ids(home) == [rim_id]
    assert "BBS CH-R" in home
    assert 'form="job-form"' in home and 'id="job-form"' in home


def test_model_gets_cut_rim_on_plain_background_not_the_original(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim = noise_photo(100, 80)
    rim_id = library_rim(client, encode(rim), remover=CenterDiscRemover(radius=32))
    job_id = upload(client, rim_id=rim_id)
    editor = SolidEditor()
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
    )
    drain(session_factory, storage, processor)  # detekcja kół po wgraniu (atrapa: brak)
    click_wheel(client, job_id)
    assert drain(session_factory, storage, processor)
    generate(client, job_id)
    assert drain(session_factory, storage, processor)

    reference = np.asarray(editor.calls[0].images[1].convert("RGB"))
    assert reference.shape == (65, 65, 3)  # przycięta do felgi
    assert tuple(reference[0, 0]) == (255, 255, 255)  # tło usunięte
    assert tuple(reference[32, 32]) == rim.getpixel((50, 40))


def _library_item(html: str, rim_id: int) -> str:
    match = re.search(rf'<li data-library-rim-id="{rim_id}">.*?</li>', html, re.S)
    assert match, html
    return match.group(0)


def test_library_is_sorted_by_last_use_and_counts_uses(client: TestClient) -> None:
    first = library_rim(client, name="Pierwsza")
    second = library_rim(client, name="Druga")
    third = library_rim(client)
    assert _library_ids(client.get("/rims").text) == [third, second, first]

    upload(client, rim_id=first)
    upload(client, rim_id=first)

    page = client.get("/rims").text
    assert _library_ids(page) == [first, third, second]
    assert TEXTS["rim_used"].format(count=2) in _library_item(page, first)
    assert re.search(r'<time datetime="\d{4}-', _library_item(page, first))
    assert TEXTS["rim_used"].format(count=0) in _library_item(page, second)
    assert TEXTS["rim_unnamed"].format(id=third) in _library_item(page, third)
    assert _library_ids(client.get("/").text) == [first, third, second]


def test_deleting_rim_from_library_keeps_jobs_that_used_it_working(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim_id = library_rim(client, name="Do usunięcia")
    done_job = upload(client, rim_id=rim_id)
    finish_job(client, session_factory, storage, done_job)
    draft_job = upload(client, rim_id=rim_id)
    rim_files = set(storage.root.glob("rims/*/*"))
    assert rim_files

    response = client.post(f"/rims/{rim_id}/delete", data={"mode": "manage"})
    assert response.status_code == 200
    assert _library_ids(response.text) == []
    assert not any(p.exists() for p in rim_files)  # pliki felgi usunięte z dysku

    done_page = client.get(f"/jobs/{done_job}").text
    assert TEXTS["status_done"] in done_page
    for role in ("rim", "result"):
        assert client.get(image_src(done_page, role)).status_code == 200
    # Szkic sprzed usunięcia ma własną kopię felgi, więc da się go dokończyć.
    finish_job(client, session_factory, storage, draft_job)
    assert TEXTS["status_done"] in client.get(f"/jobs/{draft_job}").text


def test_rejected_preview_is_removed_with_its_files(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _post_rim(client, noise_photo(100, 80))
    assert run_once(session_factory, storage, rim_processor(CenterDiscRemover()))
    rim_id = _rim_id(client.get("/rims/picker").text)

    response = client.post(f"/rims/{rim_id}/delete", data={"mode": "select"})
    assert response.status_code == 200
    assert "data-rim-id" not in response.text and _library_ids(response.text) == []
    assert list(storage.root.glob("rims/*/*")) == []


def test_background_is_removed_around_the_largest_detected_wheel_not_a_box_around_it(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    # Zdjęcie felgi na aucie: BiRefNet na całym zdjęciu zostawiłby karoserię, więc dostaje
    # tylko kadr wokół koła z detektora (box 80 px + 8% marginesu = kadr 93x93 od (103, 3)).
    rim = noise_photo(200, 100)
    remover = CenterDiscRemover(radius=40)
    # Florence-2 zwraca czasem też box obejmujący auto razem z kołem: ten pomijamy.
    detector = BoxDetector([(10, 10, 50, 50), (110, 10, 190, 90), (0, 0, 199, 99)])
    _post_rim(client, rim)
    assert run_once(session_factory, storage, rim_processor(remover, detector))

    assert [image.size for image in remover.images] == [(93, 93)]
    cut = fetch(client, image_src(client.get("/rims/picker").text, "rim-cut"))
    assert cut.size == (81, 81)
    assert tuple(np.asarray(cut)[40, 40, :3]) == rim.getpixel((149, 49))


def test_click_on_rim_picks_the_detected_wheel_under_it(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim = noise_photo(200, 100)
    # Florence-2 zwraca czasem też box obejmujący auto razem z kołem: ten pomijamy.
    detector = BoxDetector([(10, 10, 50, 50), (110, 10, 190, 90), (0, 0, 199, 99)])
    _post_rim(client, rim)
    assert run_once(session_factory, storage, rim_processor(detector=detector))
    rim_id = _rim_id(client.get("/rims/picker").text)

    client.post(f"/rims/{rim_id}/click", data={"x": 30, "y": 30, "mode": "select"})
    remover = CenterDiscRemover(radius=15)
    assert run_once(session_factory, storage, rim_processor(remover, detector))

    # Mały box 40 px + 8% marginesu = kadr 47x47 od (6, 6); felga: dysk wokół (29, 29).
    assert [image.size for image in remover.images] == [(47, 47)]
    cut = fetch(client, image_src(client.get("/rims/picker").text, "rim-cut"))
    assert cut.size == (31, 31)
    assert tuple(np.asarray(cut)[15, 15, :3]) == rim.getpixel((29, 29))


def test_click_on_rim_without_detected_wheel_cuts_it_out_from_the_click(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim = noise_photo(100, 80)
    _post_rim(client, rim)
    assert run_once(session_factory, storage, rim_processor())  # „BiRefNet” zostawił wszystko
    review = client.get("/rims/picker").text
    rim_id = _rim_id(review)
    assert 'data-role="rim-original"' in review

    cutting = client.post(f"/rims/{rim_id}/click", data={"x": 30, "y": 25, "mode": "select"})
    assert cutting.status_code == 200
    assert TEXTS["rim_cutting"] in cutting.text and 'hx-trigger="every 1s"' in cutting.text
    processor = WheelSwapProcessor(
        detector=BoxDetector(),
        segmenter=DiscSegmenter(radius=10),
        editor=SolidEditor(),
        remover=CenterDiscRemover(),
    )
    assert run_once(session_factory, storage, processor)

    cut = fetch(client, image_src(client.get("/rims/picker").text, "rim-cut"))
    assert cut.size == (21, 21)  # „SAM 2”: koło o promieniu 10 wokół kliknięcia
    assert tuple(np.asarray(cut)[10, 10, :3]) == rim.getpixel((30, 25))


def test_failed_background_removal_shows_error_and_allows_click_or_reject(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _post_rim(client, noise_photo(100, 80))
    nothing = WheelSwapProcessor(
        detector=BoxDetector(),
        segmenter=NothingSegmenter(),
        editor=SolidEditor(),
        remover=NothingRemover(),
    )
    assert run_once(session_factory, storage, nothing)

    failed = client.get("/rims/picker").text
    assert TEXTS["rim_cut_failed"] in failed and "hx-trigger" not in failed
    assert TEXTS["failure_rim_not_found"] in failed and "ValueError" not in failed
    rim_id = _rim_id(failed)
    assert f'hx-post="/rims/{rim_id}/accept"' not in failed
    assert f'hx-post="/rims/{rim_id}/delete"' in failed
    response = client.post(f"/rims/{rim_id}/click", data={"x": 50, "y": 40})
    assert response.status_code == 200 and TEXTS["rim_cutting"] in response.text


def test_rim_cannot_be_used_or_accepted_before_review(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim_id = _rim_id(_post_rim(client, noise_photo(100, 80)))
    assert client.post(f"/rims/{rim_id}/accept").status_code == 409
    response = client.post(
        "/jobs",
        files={"photo": ("auto.png", encode(noise_photo()), "image/png")},
        data={"rim_id": str(rim_id)},
    )
    assert response.status_code == 400
    assert TEXTS["error_no_rim"] in response.text
    assert list(storage.root.glob("jobs/*")) == []


def test_rim_being_cut_by_worker_cannot_be_deleted(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    rim_id = _rim_id(_post_rim(client, noise_photo(100, 80)))
    with session_factory.begin() as session:
        assert claim_next_rim(session) is not None  # worker zaczął wycinać

    response = client.post(f"/rims/{rim_id}/delete")
    assert response.status_code == 409
    assert TEXTS["error_rim_busy"] in response.text
    assert list(storage.root.glob("rims/*/original.*"))


def test_deleting_job_keeps_rim_in_library(client: TestClient) -> None:
    rim_id = library_rim(client)
    job_id = upload(client, rim_id=rim_id)
    response = client.post("/history/delete", data={"ids": [job_id]}, follow_redirects=False)
    assert response.status_code == 303
    page = client.get("/rims").text
    assert _library_ids(page) == [rim_id]
    assert client.get(image_src(page, "library-rim")).status_code == 200
