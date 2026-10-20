"""Pole „Obecna felga” (ticket 14): obecny rozmiar felgi auta 14…22″, domyślnie 17″.

Wybierane obok stopnia rozmiaru felgi przed „Generuj” i zapisywane w `Job.params`; wyznacza
skalę cala przy zmianie rozmiaru (ticket 16, sprawdzane w `test_rim_size.py`). Testy przez
HTTP i `run_once` workera na prawdziwym Postgresie (atrapy modeli), jak `test_rim_size.py`.
"""

from __future__ import annotations

import io
import re

import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, jpeg, upload, upload_photo, wheel_photo
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import PARAM_LABELS, TEXTS
from wheeler_shared import edit_params
from wheeler_shared.edit_params import EditParams
from wheeler_shared.storage import Storage
from wheeler_worker.rim_geometry import Ellipse
from wheeler_worker.wheel_swap import WheelSwapProcessor

# Koło jak w `test_rim_size.py`: opona o promieniu 50 px, felga 35 px, którą atrapa SAM 2 widzi,
# więc generowanie ze stopniem ≠ 0 kończy się wynikiem.
WHEEL = (150, 120)
TIRE_RADIUS = 50
RIM_RADIUS = 35


def test_job_without_current_rim_means_17_inches() -> None:
    assert EditParams.from_params({}).current_rim == 17
    assert EditParams.from_params({edit_params.CURRENT_RIM: 19}).current_rim == 19


def _processor() -> WheelSwapProcessor:
    return WheelSwapProcessor(
        detector=BoxDetector(),
        segmenter=DiscSegmenter(radius=TIRE_RADIUS, rim_radius=RIM_RADIUS),
        editor=SolidEditor(echo=True),
    )


def _draft_with_wheel(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> int:
    """Szkic ze zdjęciem koła, felgą i maską koła, gotowy do „Generuj”."""
    rim = Ellipse(*WHEEL, width=2 * RIM_RADIUS, height=2 * RIM_RADIUS, angle=0)
    photo = io.BytesIO()
    wheel_photo((400, 240), rim, TIRE_RADIUS / RIM_RADIUS).save(photo, format="PNG")
    job_id = upload(client, photo.getvalue(), filename="auto.png")
    drain(session_factory, storage, _processor())
    click_wheel(client, job_id, WHEEL)
    assert drain(session_factory, storage, _processor())
    return job_id


def _current_rim_choices(html: str) -> list[tuple[str, str, bool]]:
    """Wybór obecnej felgi: (wartość, etykieta, wybrany) każdej opcji pola formularza zadania."""
    section = re.search(r'<fieldset[^>]*data-role="rim-size".*?</fieldset>', html, re.S)
    assert section, html
    field = re.search(
        r'<select[^>]*name="current_rim"[^>]*form="job-form"[^>]*>(.*?)</select>',
        section.group(0),
        re.S,
    )
    assert field, section.group(0)
    options = re.findall(r'<option value="([^"]+)"([^>]*)>([^<]*)</option>', field.group(1))
    return [(value, label.strip(), "selected" in attrs) for value, attrs, label in options]


def test_job_page_offers_current_rim_next_to_rim_size_defaulting_to_17(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _draft_with_wheel(client, session_factory, storage)

    page = client.get(f"/jobs/{job_id}").text

    assert "Obecna felga" in page
    assert _current_rim_choices(page) == [
        (str(inches), f"{inches}″", inches == 17) for inches in range(14, 23)
    ]


def _generate(client: TestClient, job_id: int, **data: str) -> None:
    response = client.post(f"/jobs/{job_id}/generate", data=data, follow_redirects=False)
    assert response.status_code == 303, response.text


def _shown_current_rim(client: TestClient, job_id: int) -> str | None:
    """Obecna felga w szczegółach zadania (wszystkie klucze `Job.params`); None = brak klucza."""
    page = client.get(f"/jobs/{job_id}").text
    params = re.search(r"<dl[^>]*data-params.*?</dl>", page, re.S)
    if params is None:
        return None
    pairs = re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", params.group(0), re.S)
    return {label.strip(): value.strip() for label, value in pairs}.get(
        PARAM_LABELS[edit_params.CURRENT_RIM]
    )


def test_chosen_current_rim_is_saved_in_job_params_with_generate(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    for current_rim in ("14", "19", "22"):
        job_id = _draft_with_wheel(client, session_factory, storage)
        assert _shown_current_rim(client, job_id) is None

        _generate(client, job_id, rim_size="2", current_rim=current_rim)

        assert _shown_current_rim(client, job_id) == f"{current_rim}″"
    assert PARAM_LABELS[edit_params.CURRENT_RIM] == "Obecna felga"


@pytest.mark.parametrize("current_rim", ["13", "23", "17.5", "duża"])
def test_current_rim_outside_range_is_refused_and_job_stays_a_draft(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage, current_rim: str
) -> None:
    job_id = _draft_with_wheel(client, session_factory, storage)

    response = client.post(
        f"/jobs/{job_id}/generate",
        data={"rim_size": "1", "current_rim": current_rim},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "Obecna felga musi być pełnym rozmiarem od 14″ do 22″." in response.text
    assert 'id="rim-picker"' in response.text  # wybór felgi i rozmiaru zostaje na miejscu
    assert TEXTS["status_draft"] in client.get(f"/jobs/{job_id}/status").text
    assert _shown_current_rim(client, job_id) is None


@pytest.mark.parametrize(
    ("data", "shown"), [({"current_rim": "19"}, "obecna 19″"), ({}, "obecna 17″")]
)
def test_result_screen_shows_current_rim_next_to_chosen_rim_size(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    data: dict[str, str],
    shown: str,
) -> None:
    job_id = _draft_with_wheel(client, session_factory, storage)
    _generate(client, job_id, rim_size="2", **data)
    assert drain(session_factory, storage, _processor())

    html = client.get(f"/jobs/{job_id}/status").text

    wheels = re.findall(r'<li[^>]*data-wheel-id="\d+".*?</li>', html, re.S)
    assert wheels
    for wheel in wheels:
        size = re.search(r'data-role="result-rim-size"[^>]*>([^<]*)<', wheel)
        current = re.search(r'data-role="result-current-rim"[^>]*>([^<]*)<', wheel)
        assert size and current, wheel
        assert wheel.index("result-rim-size") < wheel.index("result-current-rim")
        assert current.group(1).strip() == shown
    assert "result-current-rim" not in client.get("/history").text  # siatka Historii bez zmian


def test_result_screen_hides_current_rim_when_rim_size_is_unchanged(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _draft_with_wheel(client, session_factory, storage)
    _generate(client, job_id, rim_size="0", current_rim="19")
    assert drain(session_factory, storage, _processor())

    html = client.get(f"/jobs/{job_id}/status").text

    assert 'data-role="wheel-seed"' in html and "result-current-rim" not in html
    assert _shown_current_rim(client, job_id) == "19″"  # zapisana mimo „bez zmian”


def test_retried_and_regenerated_jobs_keep_the_same_current_rim(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _draft_with_wheel(client, session_factory, storage)
    _generate(client, job_id, rim_size="1", current_rim="20")
    assert drain(session_factory, storage, _processor())
    [wheel_id] = re.findall(r'<li[^>]*data-wheel-id="(\d+)"', client.get(f"/jobs/{job_id}").text)
    response = client.post(f"/jobs/{job_id}/wheels/{wheel_id}/regenerate", follow_redirects=False)
    assert response.status_code == 303, response.text
    assert drain(session_factory, storage, _processor())
    assert "obecna 20″" in client.get(f"/jobs/{job_id}/status").text
    assert _shown_current_rim(client, job_id) == "20″"

    second = _draft_with_wheel(client, session_factory, storage)
    _generate(client, second, rim_size="-1", current_rim="15")
    assert client.post(f"/jobs/{second}/cancel").status_code == 200  # w kolejce: od razu
    response = client.post(f"/jobs/{second}/retry", follow_redirects=False)
    assert response.status_code == 303, response.text
    retried_id = int(response.headers["location"].rsplit("/", 1)[1])
    assert _shown_current_rim(client, retried_id) == "15″"


def test_chosen_current_rim_stays_selected_when_generation_is_refused(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload_photo(client)  # bez felgi: „Generuj” zostanie odrzucone
    drain(session_factory, storage)
    click_wheel(client, job_id)
    assert drain(session_factory, storage)

    response = client.post(
        f"/jobs/{job_id}/generate",
        data={"rim_size": "1", "current_rim": "21"},
        follow_redirects=False,
    )

    assert response.status_code == 409
    assert [value for value, _, chosen in _current_rim_choices(response.text) if chosen] == ["21"]


def test_upload_does_not_take_current_rim(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        files={"photo": ("auto.jpg", jpeg(), "image/jpeg")},
        data={"current_rim": "19"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    job_id = int(response.headers["location"].rsplit("/", 1)[1])
    assert _shown_current_rim(client, job_id) is None
