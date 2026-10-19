"""Rozmiar felgi (plus-sizing, ticket 12): stopień -2…+2″ wybierany przed „Generuj”.

Zmienia się tylko średnica felgi, zewnętrzna średnica koła zostaje; steruje tym wyłącznie
prompt. Prompt testowany bezpośrednio (`build_prompt`), reszta przez HTTP i `run_once`
workera z atrapami modeli, na prawdziwym Postgresie.
"""

from __future__ import annotations

import re

import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, jpeg, upload, upload_photo
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import PARAM_LABELS, TEXTS
from wheeler_shared import edit_params
from wheeler_shared.storage import Storage
from wheeler_worker.wheel_swap import WheelSwapProcessor, build_prompt

MINUS = "\N{MINUS SIGN}"  # w UI stopnie ujemne mają minus typograficzny

# Szablon promptu sprzed ticketu 12 (DESIGN.md, spike 01): przy „bez zmian” bez zmian.
CURRENT_TEMPLATE = (
    "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. "
    "Keep the tire, the car body, the wheel arch, the brake caliper position, the lighting, "
    "the shadows and the camera angle of image 1 unchanged. The new rim must follow the "
    "perspective of the wheel in image 1 and fill the tire opening exactly. Photorealistic."
)


def test_unchanged_rim_size_gives_exactly_the_current_prompt() -> None:
    assert build_prompt(rim_size=0) == CURRENT_TEMPLATE
    assert build_prompt("Chrome finish.", rim_size=0) == f"{CURRENT_TEMPLATE} Chrome finish."


@pytest.mark.parametrize(
    ("rim_size", "direction", "inches", "percent", "sidewall"),
    [
        (1, "larger", "1 inch", "4%", "thinner"),
        (2, "larger", "2 inches", "8%", "thinner"),
        (-1, "smaller", "1 inch", "4%", "thicker"),
        (-2, "smaller", "2 inches", "8%", "thicker"),
    ],
)
def test_plus_sizing_prompt_tells_direction_scale_and_sidewall_and_keeps_wheel_diameter(
    rim_size: int, direction: str, inches: str, percent: str, sidewall: str
) -> None:
    prompt = build_prompt("Chrome finish.", rim_size=rim_size)

    assert prompt.startswith(
        "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2."
    )
    assert f"about {inches} {direction} in diameter" in prompt
    assert f"about {percent} of the overall wheel diameter" in prompt
    assert f"{sidewall} tire sidewall" in prompt
    assert "Keep the overall outer diameter of the wheel" in prompt
    # Sprzeczne z plus-sizingiem: felga ma nie wypełniać obecnego otworu opony.
    assert "fill the tire opening exactly" not in prompt
    assert "Keep the tire," not in prompt
    assert prompt.endswith("Photorealistic. Chrome finish.")


def _editor_processor() -> tuple[SolidEditor, WheelSwapProcessor]:
    editor = SolidEditor()
    return editor, WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=10), editor=editor
    )


def _draft_with_wheel(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    processor: WheelSwapProcessor,
) -> int:
    """Szkic z felgą i jednym kołem z maską, gotowy do „Generuj”."""
    job_id = upload(client)
    drain(session_factory, storage, processor)
    click_wheel(client, job_id)
    assert drain(session_factory, storage, processor)
    return job_id


def _generate(client: TestClient, job_id: int, rim_size: str | None) -> None:
    data = {} if rim_size is None else {"rim_size": rim_size}
    response = client.post(f"/jobs/{job_id}/generate", data=data, follow_redirects=False)
    assert response.status_code == 303, response.text


def test_chosen_rim_size_reaches_the_editor_also_when_regenerating_one_wheel(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    _generate(client, job_id, "2")
    assert drain(session_factory, storage, processor)
    [wheel_id] = re.findall(r'<li[^>]*data-wheel-id="(\d+)"', client.get(f"/jobs/{job_id}").text)
    response = client.post(f"/jobs/{job_id}/wheels/{wheel_id}/regenerate", follow_redirects=False)
    assert response.status_code == 303, response.text
    assert drain(session_factory, storage, processor)

    generated, regenerated = editor.calls
    assert "about 2 inches larger in diameter" in generated.prompt
    assert regenerated.prompt == generated.prompt
    assert regenerated.seed != generated.seed


def test_retried_job_edits_with_the_same_rim_size(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)
    _generate(client, job_id, "-2")
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 200  # w kolejce: od razu

    response = client.post(f"/jobs/{job_id}/retry", follow_redirects=False)
    assert response.status_code == 303, response.text
    assert drain(session_factory, storage, processor)

    [call] = editor.calls
    assert "about 2 inches smaller in diameter" in call.prompt
    retried_id = int(response.headers["location"].rsplit("/", 1)[1])
    assert _shown_rim_size(client, retried_id) == f"{MINUS}2″"


@pytest.mark.parametrize("rim_size", ["3", "-3", "1.5", "duży"])
def test_rim_size_outside_range_is_refused_and_job_stays_a_draft(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage, rim_size: str
) -> None:
    _, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    response = client.post(
        f"/jobs/{job_id}/generate", data={"rim_size": rim_size}, follow_redirects=False
    )

    assert response.status_code == 400
    assert "Rozmiar felgi musi być stopniem od -2 do +2 cali." in response.text
    assert 'id="rim-picker"' in response.text  # wybór felgi i rozmiaru zostaje na miejscu
    assert TEXTS["status_draft"] in client.get(f"/jobs/{job_id}/status").text
    assert _shown_rim_size(client, job_id) is None


@pytest.mark.parametrize(("rim_size", "shown"), [("0", "bez zmian"), (None, None)])
def test_unchanged_rim_size_or_missing_field_edits_with_the_current_prompt(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    rim_size: str | None,
    shown: str | None,
) -> None:
    editor, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    _generate(client, job_id, rim_size)
    assert drain(session_factory, storage, processor)

    [call] = editor.calls
    assert call.prompt == CURRENT_TEMPLATE
    assert _shown_rim_size(client, job_id) == shown


def test_rim_size_is_saved_in_job_params_for_each_step(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _, processor = _editor_processor()
    for rim_size, shown in (
        ("-2", f"{MINUS}2″"),
        ("-1", f"{MINUS}1″"),
        ("+1", "+1″"),
        ("2", "+2″"),
    ):
        job_id = _draft_with_wheel(client, session_factory, storage, processor)
        _generate(client, job_id, rim_size)
        assert _shown_rim_size(client, job_id) == shown


def test_upload_does_not_take_rim_size(client: TestClient) -> None:
    response = client.post(
        "/jobs",
        files={"photo": ("auto.jpg", jpeg(), "image/jpeg")},
        data={"rim_size": "2"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    job_id = int(response.headers["location"].rsplit("/", 1)[1])
    assert _shown_rim_size(client, job_id) is None


def _shown_rim_size(client: TestClient, job_id: int) -> str | None:
    """Rozmiar felgi w szczegółach zadania (wszystkie klucze `Job.params`); None = brak."""
    page = client.get(f"/jobs/{job_id}").text
    params = re.search(r"<dl[^>]*data-params.*?</dl>", page, re.S)
    if params is None:
        return None
    pairs = re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", params.group(0), re.S)
    return {label.strip(): value.strip() for label, value in pairs}.get(
        PARAM_LABELS[edit_params.RIM_SIZE]
    )


def _rim_size_choices(html: str) -> list[tuple[str, str, bool]]:
    """Wybór rozmiaru felgi: (wartość, etykieta, zaznaczony) każdego radia formularza zadania."""
    section = re.search(r'<fieldset[^>]*data-role="rim-size".*?</fieldset>', html, re.S)
    assert section, html
    choices = re.findall(
        r'<input type="radio" name="rim_size" value="([^"]+)" form="job-form"([^>]*)>\s*'
        r"<span[^>]*>([^<]*)</span>",
        section.group(0),
    )
    return [(value, label.strip(), "checked" in attrs) for value, attrs, label in choices]


def test_job_page_offers_rim_size_next_to_rim_choice_defaulting_to_unchanged(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    page = client.get(f"/jobs/{job_id}").text

    # W sekcji wyboru felgi, przed szczegółami zadania.
    assert (
        page.index('data-role="rim-choice"')
        < page.index('data-role="rim-size"')
        < page.index("data-params")
    )
    assert _rim_size_choices(page) == [
        ("-2", f"{MINUS}2″", False),
        ("-1", f"{MINUS}1″", False),
        ("0", "bez zmian", True),
        ("1", "+1″", False),
        ("2", "+2″", False),
    ]


def test_chosen_rim_size_stays_selected_when_generation_is_refused(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = upload_photo(client)  # bez felgi: „Generuj” zostanie odrzucone
    drain(session_factory, storage)
    click_wheel(client, job_id)
    assert drain(session_factory, storage)

    response = client.post(
        f"/jobs/{job_id}/generate", data={"rim_size": "-1"}, follow_redirects=False
    )

    assert response.status_code == 409
    assert [value for value, _, checked in _rim_size_choices(response.text) if checked] == ["-1"]


@pytest.mark.parametrize(("rim_size", "shown"), [("2", "Felga: +2″"), ("-1", f"Felga: {MINUS}1″")])
def test_result_screen_shows_chosen_rim_size_next_to_seed(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    rim_size: str,
    shown: str,
) -> None:
    _, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)
    _generate(client, job_id, rim_size)
    assert drain(session_factory, storage, processor)

    html = client.get(f"/jobs/{job_id}/status").text

    wheels = re.search(r'<ul[^>]*data-role="wheels".*?</ul>', html, re.S)
    assert wheels and 'data-role="wheel-seed"' in wheels.group(0)
    label = re.search(r'data-role="result-rim-size"[^>]*>([^<]*)<', wheels.group(0))
    assert label and label.group(1).strip() == shown
    assert "result-rim-size" not in client.get("/history").text  # siatka Historii bez zmian


@pytest.mark.parametrize("rim_size", ["0", None])
def test_result_screen_hides_unchanged_rim_size(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    rim_size: str | None,
) -> None:
    _, processor = _editor_processor()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)
    _generate(client, job_id, rim_size)
    assert drain(session_factory, storage, processor)

    html = client.get(f"/jobs/{job_id}/status").text

    assert 'data-role="wheel-seed"' in html and "result-rim-size" not in html
