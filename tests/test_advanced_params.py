"""Sekcja „Zaawansowane” na stronie głównej: seed, rozdzielczość robocza, poszerzenie maski
i dopisek do promptu. Wartości trafiają do zadania i są widoczne w jego szczegółach.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, finish_job, generate, jpeg, rim_png
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.app import create_app
from wheeler_api.settings import ApiSettings
from wheeler_api.texts import PARAM_LABELS, TEXTS
from wheeler_shared.storage import Storage
from wheeler_worker.runner import run_once
from wheeler_worker.wheel_swap import WheelSwapProcessor

RESOLUTION_ERROR = "Rozdzielczość robocza musi być wielokrotnością 32 px, od 768 do 1152 px."
MASK_GROW_ERROR = "Poszerzenie maski musi być liczbą od 0 do 10%."
SEED_ERROR = "Seed musi być liczbą całkowitą od 0 do 2147483647 albo pustym polem."


@pytest.fixture
def client(database_url: str, storage: Storage) -> Iterator[TestClient]:
    settings = ApiSettings(database_url=database_url, data_dir=storage.root)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _input_value(html: str, name: str) -> str:
    tag = re.search(rf'<(?:input|textarea)[^>]*name="{name}"[^>]*>', html)
    assert tag, html
    value = re.search(r'value="([^"]*)"', tag.group(0))
    return value.group(1) if value else ""


def test_advanced_section_is_collapsed_and_prefilled_with_spike_defaults(
    client: TestClient,
) -> None:
    html = client.get("/").text
    details = re.search(r"<details[^>]*data-role=\"advanced\"[^>]*>.*?</details>", html, re.S)
    assert details, html
    assert " open" not in details.group(0).split(">", 1)[0]
    section = details.group(0)
    assert _input_value(section, "seed") == ""
    assert _input_value(section, "resolution") == "1024"
    assert _input_value(section, "mask_grow") == "3"  # procent boku koła
    assert _input_value(section, "prompt_suffix") == ""


def _submit(client: TestClient, **fields: str) -> Any:  # odpowiedź TestClient (httpx/httpx2)
    return client.post(
        "/jobs",
        files={
            "photo": ("auto.jpg", jpeg(), "image/jpeg"),
            "rim": ("felga.png", rim_png(), "image/png"),
        },
        data=fields,
        follow_redirects=False,
    )


def _upload(client: TestClient, **fields: str) -> int:
    response = _submit(client, **fields)
    assert response.status_code == 303, response.text
    return int(response.headers["location"].rsplit("/", 1)[1])


def _shown_params(client: TestClient, job_id: int) -> dict[str, str]:
    """Parametry w szczegółach zadania: etykieta -> wyświetlona wartość."""
    page = client.get(f"/jobs/{job_id}").text
    params = re.search(r"<dl[^>]*data-params.*?</dl>", page, re.S)
    assert params, page
    pairs = re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", params.group(0), re.S)
    return {label.strip(): value.strip() for label, value in pairs}


def test_values_from_advanced_section_are_saved_and_shown_in_job_details(
    client: TestClient,
) -> None:
    job_id = _upload(
        client, seed="42", resolution="896", mask_grow="5", prompt_suffix="Chrome finish."
    )
    assert _shown_params(client, job_id) == {
        PARAM_LABELS["seed"]: "42",
        PARAM_LABELS["resolution"]: "896 px",
        PARAM_LABELS["mask_grow"]: "5%",
        PARAM_LABELS["prompt_suffix"]: "Chrome finish.",
    }


def test_untouched_section_saves_defaults_and_empty_seed_is_drawn_at_generation(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    # Przeglądarka wysyła pola zwiniętej sekcji z wartościami domyślnymi (seed pusty).
    job_id = _upload(client, seed="", resolution="1024", mask_grow="3", prompt_suffix="")
    assert _shown_params(client, job_id) == {
        PARAM_LABELS["resolution"]: "1024 px",
        PARAM_LABELS["mask_grow"]: "3%",
        PARAM_LABELS["prompt_suffix"]: TEXTS["param_empty"],
    }

    finish_job(client, session_factory, storage, job_id)
    shown = _shown_params(client, job_id)
    assert re.fullmatch(r"\d+", shown[PARAM_LABELS["seed"]]), shown
    assert shown[PARAM_LABELS["resolution"]] == "1024 px"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resolution", "1000", RESOLUTION_ERROR),  # nie jest wielokrotnością 32
        ("resolution", "1536", RESOLUTION_ERROR),  # swap przy q8 (spike 01)
        ("resolution", "512", RESOLUTION_ERROR),
        ("resolution", "duża", RESOLUTION_ERROR),
        ("mask_grow", "-1", MASK_GROW_ERROR),
        ("mask_grow", "12.5", MASK_GROW_ERROR),
        ("mask_grow", "nan", MASK_GROW_ERROR),
        ("seed", "-5", SEED_ERROR),
        ("seed", "1.5", SEED_ERROR),
        ("prompt_suffix", "x" * 301, "Dopisek do promptu może mieć najwyżej 300 znaków."),
    ],
)
def test_out_of_range_value_is_rejected_with_readable_message_and_kept_in_form(
    client: TestClient, field: str, value: str, message: str
) -> None:
    response = _submit(client, **{field: value})
    assert response.status_code == 400
    assert message in response.text
    details = re.search(r"<details[^>]*data-role=\"advanced\"[^>]*>", response.text)
    assert details and " open" in details.group(0)  # błąd jest widoczny bez szukania
    assert _input_value(response.text, field) == value
    assert "data-job-id" not in client.get("/history").text  # zadanie nie powstało


def test_mask_grow_accepts_decimal_comma_and_bounds(client: TestClient) -> None:
    for value, shown in [("2,5", "2.5%"), ("0", "0%"), ("10", "10%")]:
        job_id = _upload(client, mask_grow=value, resolution="1152")
        params = _shown_params(client, job_id)
        assert params[PARAM_LABELS["mask_grow"]] == shown
        assert params[PARAM_LABELS["resolution"]] == "1152 px"


def _mask_area(client: TestClient, job_id: int) -> int:
    status = client.get(f"/jobs/{job_id}/status").text
    src = re.search(r'<img[^>]+src="(/files/[^"]+)"[^>]*data-role="mask-overlay"', status)
    assert src, status
    overlay = Image.open(io.BytesIO(client.get(src.group(1)).content))
    return int((np.asarray(overlay)[..., 3] > 0).sum())


def test_worker_edits_with_values_from_the_section(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor = SolidEditor()
    processor = WheelSwapProcessor(
        detector=BoxDetector(), segmenter=DiscSegmenter(radius=15), editor=editor
    )
    areas = {}
    for grow in ("0", "10"):
        job_id = _upload(
            client, seed="99", resolution="768", mask_grow=grow, prompt_suffix="Gold rim."
        )
        drain(session_factory, storage, processor)  # detekcja kół (atrapa: brak)
        click_wheel(client, job_id)
        assert run_once(session_factory, storage, processor)
        areas[grow] = _mask_area(client, job_id)
        generate(client, job_id)
        assert run_once(session_factory, storage, processor)

    assert areas["10"] > areas["0"]  # poszerzenie maski działa już przy segmentacji
    for call in editor.calls:
        width, height = call.size
        assert abs(width * height - 768 * 768) / (768 * 768) < 0.15, call.size
        assert call.seed == 99
        assert call.prompt.endswith(" Gold rim.")


def test_chosen_seed_is_kept_at_generation(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    job_id = _upload(client, seed="7")
    finish_job(client, session_factory, storage, job_id)
    assert _shown_params(client, job_id)[PARAM_LABELS["seed"]] == "7"
