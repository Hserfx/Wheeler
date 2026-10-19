"""Rozmiar felgi (plus-sizing, ticket 12): stopień -2…+2″ wybierany przed „Generuj”.

Zmienia się tylko średnica felgi, zewnętrzna średnica koła zostaje; steruje tym geometria:
worker przeskalowuje felgę w wycinku przed modelem (geometria warpu: `test_rim_geometry.py`).
Procesor testowany bezpośrednio z atrapami SAM 2 i mflux, reszta przez HTTP i `run_once`
workera na prawdziwym Postgresie.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fake_models import BoxDetector, DiscSegmenter, SolidEditor, disc_mask
from fastapi.testclient import TestClient
from helpers import click_wheel, drain, jpeg, upload, upload_photo
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import PARAM_LABELS, TEXTS
from wheeler_shared import edit_params
from wheeler_shared.failures import Failure, ProcessingFailed
from wheeler_shared.storage import Storage
from wheeler_worker.processing import GenerateItem, RegenerateWheelItem, SegmentItem
from wheeler_worker.wheel_swap import WheelSwapProcessor, build_prompt

MINUS = "\N{MINUS SIGN}"  # w UI stopnie ujemne mają minus typograficzny

# Szablon promptu sprzed ticketu 12 (DESIGN.md, spike 01): przy „bez zmian” bez zmian.
CURRENT_TEMPLATE = (
    "Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. "
    "Keep the tire, the car body, the wheel arch, the brake caliper position, the lighting, "
    "the shadows and the camera angle of image 1 unchanged. The new rim must follow the "
    "perspective of the wheel in image 1 and fill the tire opening exactly. Photorealistic."
)


def test_prompt_is_the_current_template_with_the_users_suffix() -> None:
    # Jeden prompt dla każdego stopnia: rozmiarem steruje geometria wycinka (model polecenie
    # rozmiaru ignoruje); stopień ≠ 0 sprawdzają testy procesora niżej.
    assert build_prompt() == CURRENT_TEMPLATE
    assert build_prompt("Chrome finish.") == f"{CURRENT_TEMPLATE} Chrome finish."


# Koło na zdjęciu 400x240: opona o promieniu 50 px, w niej jasna felga o promieniu 35 px
# (rho_f = 0,7 średnicy koła). +2″ = +8% średnicy koła: felga 39 px; -2″: 31 px.
WHEEL = (150, 120)
TIRE_RADIUS = 50
RIM_RADIUS = 35


def _wheel_photo(path: Path) -> Path:
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (240, 400, 3), dtype=np.uint8)
    pixels[np.asarray(disc_mask((400, 240), WHEEL, TIRE_RADIUS)) > 0] = (30, 30, 30)
    pixels[np.asarray(disc_mask((400, 240), WHEEL, RIM_RADIUS)) > 0] = (220, 220, 220)
    Image.fromarray(pixels).save(path)
    return path


def _model_rim_radius(crop: Image.Image) -> float:
    """Promień jasnej felgi w wycinku dla modelu, w pikselach zdjęcia.

    Wycinek: koło 107 px (maska poszerzona o 3%) + 25% marginesu = 160 px, skalowany do 1024."""
    scale = crop.width / 160
    lum = np.asarray(crop.convert("L"))
    near = np.asarray(disc_mask(crop.size, (crop.width / 2, crop.height / 2), round(45 * scale)))
    bright = (lum > 125) & (near > 0)
    return float(np.sqrt(bright.sum() / np.pi) / scale)


class _Wheel:
    """Zdjęcie z jednym kołem i jego maską, procesor z atrapami („SAM 2” widzi felgę 35 px)."""

    def __init__(self, tmp_path: Path, *, rim_radius: int | None = RIM_RADIUS) -> None:
        self.out = tmp_path
        self.photo = _wheel_photo(tmp_path / "input.png")
        self.rim = tmp_path / "rim.png"
        Image.new("RGB", (60, 60), (200, 20, 20)).save(self.rim)
        self.segmenter = DiscSegmenter(radius=TIRE_RADIUS, rim_radius=rim_radius)
        self.editor = SolidEditor(echo=True)
        self.processor = WheelSwapProcessor(
            detector=BoxDetector(), segmenter=self.segmenter, editor=self.editor
        )
        self.mask = (
            self.processor.segment(
                SegmentItem(
                    job_id=1,
                    photo_path=self.photo,
                    output_dir=tmp_path,
                    params={},
                    clicks={7: WHEEL},
                )
            )
            .masks[7]
            .mask
        )

    def generate(self, rim_size: int) -> Path:
        return self.processor.generate(
            GenerateItem(
                job_id=1,
                photo_path=self.photo,
                rim_path=self.rim,
                output_dir=self.out,
                params={"seed": 5, edit_params.RIM_SIZE: rim_size},
                masks={7: self.mask},
            )
        ).result_path

    def regenerate(self, result: Path, rim_size: int) -> Path:
        return self.processor.regenerate_wheel(
            RegenerateWheelItem(
                job_id=1,
                photo_path=self.photo,
                result_path=result,
                rim_path=self.rim,
                output_dir=self.out,
                params={"seed": 5, edit_params.RIM_SIZE: rim_size},
                wheel_id=7,
                mask_path=self.mask,
                seed=6,
            )
        ).result_path


@pytest.mark.parametrize(("rim_size", "radius"), [(2, 39), (1, 37), (-1, 33), (-2, 31)])
def test_model_gets_crop_with_rim_already_resized_and_the_unchanged_prompt(
    tmp_path: Path, rim_size: int, radius: int
) -> None:
    wheel = _Wheel(tmp_path)

    result = wheel.generate(rim_size)

    [call] = wheel.editor.calls
    assert _model_rim_radius(call.images[0]) == pytest.approx(radius, abs=1)
    assert call.prompt == CURRENT_TEMPLATE
    # Felgę mierzy SAM 2 na oryginale, przed mflux.
    [measured] = wheel.segmenter.rim_photos
    assert np.array_equal(np.asarray(measured), np.asarray(Image.open(wheel.photo)))
    # Piksele poza maską koła bit w bit.
    with Image.open(result) as output, Image.open(wheel.photo) as original:
        outside = np.asarray(Image.open(wheel.mask)) == 0
        assert np.array_equal(np.asarray(output)[outside], np.asarray(original)[outside])
    # Obrazy diagnostyczne w katalogu koła.
    assert (tmp_path / "wheels/7/rim_ellipse.png").is_file()
    assert (tmp_path / "wheels/7/warped.png").is_file()


def test_rim_is_resized_only_inside_the_wheel_mask_the_model_edits(tmp_path: Path) -> None:
    # „SAM 2” widzi felgę 45 px w kole 50 px: profil opony (1,11) poza zakresem, więc typowe
    # k = 1,32 i krawędź opony (59 px) wychodzi poza maskę koła (50 px + 3%). Poza maską model
    # nie może zobaczyć przeskalowanego boku opony, bo kompozycja i tak go odetnie (szew).
    wheel = _Wheel(tmp_path, rim_radius=45)

    wheel.generate(2)

    [call] = wheel.editor.calls
    with Image.open(tmp_path / "wheels/7/crop.png") as plain_crop:
        plain = np.asarray(plain_crop.convert("RGB"))
    model_crop = np.asarray(call.images[0].convert("RGB"))
    outside = np.asarray(call.mask.convert("L")) == 0
    assert outside.any() and not np.array_equal(model_crop, plain)
    assert np.array_equal(model_crop[outside], plain[outside])


def test_sam_measures_the_rim_and_is_released_before_mflux_runs(tmp_path: Path) -> None:
    # SAM 2 (~3 GB) i mflux (szczyt ~27 GB) nie mieszczą się razem w 32 GB: pomiar felgi musi
    # się skończyć, a SAM zostać zwolniony, zanim ruszy model edycji.
    wheel = _Wheel(tmp_path)
    sam_at_mflux_steps: list[list[str]] = []
    wheel.editor.after_step = lambda step: sam_at_mflux_steps.append(list(wheel.segmenter.log))

    wheel.generate(1)

    assert sam_at_mflux_steps == [["rim", "release"]]


def test_unchanged_rim_size_sends_the_plain_crop_without_measuring_the_rim(
    tmp_path: Path,
) -> None:
    wheel = _Wheel(tmp_path)

    wheel.generate(0)

    [call] = wheel.editor.calls
    assert _model_rim_radius(call.images[0]) == pytest.approx(RIM_RADIUS, abs=1)
    assert wheel.segmenter.rim_photos == []
    assert not (tmp_path / "wheels/7/rim_ellipse.png").exists()
    assert not (tmp_path / "wheels/7/warped.png").exists()


def test_regenerated_wheel_is_cropped_and_measured_from_the_original_not_resized_twice(
    tmp_path: Path,
) -> None:
    wheel = _Wheel(tmp_path)
    first = wheel.generate(2)  # „model” oddaje wycinek: w wyniku felga ma już 39 px

    second = wheel.regenerate(first, 2)

    generated, regenerated = wheel.editor.calls
    assert _model_rim_radius(regenerated.images[0]) == pytest.approx(39, abs=1)
    assert np.array_equal(np.asarray(regenerated.images[0]), np.asarray(generated.images[0]))
    original = np.asarray(Image.open(wheel.photo))
    assert all(np.array_equal(np.asarray(photo), original) for photo in wheel.segmenter.rim_photos)
    # Składanie na bieżącym wyniku: poza maską koła wynik zostaje jak był.
    outside = np.asarray(Image.open(wheel.mask)) == 0
    assert np.array_equal(
        np.asarray(Image.open(second))[outside], np.asarray(Image.open(first))[outside]
    )


@pytest.mark.parametrize(
    "rim_radius",
    [None, 15],  # SAM nie widzi felgi; „felga” ma 30% średnicy koła (poza 40-95%)
)
def test_unmeasurable_rim_fails_the_job_before_the_model_runs(
    tmp_path: Path, rim_radius: int | None
) -> None:
    wheel = _Wheel(tmp_path, rim_radius=rim_radius)

    with pytest.raises(ProcessingFailed) as failed:
        wheel.generate(1)

    assert failed.value.failure == Failure.RIM_SIZE_UNAVAILABLE
    assert wheel.editor.calls == []


def _editor_processor(
    rim_radius: int | None = RIM_RADIUS,
) -> tuple[SolidEditor, WheelSwapProcessor]:
    """Procesor z atrapami: „SAM 2” widzi koło ze zdjęcia `_wheel_photo`, „mflux” oddaje wycinek."""
    editor = SolidEditor(echo=True)
    segmenter = DiscSegmenter(radius=TIRE_RADIUS, rim_radius=rim_radius)
    return editor, WheelSwapProcessor(detector=BoxDetector(), segmenter=segmenter, editor=editor)


def _draft_with_wheel(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    processor: WheelSwapProcessor,
) -> int:
    """Szkic ze zdjęciem koła (`_wheel_photo`), felgą i maską koła, gotowy do „Generuj”."""
    with tempfile.TemporaryDirectory() as folder:
        photo = _wheel_photo(Path(folder) / "auto.png").read_bytes()
    job_id = upload(client, photo, filename="auto.png")
    drain(session_factory, storage, processor)
    click_wheel(client, job_id, WHEEL)
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
    assert _model_rim_radius(generated.images[0]) == pytest.approx(39, abs=1)
    assert _model_rim_radius(regenerated.images[0]) == pytest.approx(39, abs=1)
    assert regenerated.prompt == generated.prompt == CURRENT_TEMPLATE
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
    assert _model_rim_radius(call.images[0]) == pytest.approx(31, abs=1)
    retried_id = int(response.headers["location"].rsplit("/", 1)[1])
    assert _shown_rim_size(client, retried_id) == f"{MINUS}2″"


def test_unmeasurable_rim_fails_the_job_with_a_hint_to_generate_without_resizing(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    editor, processor = _editor_processor(rim_radius=None)  # SAM 2 nie widzi felgi
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    _generate(client, job_id, "1")
    assert drain(session_factory, storage, processor)

    status = client.get(f"/jobs/{job_id}/status").text
    assert TEXTS["status_failed"] in status
    assert TEXTS["failure_rim_size_unavailable"] in status
    assert "bez zmiany rozmiaru" in TEXTS["failure_rim_size_unavailable"]
    assert editor.calls == []


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
    assert f"Rozmiar felgi musi być stopniem od {MINUS}2″ do +2″." in response.text
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

    # Przy seedzie każdego koła (ta sama pozycja listy kół), nie osobno.
    wheels = re.findall(r'<li[^>]*data-wheel-id="\d+".*?</li>', html, re.S)
    assert wheels
    for wheel in wheels:
        assert 'data-role="wheel-seed"' in wheel
        label = re.search(r'data-role="result-rim-size"[^>]*>([^<]*)<', wheel)
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
