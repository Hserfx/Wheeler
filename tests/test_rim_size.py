"""Rozmiar koła (tickety 12, 16 i 17): stopień -2…+2″ wybierany przed „Generuj”.

Całe koło (felga i zewnętrzna krawędź opony) rośnie albo maleje o tyle cali według skali
z „Obecnej felgi”, punkt styku z ziemią stoi, a karoseria zostaje bit w bit; odsłonięte przy
zmniejszaniu nadkole jest wstępnie wypełnione (ticket 17). Steruje tym geometria: worker
zmienia rozmiar koła w wycinku przed modelem (geometria: `test_wheel_resize.py`). Procesor
testowany bezpośrednio z atrapami SAM 2 i mflux (atrapa mflux oddaje wycinek, więc wynik
pokazuje koło, które dostał model), reszta przez HTTP i `run_once` workera na prawdziwym
Postgresie.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
from fake_models import BodyFake, BoxDetector, DiscSegmenter, SolidEditor, disc_mask, fender_mask
from fastapi.testclient import TestClient
from helpers import (
    click_wheel,
    drain,
    fetch,
    image_src,
    jpeg,
    upload,
    upload_photo,
    wheel_photo,
)
from PIL import Image
from sqlalchemy.orm import Session, sessionmaker

from wheeler_api.texts import PARAM_LABELS, TEXTS
from wheeler_shared import edit_params
from wheeler_shared.failures import Failure, ProcessingFailed
from wheeler_shared.storage import Storage
from wheeler_worker.body_mask import BodyMaskProblem
from wheeler_worker.processing import GenerateItem, RegenerateWheelItem, SegmentItem
from wheeler_worker.rim_geometry import Ellipse
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


# Koło na zdjęciu 400x240: opona o promieniu 50 px, w niej jasna felga o promieniu 35 px.
# Przy obecnej feldze 17″: 1″ = 70 px / 18,4 ≈ 3,8 px; +2″: felga 38,8 px, opona 53,8 px.
WHEEL = (150, 120)
TIRE_RADIUS = 50
RIM_RADIUS = 35


def _wheel_photo(path: Path) -> Path:
    rim = Ellipse(*WHEEL, width=2 * RIM_RADIUS, height=2 * RIM_RADIUS, angle=0)
    wheel_photo((400, 240), rim, TIRE_RADIUS / RIM_RADIUS).save(path)
    return path


class _Fakes(NamedTuple):
    """Procesor z atrapami: „SAM 2” widzi koło ze zdjęcia `_wheel_photo` (felga `rim_radius`),
    „mflux” oddaje wycinek."""

    segmenter: DiscSegmenter
    editor: SolidEditor
    processor: WheelSwapProcessor


def _fakes(rim_radius: int | None = RIM_RADIUS, body: BodyFake | None = None) -> _Fakes:
    segmenter = DiscSegmenter(radius=TIRE_RADIUS, rim_radius=rim_radius)
    if body is not None:
        segmenter.body = body
    editor = SolidEditor(echo=True)
    processor = WheelSwapProcessor(detector=BoxDetector(), segmenter=segmenter, editor=editor)
    return _Fakes(segmenter, editor, processor)


class _Measured(NamedTuple):
    """Koło na zdjęciu (piksele zdjęcia): promień i środek jasnej felgi, połowa poziomej
    szerokości koła w wierszu środka felgi i najniższy wiersz koła (styk z ziemią)."""

    rim_radius: float
    rim_cy: float
    half_width: float
    bottom: int


def _measure(image: Image.Image) -> _Measured:
    """Koło `_wheel_photo` na zdjęciu (np. wyniku, w którym „model” oddał wycinek): felga =
    jasne piksele blisko środka koła (w oponie), koło = szary obszar wokół felgi (opona,
    felga i rozmyta granica między nimi; zaszumione tło prawie nigdy nie jest szare), a jego
    brzeg = ostatni piksel ciemny jak opona. Wstępne wypełnienie odsłoniętego nadkola przy
    zmniejszaniu też jest szare (mediana szumu x 0,5, ok. 64), ale jaśniejsze od opony (30)."""
    rgb = np.asarray(image.convert("RGB")).astype(int)
    near = np.asarray(disc_mask(image.size, WHEEL, 45)) > 0
    # Próg w połowie między felgą (220) a oponą (30): rozmyta krawędź liczy się po połowie.
    rim = (rgb.mean(axis=-1) > 125) & near
    ys, _ = np.nonzero(rim)
    cy = float(ys.mean())
    gray = rgb.max(axis=-1) - rgb.min(axis=-1) <= 12
    # Próg w połowie między oponą (30) a wypełnieniem nadkola (ok. 64).
    tire = gray & (rgb.mean(axis=-1) < 47)
    x, y = WHEEL[0], round(cy)

    def edge(step_y: int, step_x: int) -> int:
        """Odległość od (x, y) do ostatniego piksela opony w szarym obszarze koła."""
        last = distance = 0
        while gray[y + (distance + 1) * step_y, x + (distance + 1) * step_x]:
            distance += 1
            if tire[y + distance * step_y, x + distance * step_x]:
                last = distance
        return last

    half_width = (edge(0, -1) + edge(0, 1) + 1) / 2
    return _Measured(float(np.sqrt(rim.sum() / np.pi)), cy, half_width, y + edge(1, 0))


def test_measure_reads_the_wheel_of_the_test_photo(tmp_path: Path) -> None:
    measured = _measure(Image.open(_wheel_photo(tmp_path / "input.png")))

    assert measured.rim_radius == pytest.approx(RIM_RADIUS, abs=0.5)
    assert measured.rim_cy == pytest.approx(WHEEL[1], abs=0.5)
    assert measured.half_width == pytest.approx(TIRE_RADIUS + 0.5, abs=1)
    assert measured.bottom == WHEEL[1] + TIRE_RADIUS


class _Wheel:
    """Zdjęcie z jednym kołem i jego maską, procesor z atrapami (`_fakes`)."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        rim_radius: int | None = RIM_RADIUS,
        body: BodyFake | None = None,
    ) -> None:
        self.out = tmp_path
        self.photo = _wheel_photo(tmp_path / "input.png")
        self.rim = tmp_path / "rim.png"
        Image.new("RGB", (60, 60), (200, 20, 20)).save(self.rim)
        self.segmenter, self.editor, self.processor = _fakes(rim_radius, body)
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

    def generate(self, rim_size: int, current_rim: int | None = None) -> Path:
        return self.processor.generate(
            GenerateItem(
                job_id=1,
                photo_path=self.photo,
                rim_path=self.rim,
                output_dir=self.out,
                params=_params(rim_size, current_rim),
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
                params=_params(rim_size, None),
                wheel_id=7,
                mask_path=self.mask,
                seed=6,
            )
        ).result_path


def _params(rim_size: int, current_rim: int | None) -> dict[str, int]:
    params = {"seed": 5, edit_params.RIM_SIZE: rim_size}
    if current_rim is not None:
        params[edit_params.CURRENT_RIM] = current_rim
    return params


@pytest.fixture(scope="module")
def unchanged(tmp_path_factory: pytest.TempPathFactory) -> _Measured:
    """Koło na wyniku przy „bez zmian”: odniesienie, które przeszło tę samą drogę (wycinek,
    skalowanie do rozdzielczości roboczej i z powrotem, kompozycja)."""
    wheel = _Wheel(tmp_path_factory.mktemp("unchanged"))
    return _measure(Image.open(wheel.generate(0)))


def _expected(unchanged: _Measured, rim_size: int, current_rim: int = 17) -> _Measured:
    """Koło o `rim_size` cali większe: 1″ = średnica felgi / (obecna + 1,4″); felga i opona
    rosną o tyle samo pikseli, styk z ziemią stoi, więc środek koła idzie w górę."""
    grow = rim_size / (current_rim + 1.4) * unchanged.rim_radius  # przyrost promienia
    return _Measured(
        rim_radius=unchanged.rim_radius + grow,
        rim_cy=unchanged.rim_cy - grow,
        half_width=unchanged.half_width + grow,
        bottom=unchanged.bottom,
    )


def _assert_wheel(found: _Measured, expected: _Measured) -> None:
    assert found.rim_radius == pytest.approx(expected.rim_radius, abs=0.5)
    assert found.rim_cy == pytest.approx(expected.rim_cy, abs=1)
    assert found.half_width == pytest.approx(expected.half_width, abs=1)
    assert found.bottom == pytest.approx(expected.bottom, abs=1)


@pytest.mark.parametrize("rim_size", [2, 1, -1, -2])
def test_whole_wheel_is_resized_with_ground_contact_kept_and_the_unchanged_prompt(
    tmp_path: Path, unchanged: _Measured, rim_size: int
) -> None:
    wheel = _Wheel(tmp_path)

    result = wheel.generate(rim_size)

    # „Model” oddaje wycinek, więc wynik pokazuje koło, które model dostał.
    _assert_wheel(_measure(Image.open(result)), _expected(unchanged, rim_size))
    [call] = wheel.editor.calls
    assert call.prompt == CURRENT_TEMPLATE
    # Felgę mierzy SAM 2 na oryginale, przed mflux.
    [measured] = wheel.segmenter.rim_photos
    assert np.array_equal(np.asarray(measured), np.asarray(Image.open(wheel.photo)))
    # Obrazy diagnostyczne w katalogu koła.
    assert (tmp_path / "wheels/7/rim_ellipse.png").is_file()
    assert (tmp_path / "wheels/7/warped.png").is_file()


def test_inch_scale_comes_from_the_current_rim(tmp_path: Path, unchanged: _Measured) -> None:
    # Ta sama felga na zdjęciu: przy obecnej 14″ cal to więcej pikseli niż przy 22″, więc
    # -2″ zmniejsza ją bardziej (o 13% zamiast 8,5% średnicy).
    for current_rim in (14, 22):
        (tmp_path / str(current_rim)).mkdir()
        wheel = _Wheel(tmp_path / str(current_rim))

        result = wheel.generate(-2, current_rim=current_rim)

        _assert_wheel(_measure(Image.open(result)), _expected(unchanged, -2, current_rim))


def _changed(before: Image.Image, after: Image.Image) -> npt.NDArray[np.bool_]:
    old = np.asarray(before.convert("RGB"))
    new = np.asarray(after.convert("RGB"))
    return np.asarray((old != new).any(axis=-1))


@pytest.mark.parametrize("rim_size", [2, -2])
def test_body_and_everything_outside_old_and_new_wheel_stay_bit_exact(
    tmp_path: Path, unchanged: _Measured, rim_size: int
) -> None:
    # „Model” maluje cały wycinek na zielono: zmienić się może tylko maska edycji, czyli stare
    # i nowe koło poszerzone o 3% bez karoserii (błotnik nad kołem, `fender_mask`).
    wheel = _Wheel(tmp_path)
    wheel.editor.echo = False

    result = wheel.generate(rim_size)

    with Image.open(result) as output, Image.open(wheel.photo) as original:
        changed = _changed(original, output)
    old_mask = np.asarray(Image.open(wheel.mask)) > 0
    new = _expected(unchanged, rim_size)
    # Nowa opona poszerzona o 3% boku koła, z zapasem 2 px na zaokrąglenia masek.
    grown_new = np.asarray(
        disc_mask((400, 240), (WHEEL[0], new.rim_cy), round(new.half_width * 1.06 + 2))
    )
    assert changed.any()
    assert not changed[~old_mask & ~(grown_new > 0)].any()
    body = np.asarray(fender_mask(_ungrown(wheel.mask))) > 0
    assert body.any() and not changed[body].any()
    if rim_size > 0:  # koło wychodzi poza zapisaną maskę, ale nie w karoserię
        assert changed[~old_mask].sum() > 100


def _ungrown(mask: Path) -> Image.Image:
    """Maska koła z „SAM 2” (dysk `TIRE_RADIUS`), na której atrapa liczy błotnik."""
    return disc_mask((400, 240), WHEEL, TIRE_RADIUS)


def test_model_crop_outside_the_edit_mask_is_the_original(tmp_path: Path) -> None:
    wheel = _Wheel(tmp_path)

    wheel.generate(2)

    [call] = wheel.editor.calls
    with Image.open(tmp_path / "wheels/7/crop.png") as plain_crop:
        plain = np.asarray(plain_crop.convert("RGB"))
    model_crop = np.asarray(call.images[0].convert("RGB"))
    edit = np.asarray(call.mask.convert("L")) > 0
    assert (~edit).any() and not np.array_equal(model_crop, plain)
    assert np.array_equal(model_crop[~edit], plain[~edit])
    # Wycinek wokół maski edycji: większe koło mieści się w nim z marginesem.
    rows, cols = np.nonzero(edit)
    assert rows.min() > 0 and cols.min() > 0
    assert rows.max() < edit.shape[0] - 1 and cols.max() < edit.shape[1] - 1


def test_sam_measures_the_rim_and_the_body_and_is_released_before_mflux_runs(
    tmp_path: Path,
) -> None:
    # SAM 2 (~3 GB) i mflux (szczyt ~27 GB) nie mieszczą się razem w 32 GB: pomiar felgi musi
    # się skończyć, a SAM zostać zwolniony, zanim ruszy model edycji.
    wheel = _Wheel(tmp_path)
    sam_at_mflux_steps: list[list[str]] = []
    wheel.editor.after_step = lambda step: sam_at_mflux_steps.append(list(wheel.segmenter.log))

    wheel.generate(1)

    assert sam_at_mflux_steps == [["rim", "body", "release"]]


def test_unchanged_rim_size_sends_the_plain_crop_without_measuring_the_rim(
    tmp_path: Path,
) -> None:
    wheel = _Wheel(tmp_path)

    wheel.generate(0)

    [call] = wheel.editor.calls
    with Image.open(tmp_path / "wheels/7/crop.png") as crop:
        assert np.array_equal(np.asarray(call.images[0]), np.asarray(crop))
    assert wheel.segmenter.rim_photos == wheel.segmenter.body_photos == []
    assert not (tmp_path / "wheels/7/rim_ellipse.png").exists()
    assert not (tmp_path / "wheels/7/body.png").exists()
    assert not (tmp_path / "wheels/7/warped.png").exists()


def test_regenerated_wheel_is_cropped_and_measured_from_the_original_not_resized_twice(
    tmp_path: Path, unchanged: _Measured
) -> None:
    wheel = _Wheel(tmp_path)
    first = wheel.generate(2)  # „model” oddaje wycinek: w wyniku koło jest już większe

    second = wheel.regenerate(first, 2)

    generated, regenerated = wheel.editor.calls
    assert np.array_equal(np.asarray(regenerated.images[0]), np.asarray(generated.images[0]))
    assert np.array_equal(np.asarray(regenerated.mask), np.asarray(generated.mask))
    _assert_wheel(_measure(Image.open(second)), _expected(unchanged, 2))
    original = np.asarray(Image.open(wheel.photo))
    measured = [*wheel.segmenter.rim_photos, *wheel.segmenter.body_photos]
    assert len(measured) == 4  # felga i karoseria przy generowaniu i przy „Ponów to koło”
    assert all(np.array_equal(np.asarray(photo), original) for photo in measured)


def test_regenerated_wheel_is_composited_on_the_current_result(tmp_path: Path) -> None:
    # Pierwszy wynik: zielone koło. „Ponów to koło” składa na nim, więc poza maską edycji
    # (i w karoserii) zostaje pierwszy wynik, nie oryginał.
    wheel = _Wheel(tmp_path)
    wheel.editor.echo = False
    first = wheel.generate(2)
    wheel.editor.color = (200, 10, 200)

    second = wheel.regenerate(first, 2)

    with Image.open(first) as before, Image.open(second) as after:
        changed = _changed(before, after)
    with Image.open(wheel.photo) as original, Image.open(first) as before:
        edited = _changed(original, before)
    # Zmienia się tylko obszar, który zmieniło już pierwsze generowanie (ta sama maska edycji).
    assert changed.any()
    assert not (changed & ~edited).any()
    assert not changed[np.asarray(fender_mask(_ungrown(wheel.mask))) > 0].any()


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


def test_good_body_mask_lets_the_resize_through_and_lands_in_the_wheel_diagnostics(
    tmp_path: Path,
) -> None:
    wheel = _Wheel(tmp_path)

    wheel.generate(2)

    assert len(wheel.editor.calls) == 1
    [measured] = wheel.segmenter.body_photos  # na oryginale, przed mflux
    assert np.array_equal(np.asarray(measured), np.asarray(Image.open(wheel.photo)))
    # Maska karoserii w wycinku koła (rozmiar wycinka sprzed skalowania, jak `rim_ellipse.png`):
    # błotnik nad kołem, bez pikseli koła.
    with Image.open(tmp_path / "wheels/7/body.png") as body_file:
        body = np.asarray(body_file.convert("L")) > 0
    with Image.open(tmp_path / "wheels/7/rim_ellipse.png") as outline:
        assert body.shape == (outline.height, outline.width)
    assert body[:10].all() and not body[-40:].any()


def _empty(photo: Image.Image, wheel: Image.Image) -> Image.Image:
    return Image.new("L", photo.size, 0)


def _car_with_wheels(photo: Image.Image, wheel: Image.Image) -> Image.Image:
    both = (np.asarray(fender_mask(wheel)) > 0) | (np.asarray(wheel) > 0)
    return Image.fromarray(both.astype(np.uint8) * 255)


def _everything_but_wheel(photo: Image.Image, wheel: Image.Image) -> Image.Image:
    return Image.fromarray(((np.asarray(wheel) == 0) * 255).astype(np.uint8))


def _sill(photo: Image.Image, wheel: Image.Image) -> Image.Image:
    sill = Image.new("L", photo.size, 0)
    sill.paste(255, (80, 175, 120, 195))  # w wycinku obok koła, nie błotnik nad nadkolem
    return sill


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (_empty, BodyMaskProblem.EMPTY),
        (_car_with_wheels, BodyMaskProblem.COVERS_WHEEL),
        (_everything_but_wheel, BodyMaskProblem.COVERS_CROP),
        (_sill, BodyMaskProblem.MISSES_ARCH),
    ],
)
def test_bad_body_mask_fails_the_job_before_the_model_runs(
    tmp_path: Path,
    body: Callable[[Image.Image, Image.Image], Image.Image],
    problem: BodyMaskProblem,
) -> None:
    wheel = _Wheel(tmp_path, body=body)

    with pytest.raises(ProcessingFailed) as failed:
        wheel.generate(1)

    assert failed.value.failure == Failure.RIM_SIZE_UNAVAILABLE
    assert problem.value in str(failed.value)  # powód w logu workera
    assert wheel.editor.calls == []


def test_tire_hidden_by_something_other_than_the_body_fails_the_job(tmp_path: Path) -> None:
    # „SAM 2” widzi felgę 45 px w kole 50 px: typowe k = 1,32 daje oponę 59 px, a pierścień
    # 50-59 px nie należy ani do koła, ani do karoserii (np. zderzak innego auta przed kołem).
    wheel = _Wheel(tmp_path, rim_radius=45)

    with pytest.raises(ProcessingFailed) as failed:
        wheel.generate(-1)

    assert failed.value.failure == Failure.RIM_SIZE_UNAVAILABLE
    assert BodyMaskProblem.FOREIGN_OCCLUSION.value in str(failed.value)
    assert wheel.editor.calls == []


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


def _result_wheel(client: TestClient, job_id: int) -> _Measured:
    """Koło na bieżącym wyniku zadania (ekran wyniku)."""
    html = client.get(f"/jobs/{job_id}/status").text
    return _measure(fetch(client, image_src(html, "result")))


def _generate(client: TestClient, job_id: int, rim_size: str | None) -> None:
    data = {} if rim_size is None else {"rim_size": rim_size}
    response = client.post(f"/jobs/{job_id}/generate", data=data, follow_redirects=False)
    assert response.status_code == 303, response.text


def test_chosen_rim_size_reaches_the_editor_also_when_regenerating_one_wheel(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    unchanged: _Measured,
) -> None:
    _, editor, processor = _fakes()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    _generate(client, job_id, "2")
    assert drain(session_factory, storage, processor)
    [wheel_id] = re.findall(r'<li[^>]*data-wheel-id="(\d+)"', client.get(f"/jobs/{job_id}").text)
    response = client.post(f"/jobs/{job_id}/wheels/{wheel_id}/regenerate", follow_redirects=False)
    assert response.status_code == 303, response.text
    assert drain(session_factory, storage, processor)

    generated, regenerated = editor.calls
    assert np.array_equal(np.asarray(regenerated.images[0]), np.asarray(generated.images[0]))
    _assert_wheel(_result_wheel(client, job_id), _expected(unchanged, 2))
    assert regenerated.prompt == generated.prompt == CURRENT_TEMPLATE
    assert regenerated.seed != generated.seed


def test_retried_job_edits_with_the_same_rim_size(
    client: TestClient,
    session_factory: sessionmaker[Session],
    storage: Storage,
    unchanged: _Measured,
) -> None:
    _, editor, processor = _fakes()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)
    _generate(client, job_id, "-2")
    assert client.post(f"/jobs/{job_id}/cancel").status_code == 200  # w kolejce: od razu

    response = client.post(f"/jobs/{job_id}/retry", follow_redirects=False)
    assert response.status_code == 303, response.text
    assert drain(session_factory, storage, processor)

    assert len(editor.calls) == 1
    retried_id = int(response.headers["location"].rsplit("/", 1)[1])
    _assert_wheel(_result_wheel(client, retried_id), _expected(unchanged, -2))
    assert _shown_rim_size(client, retried_id) == f"{MINUS}2″"


def test_unmeasurable_rim_fails_the_job_with_a_hint_to_generate_without_resizing(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    _, editor, processor = _fakes(rim_radius=None)  # SAM 2 nie widzi felgi
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
    processor = _fakes().processor
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
    _, editor, processor = _fakes()
    job_id = _draft_with_wheel(client, session_factory, storage, processor)

    _generate(client, job_id, rim_size)
    assert drain(session_factory, storage, processor)

    [call] = editor.calls
    assert call.prompt == CURRENT_TEMPLATE
    assert _shown_rim_size(client, job_id) == shown


def test_rim_size_is_saved_in_job_params_for_each_step(
    client: TestClient, session_factory: sessionmaker[Session], storage: Storage
) -> None:
    processor = _fakes().processor
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
    processor = _fakes().processor
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
    processor = _fakes().processor
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
    processor = _fakes().processor
    job_id = _draft_with_wheel(client, session_factory, storage, processor)
    _generate(client, job_id, rim_size)
    assert drain(session_factory, storage, processor)

    html = client.get(f"/jobs/{job_id}/status").text

    assert 'data-role="wheel-seed"' in html and "result-rim-size" not in html
