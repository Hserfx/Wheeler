"""Cykl życia modelu w workerze: uśpiony -> ładowanie -> gotowy -> (bezczynność) -> uśpiony.

Czas bezczynności liczymy na wstrzykiwanym zegarze, bez czekania prawdziwych minut.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from wheeler_shared.worker_state import ModelState
from wheeler_worker.model_lifecycle import ModelLifecycle
from wheeler_worker.processing import SegmentItem, build_processor
from wheeler_worker.settings import WorkerSettings


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class Released:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def state(lifecycle: ModelLifecycle) -> ModelState:
    # Funkcja, a nie odczyt właściwości wprost: mypy nie zawęża typu między asercjami.
    return lifecycle.state


def make(timeout: float = 600.0) -> tuple[ModelLifecycle, list[ModelState], FakeClock]:
    reported: list[ModelState] = []
    clock = FakeClock()
    lifecycle = ModelLifecycle(idle_timeout=timeout, report=reported.append, clock=clock)
    return lifecycle, reported, clock


def test_fresh_worker_reports_model_asleep() -> None:
    lifecycle, reported, _ = make()
    assert state(lifecycle) is ModelState.ASLEEP
    assert reported == [ModelState.ASLEEP]


def test_loading_a_model_reports_loading_then_ready() -> None:
    lifecycle, reported, _ = make()
    with lifecycle.loading():
        assert state(lifecycle) is ModelState.LOADING
    assert state(lifecycle) is ModelState.READY
    assert reported == [ModelState.ASLEEP, ModelState.LOADING, ModelState.READY]


def test_models_are_released_only_after_idle_timeout_since_last_task() -> None:
    lifecycle, reported, clock = make(timeout=600.0)
    release = Released()
    with lifecycle.loading():
        pass
    clock.now += 500.0
    lifecycle.mark_used()  # kolejne zadanie z modelem w pamięci przesuwa początek bezczynności

    clock.now += 599.0
    assert not lifecycle.release_if_idle(release)
    assert state(lifecycle) is ModelState.READY and release.count == 0

    clock.now += 1.0
    assert lifecycle.release_if_idle(release)
    assert state(lifecycle) is ModelState.ASLEEP and release.count == 1
    assert reported[-1] is ModelState.ASLEEP


def test_asleep_models_are_not_released_again() -> None:
    lifecycle, reported, clock = make(timeout=1.0)
    release = Released()
    clock.now += 3600.0
    assert not lifecycle.release_if_idle(release)
    assert release.count == 0
    assert reported == [ModelState.ASLEEP]


def test_failed_load_does_not_leave_state_loading() -> None:
    lifecycle, reported, _ = make()
    with pytest.raises(FileNotFoundError), lifecycle.loading():
        raise FileNotFoundError("brak wag")
    assert state(lifecycle) is ModelState.ASLEEP
    assert reported == [ModelState.ASLEEP, ModelState.LOADING, ModelState.ASLEEP]

    with lifecycle.loading():
        pass
    with pytest.raises(FileNotFoundError), lifecycle.loading():  # np. drugi model po SAM 2
        raise FileNotFoundError("brak wag")
    assert state(lifecycle) is ModelState.READY


def test_idle_timeout_defaults_to_ten_minutes_and_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WHEELER_MODEL_IDLE_TIMEOUT", raising=False)
    assert WorkerSettings().model_idle_timeout == 600.0
    monkeypatch.setenv("WHEELER_MODEL_IDLE_TIMEOUT", "5")
    assert WorkerSettings().model_idle_timeout == 5.0


def test_placeholder_models_load_through_lifecycle(tmp_path: Path) -> None:
    lifecycle, reported, _ = make()
    processor = build_processor(WorkerSettings(processor="placeholder"), lifecycle)
    photo = tmp_path / "input.png"
    Image.new("RGB", (64, 48), (90, 90, 90)).save(photo)
    processor.segment(
        SegmentItem(
            job_id=1, photo_path=photo, output_dir=tmp_path, params={}, clicks={1: (30, 20)}
        )
    )
    assert state(lifecycle) is ModelState.READY
    lifecycle.release(processor.release)
    assert reported == [ModelState.ASLEEP, ModelState.LOADING, ModelState.READY, ModelState.ASLEEP]
