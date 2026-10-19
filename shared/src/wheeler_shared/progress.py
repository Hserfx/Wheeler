"""Postęp generowania: które koło i który krok denoisingu (zapisywany w `Job.progress`).

Worker raportuje go po każdym kroku modelu i przed każdym kołem; API pokazuje go w UI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Progress:
    wheel: int  # bieżące koło, od 1
    wheels: int  # liczba kół w zadaniu
    step: int = 0  # ukończone kroki denoisingu bieżącego koła
    steps: int | None = None  # liczba kroków (None: model jeszcze nie zaczął pętli)

    @property
    def fraction(self) -> float:
        """Postęp całego zadania, 0..1 (koła równo ważone)."""
        within = self.step / self.steps if self.steps else 0.0
        return min(1.0, (self.wheel - 1 + within) / max(1, self.wheels))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> Progress | None:
        if not data:
            return None
        steps = data.get("steps")
        return cls(
            wheel=int(data["wheel"]),
            wheels=int(data["wheels"]),
            step=int(data.get("step", 0)),
            steps=None if steps is None else int(steps),
        )
