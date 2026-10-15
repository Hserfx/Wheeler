"""Pliki na wolumenie danych (`./data`), wspólnym dla kontenera API i natywnego workera.

W bazie trzymamy wyłącznie ścieżki WZGLĘDNE wobec `root`, bo ten sam katalog jest widoczny
pod różnymi ścieżkami bezwzględnymi (`/data` w kontenerze, `./data` na hoście).
"""

from __future__ import annotations

import uuid
from pathlib import Path, PurePosixPath


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path(self, relative: str) -> Path:
        """Ścieżka bezwzględna dla ścieżki względnej z bazy; odrzuca wyjście poza `root`."""
        candidate = (self.root / PurePosixPath(relative)).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Ścieżka poza katalogiem danych: {relative!r}")
        return candidate

    def relative(self, absolute: Path) -> str:
        """Ścieżka względna (POSIX) do zapisania w bazie."""
        return absolute.resolve().relative_to(self.root).as_posix()

    def new_job_dir(self) -> str:
        """Tworzy pusty katalog na pliki nowego zadania i zwraca jego ścieżkę względną."""
        relative = f"jobs/{uuid.uuid4().hex}"
        self.path(relative).mkdir(parents=True, exist_ok=False)
        return relative

    def write_bytes(self, relative: str, data: bytes) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target
