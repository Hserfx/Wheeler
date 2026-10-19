"""Pliki na wolumenie danych (`./data`), wspólnym dla kontenera API i natywnego workera.

W bazie trzymamy wyłącznie ścieżki WZGLĘDNE wobec `root`, bo ten sam katalog jest widoczny
pod różnymi ścieżkami bezwzględnymi (`/data` w kontenerze, `./data` na hoście).
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath

JOBS_DIR = "jobs"  # zadania: `jobs/<hex>/` (wejście, felga, koła, wynik, miniatury)
RIMS_DIR = "rims"  # biblioteka felg: `rims/<hex>/` (oryginał, wycięta felga)
WHEELS_DIR = "wheels"  # w katalogu zadania: `wheels/<id koła>/` (maska, nakładka, wycinki)


def wheel_dir(wheel: int | str) -> str:
    """Katalog plików koła względem katalogu zadania (`wheels/<id>`; z detekcji
    `wheels/detected-<n>`, bo id koła powstaje dopiero przy zapisie)."""
    return f"{WHEELS_DIR}/{wheel}"


def _ignore_missing(function: Callable[..., object], path: str, exc: BaseException) -> None:
    if not isinstance(exc, FileNotFoundError):
        raise exc


def _owner_dir(relative: str, top: str) -> str | None:
    """`<top>/<hex>` dla pliku leżącego w takim katalogu (np. `jobs/ab12/input.jpg`)."""
    parts = PurePosixPath(relative).parts
    if len(parts) >= 3 and parts[0] == top and parts[1] not in ("", ".", ".."):
        return f"{top}/{parts[1]}"
    return None


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
        """Tworzy pusty katalog na pliki nowego zadania i zwraca jego ścieżkę względną.

        Wszystko, co należy do zadania (wejście, maski, wycinki, wynik, miniatury), trzymamy
        w tym katalogu: usunięcie zadania kasuje go w całości (`job_dir_of`, `remove_tree`).
        """
        return self._new_dir(JOBS_DIR)

    def new_rim_dir(self) -> str:
        """Tworzy pusty katalog na pliki nowej felgi z biblioteki (`rims/<hex>`)."""
        return self._new_dir(RIMS_DIR)

    @staticmethod
    def job_dir_of(relative: str) -> str | None:
        """Katalog zadania (`jobs/<id>`) zawierający plik albo None dla plików spoza zadań."""
        return _owner_dir(relative, JOBS_DIR)

    @staticmethod
    def rim_dir_of(relative: str) -> str | None:
        """Katalog felgi (`rims/<hex>`) zawierający plik albo None."""
        return _owner_dir(relative, RIMS_DIR)

    def _new_dir(self, top: str) -> str:
        relative = f"{top}/{uuid.uuid4().hex}"
        self.path(relative).mkdir(parents=True, exist_ok=False)
        return relative

    def remove_tree(self, relative: str) -> None:
        """Usuwa katalog z całą zawartością; brak katalogu nie jest błędem."""
        target = self.path(relative)
        if target == self.root:
            raise ValueError("Nie wolno usunąć całego katalogu danych")
        shutil.rmtree(target, ignore_errors=False, onexc=_ignore_missing)

    def write_bytes(self, relative: str, data: bytes) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target
