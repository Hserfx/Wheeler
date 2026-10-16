"""Odczyt masek kół w API (czy klik trafił w maskę, `wheeler_shared.editing.click_photo`)."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from wheeler_shared.storage import Storage


def mask_contains(storage: Storage) -> Callable[[str, tuple[float, float]], bool]:
    """Funkcja: czy maska (ścieżka względna, L 0/255) ma niezerowy piksel w punkcie."""

    def contains(mask_path: str, point: tuple[float, float]) -> bool:
        x, y = int(point[0]), int(point[1])
        try:
            with Image.open(storage.path(mask_path)) as mask:
                if not (0 <= x < mask.width and 0 <= y < mask.height):
                    return False
                value = mask.convert("L").getpixel((x, y))
        except OSError:
            return False  # brak pliku maski: klik dodaje nowe koło
        return isinstance(value, int | float) and value > 0

    return contains
