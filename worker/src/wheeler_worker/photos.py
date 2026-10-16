"""Odczyt zdjęć wejściowych i zapis wyników (jedno miejsce na formaty i metadane, ticket 10)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_photo(path: Path) -> Image.Image:
    """Zdjęcie jako RGB, obrócone według EXIF (współrzędne kliknięć i masek są w tym układzie)."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def save_png(image: Image.Image, path: Path) -> Path:
    """Zapis zawsze jako PNG (bezstratnie: piksele poza maską zostają bit w bit)."""
    image.save(path, format="PNG")
    return path
