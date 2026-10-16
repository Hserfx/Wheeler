"""Miniatury do Historii, generowane leniwie przy pierwszym żądaniu.

Miniatura leży obok obrazu źródłowego, w katalogu zadania (`<nazwa>.thumb.jpg`), więc
usunięcie zadania kasuje ją razem z resztą plików. Jest odtwarzana, gdy źródło jest nowsze
(np. po ponownym wygenerowaniu wyniku).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from PIL import Image, ImageOps

THUMBNAIL_MAX_SIDE = 480


def ensure_thumbnail(source: Path) -> Path:
    """Zwraca ścieżkę miniatury JPEG dla `source`, tworząc ją w razie potrzeby.

    Rzuca `OSError` (w tym `PIL.UnidentifiedImageError`), gdy źródła nie da się odczytać.
    """
    target = source.with_name(f"{source.name}.thumb.jpg")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE))
        rgb = image.convert("RGB")
    # Zapis do pliku tymczasowego i podmiana: równoległe żądania nie widzą połowy pliku.
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        rgb.save(temporary, format="JPEG", quality=85)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
