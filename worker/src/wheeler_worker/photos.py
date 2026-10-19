"""Odczyt zdjęć wejściowych i zapis wyników (jedno miejsce na formaty i metadane, ticket 10).

Wejście: JPEG, PNG, WebP, HEIC (`pillow-heif`), obrócone według orientacji EXIF; współrzędne
kliknięć, masek i kompozycji są w układzie obróconego zdjęcia. Modele dostają RGB
(`load_photo`), a kompozycja oryginał w trybie, który wynik zachowuje bit w bit
(`load_original`, `wheeler_shared.photo_modes`). Wynik: PNG z profilem ICC
oryginału i jego EXIF bez lokalizacji GPS (orientacja usunięta, bo piksele są już obrócone).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pillow_heif
from PIL import ExifTags, Image, ImageOps

from wheeler_shared.failures import Failure, ProcessingFailed
from wheeler_shared.photo_modes import edit_mode

pillow_heif.register_heif_opener()

# Tagi usuwane z EXIF wyniku: lokalizacja i orientacja (piksele wyniku są już obrócone).
_DROPPED_EXIF_TAGS = (ExifTags.IFD.GPSInfo, ExifTags.Base.Orientation)


@dataclass(frozen=True)
class PhotoMetadata:
    """Metadane oryginału przenoszone do wyniku."""

    icc_profile: bytes | None = None
    exif: bytes | None = None


def load_photo(path: Path) -> Image.Image:
    """Zdjęcie jako RGB, obrócone według EXIF (współrzędne kliknięć i masek są w tym układzie)."""
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def load_original(path: Path) -> Image.Image:
    """Zdjęcie do kompozycji wyniku w trybie z `photo_modes.edit_mode` (RGB, RGBA, L, LA),
    obrócone według EXIF. Tryb, którego wynik nie zachowa wiernie: `ProcessingFailed`."""
    with Image.open(path) as source:
        mode = edit_mode(source.mode, transparency="transparency" in source.info)
        if mode is None:
            raise ProcessingFailed(Failure.UNSUPPORTED_PHOTO, f"tryb {source.mode}: {path}")
        image = ImageOps.exif_transpose(source)
    return image if image.mode == mode else image.convert(mode)


def read_metadata(path: Path) -> PhotoMetadata:
    """Profil ICC i EXIF zdjęcia do zapisania w wyniku (EXIF bez GPS i bez orientacji)."""
    with Image.open(path) as source:
        icc_profile = source.info.get("icc_profile") or None
        exif = source.getexif()
    for tag in _DROPPED_EXIF_TAGS:
        exif.pop(tag, None)
    return PhotoMetadata(icc_profile=icc_profile, exif=exif.tobytes() if len(exif) else None)


def save_png(image: Image.Image, path: Path, metadata: PhotoMetadata | None = None) -> Path:
    """Zapis zawsze jako PNG (bezstratnie: piksele poza maską zostają bit w bit).

    Bez `metadata` plik nie dostaje profilu ani EXIF (maski, nakładki, pliki robocze).
    """
    metadata = metadata or PhotoMetadata()
    options: dict[str, bytes] = {}
    if metadata.icc_profile:
        options["icc_profile"] = metadata.icc_profile
    if metadata.exif:
        options["exif"] = metadata.exif
    # Bez `info` obrazu: Pillow dopisałby z niego profil/EXIF wejścia także do masek.
    clean = image.copy()
    clean.info = {}
    clean.save(path, format="PNG", **options)
    return path
