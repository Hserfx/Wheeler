"""Odczyt zdjęć wejściowych i zapis wyników (jedno miejsce na formaty i metadane, ticket 10).

Wejście: JPEG, PNG, WebP, HEIC (`pillow-heif`), obrócone według orientacji EXIF; współrzędne
kliknięć, masek i kompozycji są w układzie obróconego zdjęcia. Wynik: PNG z profilem ICC
oryginału i jego EXIF bez lokalizacji GPS (orientacja usunięta, bo piksele są już obrócone).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pillow_heif
from PIL import ExifTags, Image, ImageOps

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
