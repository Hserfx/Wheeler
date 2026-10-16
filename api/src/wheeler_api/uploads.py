"""Wgrywanie zdjęć: walidacja formatu i limitów, zapis, podgląd dla przeglądarki (ticket 10).

Jedno miejsce dla każdego wgrywanego obrazu (zdjęcie auta, felga, biblioteka felg):
`save_image_upload` przyjmuje JPEG, PNG, WebP i HEIC do 50 MB i 40 MPx, a inne pliki
odrzuca wyjątkiem `UploadRejected` z komunikatem dla użytkownika. Format rozpoznajemy po
zawartości, nie po nazwie; rozszerzenie zapisanego pliku wynika z formatu.

Podgląd: przeglądarka nie wyświetli HEIC, a orientację EXIF obsługuje różnie w różnych
formatach. Dla takich plików obok oryginału powstaje `<nazwa>.preview.jpg`: obrócony według
EXIF, w PEŁNYM rozmiarze (wybór koła liczy kliknięcia w pikselach `naturalWidth/Height`
obrazka, a worker w układzie obróconego zdjęcia). Szablony biorą adres przez `display_path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Final

import pillow_heif
from fastapi import UploadFile
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

from wheeler_api.texts import TEXTS
from wheeler_shared.storage import Storage

pillow_heif.register_heif_opener()

MAX_UPLOAD_MB: Final = 50
MAX_UPLOAD_BYTES: Final = MAX_UPLOAD_MB * 1024 * 1024
MAX_UPLOAD_MEGAPIXELS: Final = 40
MAX_UPLOAD_PIXELS: Final = MAX_UPLOAD_MEGAPIXELS * 1_000_000

# Format Pillow -> rozszerzenie zapisanego pliku.
ACCEPTED_FORMATS: Final[dict[str, str]] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "HEIF": ".heic",
}
# Atrybut `accept` pola pliku (HEIC jawnie: nie każda przeglądarka zalicza go do `image/*`).
UPLOAD_ACCEPT: Final = "image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif"

PREVIEW_SUFFIX: Final = ".preview.jpg"
_CHUNK = 1024 * 1024


class UploadRejected(Exception):
    """Plik nie spełnia wymagań; `message` jest gotowym komunikatem dla użytkownika."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def save_image_upload(storage: Storage, upload: UploadFile, stem: str) -> str:
    """Sprawdza i zapisuje wgrany obraz jako `<stem><rozszerzenie>`; zwraca ścieżkę względną.

    Rzuca `UploadRejected`; nic wtedy nie zostaje na dysku.
    """
    name = upload.filename or ""
    temporary = storage.path(f"{stem}.upload")
    try:
        with temporary.open("wb") as target:
            if not _copy_limited(upload.file, target, MAX_UPLOAD_BYTES):
                raise UploadRejected(
                    TEXTS["error_upload_too_large"].format(name=name, limit=MAX_UPLOAD_MB)
                )
        suffix, needs_preview = _inspect(temporary, name)
        relative = f"{stem}{suffix}"
        temporary.replace(storage.path(relative))
    finally:
        temporary.unlink(missing_ok=True)
    if needs_preview:
        try:
            _write_preview(storage.path(relative))
        except OSError:
            storage.path(relative).unlink(missing_ok=True)
            raise UploadRejected(TEXTS["error_upload_format"].format(name=name)) from None
    return relative


def display_path(storage: Storage, path: str) -> str:
    """Ścieżka obrazu do wyświetlenia w przeglądarce: podgląd, jeśli istnieje, inaczej oryginał."""
    preview = f"{path}{PREVIEW_SUFFIX}"
    return preview if storage.path(preview).is_file() else path


def _copy_limited(source: BinaryIO, target: BinaryIO, limit: int) -> bool:
    """Kopiuje `source` do `target`; `False`, gdy plik przekracza `limit` bajtów."""
    copied = 0
    while chunk := source.read(_CHUNK):
        copied += len(chunk)
        if copied > limit:
            return False
        target.write(chunk)
    return True


def _inspect(path: Path, name: str) -> tuple[str, bool]:
    """(rozszerzenie, czy potrzebny podgląd); czyta tylko nagłówek, bez dekodowania pikseli."""
    try:
        with Image.open(path) as image:
            image_format = image.format or ""
            pixels = image.width * image.height
            orientation = image.getexif().get(ExifTags.Base.Orientation, 1)
    except Image.DecompressionBombError:
        pixels = MAX_UPLOAD_PIXELS + 1
        image_format = ""
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadRejected(TEXTS["error_upload_format"].format(name=name)) from None
    if pixels > MAX_UPLOAD_PIXELS:
        raise UploadRejected(
            TEXTS["error_upload_too_many_pixels"].format(name=name, limit=MAX_UPLOAD_MEGAPIXELS)
        )
    if image_format not in ACCEPTED_FORMATS:
        raise UploadRejected(TEXTS["error_upload_format"].format(name=name))
    # HEIC: przeglądarki go nie wyświetlają; pillow-heif sam obraca i zeruje orientację.
    return ACCEPTED_FORMATS[image_format], image_format == "HEIF" or orientation != 1


def _write_preview(source: Path) -> None:
    target = source.with_name(f"{source.name}{PREVIEW_SUFFIX}")
    with Image.open(source) as opened:
        icc_profile = opened.info.get("icc_profile")
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.save(target, format="JPEG", quality=90, icc_profile=icc_profile)
