"""Pliki zdjęć z metadanymi jak z telefonu (orientacja EXIF, GPS, profil ICC), generowane w teście.

Zdjęcie „prosto” (`upright`) zapisujemy obrócone o 90° w lewo z orientacją EXIF 6
(„obróć o 90° w prawo do wyświetlenia”), tak jak robi to aparat trzymany pionowo.
"""

from __future__ import annotations

import io

import numpy as np
import pillow_heif
from PIL import Image, ImageCms

pillow_heif.register_heif_opener()

ORIENTATION = 0x0112
MAKE = 0x010F
GPS_IFD = 0x8825
ROTATE_RIGHT = 6  # orientacja EXIF: obraz zapisany obrócony o 90° w lewo
GPS = {1: "N", 2: (52.0, 13.0, 30.0), 3: "E", 4: (21.0, 0.0, 15.0)}


def display_profile() -> bytes:
    """Profil ICC „z aparatu” (bajty kopiujemy 1:1, więc wystarczy dowolny poprawny profil)."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    data = bytearray(profile.tobytes())
    data[4:8] = b"WHLR"  # inny „twórca” w nagłówku: profil nie jest domyślnym sRGB Pillow
    data[24:36] = bytes(12)  # data utworzenia: bez niej każde wywołanie daje te same bajty
    return bytes(data)


def phone_exif(orientation: int = ROTATE_RIGHT) -> Image.Exif:
    exif = Image.Exif()
    exif[ORIENTATION] = orientation
    exif[MAKE] = "Apple"
    exif.get_ifd(GPS_IFD).update(GPS)
    return exif


def quadrants(width: int = 64, height: int = 48) -> Image.Image:
    """Rozpoznawalna orientacja: lewa połowa czerwona, prawa niebieska, góra jaśniejsza."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, : width // 2] = (220, 0, 0)
    pixels[:, width // 2 :] = (0, 0, 220)
    pixels[: height // 2] += 30
    return Image.fromarray(pixels)


def phone_file(upright: Image.Image, format: str) -> bytes:
    """`upright` zapisane jak z telefonu: obrócone, z orientacją 6, GPS i profilem ICC."""
    stored = upright.transpose(Image.Transpose.ROTATE_90)
    buffer = io.BytesIO()
    options: dict[str, object] = {"quality": 95} if format == "HEIF" else {}
    stored.save(
        buffer,
        format=format,
        exif=phone_exif().tobytes(),
        icc_profile=display_profile(),
        **options,
    )
    return buffer.getvalue()


def phone_heic(upright: Image.Image) -> bytes:
    return phone_file(upright, "HEIF")
