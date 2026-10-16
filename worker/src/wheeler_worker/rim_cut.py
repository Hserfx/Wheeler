"""Wycięta felga: z maski pierwszego planu do RGBA i z RGBA do obrazu referencyjnego modelu."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageOps

BoxF = tuple[float, float, float, float]  # x0, y0, x1, y1 (jak `detector.BoxF`)

CUT_MAX_SIDE = 1536  # większa felga nic nie daje modelowi (rozdzielczość robocza 1024-1536 px)
ALPHA_THRESHOLD = 8  # piksele o mniejszej alfie nie wyznaczają kadru felgi
REFERENCE_BACKGROUND = (255, 255, 255)  # tło, na którym model widzi wyciętą felgę
RIM_REGION_MARGIN = 0.08  # kadr dla BiRefNet: box koła z detektora + margines (ułamek boku)


def pick_rim_box(boxes: Sequence[BoxF], click: tuple[float, float] | None) -> BoxF | None:
    """Koło, które wycinamy: najmniejsze zawierające kliknięcie, bez kliknięcia największe.

    Pomijamy boxy obejmujące inne wykryte koło: Florence-2 zwraca czasem jako „wheel” także
    fragment auta razem z kołem (zmierzone na `rim_mesh_silver_angle.jpg`).
    """
    boxes = [b for b in boxes if not any(o is not b and _encloses(b, o) for o in boxes)]
    if click is not None:
        x, y = click
        boxes = [b for b in boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        return min(boxes, key=_area, default=None)
    return max(boxes, key=_area, default=None)


def _encloses(outer: BoxF, inner: BoxF) -> bool:
    """`inner` leży (w 90%) wewnątrz większego `outer`."""
    width = min(outer[2], inner[2]) - max(outer[0], inner[0])
    height = min(outer[3], inner[3]) - max(outer[1], inner[1])
    overlap = max(0.0, width) * max(0.0, height)
    return _area(outer) > _area(inner) and overlap >= 0.9 * _area(inner)


def rim_region(box: BoxF, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Kadr wokół koła (z marginesem, przycięty do zdjęcia) w pikselach całkowitych."""
    x0, y0, x1, y1 = box
    margin = max(x1 - x0, y1 - y0) * RIM_REGION_MARGIN
    width, height = size
    return (
        max(0, int(x0 - margin)),
        max(0, int(y0 - margin)),
        min(width, round(x1 + margin)),
        min(height, round(y1 + margin)),
    )


def _area(box: BoxF) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def cut_out(photo: Image.Image, alpha: Image.Image) -> Image.Image | None:
    """Felga jako RGBA (alfa = maska), przycięta do pierwszego planu; None, gdy maska pusta."""
    alpha = alpha.convert("L")
    box = alpha.point(lambda v: 255 if v >= ALPHA_THRESHOLD else 0).getbbox()
    if box is None:
        return None
    cut = photo.convert("RGB").crop(box)
    cut.putalpha(alpha.crop(box))
    if max(cut.size) > CUT_MAX_SIDE:
        cut.thumbnail((CUT_MAX_SIDE, CUT_MAX_SIDE), Image.Resampling.LANCZOS)
    return cut


def load_reference(path: Path) -> Image.Image:
    """Felga zadania dla modelu (wycięta z biblioteki albo zwykłe zdjęcie) jako RGB."""
    with Image.open(path) as source:
        return reference(ImageOps.exif_transpose(source))


def reference(rim: Image.Image) -> Image.Image:
    """Felga dla modelu jako RGB: przezroczyste tło wycięcia zastąpione jednolitym."""
    if rim.mode == "P" and "transparency" in rim.info:
        rim = rim.convert("RGBA")
    if "A" not in rim.getbands():
        return rim.convert("RGB")
    background = Image.new("RGB", rim.size, REFERENCE_BACKGROUND)
    background.paste(rim.convert("RGBA"), mask=rim.getchannel("A"))
    return background
