"""Wycięta felga: z maski pierwszego planu do RGBA i z RGBA do obrazu referencyjnego modelu."""

from __future__ import annotations

from PIL import Image

CUT_MAX_SIDE = 1536  # większa felga nic nie daje modelowi (rozdzielczość robocza 1024-1536 px)
ALPHA_THRESHOLD = 8  # piksele o mniejszej alfie nie wyznaczają kadru felgi
REFERENCE_BACKGROUND = (255, 255, 255)  # tło, na którym model widzi wyciętą felgę


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


def reference(rim: Image.Image) -> Image.Image:
    """Felga dla modelu jako RGB: przezroczyste tło wycięcia zastąpione jednolitym."""
    if "A" not in rim.getbands():
        return rim.convert("RGB")
    background = Image.new("RGB", rim.size, REFERENCE_BACKGROUND)
    background.paste(rim.convert("RGBA"), mask=rim.getchannel("A"))
    return background
