"""Tryby pikseli zdjęć, dla których wynik zachowuje oryginał poza maską bit w bit.

Jedno źródło dla API (walidacja przy wgraniu, `wheeler_api.uploads`) i workera (odczyt
oryginału do kompozycji, `wheeler_worker.photos.load_original`). Wynik jest PNG w trybie
kompozycji: RGB, RGBA, L i LA bez zmian; paleta rozwinięta do RGB (z przezroczystością do
RGBA), obraz 1-bitowy do L, więc odczytane wartości pikseli poza maską są identyczne.
Tryby spoza listy (16 bitów na kanał, 32-bitowe, CMYK, LAB, ...) wymagałyby przeliczenia
kolorów albo utraty precyzji, więc ich nie przyjmujemy.
"""

from __future__ import annotations

from typing import Final

# Tryb Pillow wejścia -> tryb, w którym worker składa i zapisuje wynik.
EDIT_MODES: Final[dict[str, str]] = {
    "RGB": "RGB",
    "RGBA": "RGBA",
    "L": "L",
    "LA": "LA",
    "P": "RGB",
    "PA": "RGBA",
    "1": "L",
}


def edit_mode(mode: str, *, transparency: bool = False) -> str | None:
    """Tryb kompozycji dla obrazu w trybie `mode`; None, gdy nie da się go zachować wiernie.

    `transparency`: obraz ma przezroczystość jako kolor albo indeks palety (`info`), którą
    oddaje dopiero kanał alfa.
    """
    target = EDIT_MODES.get(mode)
    if target is not None and transparency and "A" not in target:
        return "RGBA"
    return target
