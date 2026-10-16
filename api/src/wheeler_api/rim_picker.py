"""Kontekst fragmentu `_rim_picker.html` (biblioteka felg), wspólny dla strony głównej
(`mode="select"`, felgi do wyboru w formularzu zadania) i strony `/rims` (`mode="manage"`).
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy.orm import Session

from wheeler_shared.rim_library import library, pending_rims

Mode = Literal["select", "manage"]


def picker_context(
    session: Session, *, mode: Mode, selected: int | None = None, error: str | None = None
) -> dict[str, object]:
    pending = pending_rims(session)
    return {
        "mode": mode,
        "selected": selected,
        "library": library(session),
        "pending": pending,
        "polling": any(rim.status.busy for rim in pending),
        "rim_error": error,
    }
