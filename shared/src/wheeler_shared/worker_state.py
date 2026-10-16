"""Stan procesu workera widoczny dla API (API nie widzi procesu workera, tylko bazę).

Stan modelu: worker ładuje modele leniwie przy pierwszym zadaniu i zwalnia je po czasie
bezczynności (`wheeler_worker.model_lifecycle`), a każdą zmianę zapisuje tutaj
(`set_model_state`). UI czyta go przez `get_model_state`. Jeden wiersz (`id = 1`), bo
worker jest jeden.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, Session, mapped_column

from wheeler_shared.db import Base

_ROW_ID = 1


class ModelState(enum.StrEnum):
    ASLEEP = "asleep"  # modele nie zajmują pamięci
    LOADING = "loading"  # wagi ładują się do pamięci (pierwsze zadanie po przerwie)
    READY = "ready"  # modele w pamięci, kolejne zadanie startuje od razu


class WorkerState(Base):
    __tablename__ = "worker_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_state: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def get_model_state(session: Session) -> ModelState:
    """Ostatni stan zapisany przez worker; bez wpisu (worker nie startował) = uśpiony."""
    value = session.scalar(select(WorkerState.model_state).where(WorkerState.id == _ROW_ID))
    try:
        return ModelState(value) if value is not None else ModelState.ASLEEP
    except ValueError:
        return ModelState.ASLEEP


def set_model_state(session: Session, state: ModelState) -> None:
    statement = insert(WorkerState).values(id=_ROW_ID, model_state=state.value)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=[WorkerState.id],
            set_={"model_state": state.value, "updated_at": func.now()},
        )
    )
