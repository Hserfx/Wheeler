"""Model danych: zadania, ich obrazy i koła (tylko ścieżki i metadane, pliki leżą w `Storage`).

Zadanie (`Job`) to jedna edycja jednego zdjęcia, widoczna w Historii. Przechodzi przez kroki,
z których część wykonuje worker (`TaskKind`), a część użytkownik (status `draft`):

    draft --klik w koło--> queued/running (segment) --> draft (maska jako nakładka)
    draft --„Generuj”----> queued/running (generate) --> done | failed

`Job.task` to rodzaj ostatnio zleconej pracy workera; razem ze statusem mówi UI, co się dzieje.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from wheeler_shared.db import Base


class JobStatus(enum.StrEnum):
    DRAFT = "draft"  # czeka na użytkownika (wybór koła, akceptacja maski)
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)

    @property
    def busy(self) -> bool:
        """Worker ma coś do zrobienia (UI odpytuje o postęp)."""
        return self in (JobStatus.QUEUED, JobStatus.RUNNING)


class TaskKind(enum.StrEnum):
    """Rodzaj pracy workera; każdy ma swoją metodę w `wheeler_worker.processing.Processor`."""

    SEGMENT = "segment"  # kliknięcia -> maski kół (SAM 2), interaktywne: mają pierwszeństwo
    GENERATE = "generate"  # podmiana felg we wszystkich kołach (mflux)


class ImageRole(enum.StrEnum):
    """Rola obrazu w zadaniu. Maski kół są w `Wheel`."""

    INPUT = "input"
    RIM = "rim"  # felga referencyjna (w 06 może leżeć w bibliotece, poza katalogiem zadania)
    RESULT = "result"


def _str_enum(enum_cls: type[enum.StrEnum], name: str) -> Enum:
    # Enum nienatywny (VARCHAR bez CHECK): nowe wartości nie wymagają migracji.
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=False,
        length=32,
        values_callable=lambda e: [m.value for m in e],
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[JobStatus] = mapped_column(
        _str_enum(JobStatus, "job_status"), default=JobStatus.DRAFT, index=True
    )
    task: Mapped[TaskKind | None] = mapped_column(_str_enum(TaskKind, "task_kind"))
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Moment ostatniego wstawienia do kolejki: kolejność FIFO (zadanie może długo być szkicem).
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    images: Mapped[list[Image]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Image.id"
    )
    wheels: Mapped[list[Wheel]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Wheel.id"
    )

    def image(self, role: ImageRole) -> Image | None:
        return next((img for img in self.images if img.role == role), None)

    @property
    def masked_wheels(self) -> list[Wheel]:
        return [wheel for wheel in self.wheels if wheel.mask_path is not None]


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    role: Mapped[ImageRole] = mapped_column(_str_enum(ImageRole, "image_role"))
    # Ścieżka względna wobec katalogu danych (`Storage.root`), zapis POSIX.
    path: Mapped[str] = mapped_column(String(512))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="images")


class Wheel(Base):
    """Koło wybrane do podmiany. Bez maski = czeka na segmentację przez worker."""

    __tablename__ = "wheels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    # Kliknięcie w pikselach zdjęcia (po obrocie EXIF); 04 doda koła z detektora (box).
    click_x: Mapped[float | None] = mapped_column(Float)
    click_y: Mapped[float | None] = mapped_column(Float)
    # Ścieżki względne w katalogu zadania (`jobs/<hex>/wheels/<id>/…`).
    mask_path: Mapped[str | None] = mapped_column(String(512))  # L 0/255: obszar do zmiany
    overlay_path: Mapped[str | None] = mapped_column(String(512))  # RGBA nakładka dla UI
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="wheels")
