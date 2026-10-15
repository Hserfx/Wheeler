"""Model danych: zadania w kolejce i obrazy (tylko ścieżki i metadane, pliki leżą w `Storage`)."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from wheeler_shared.db import Base


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


class ImageRole(enum.StrEnum):
    """Rola obrazu w zadaniu. Kolejne tickety dopiszą np. maski i wycinki kół."""

    INPUT = "input"
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
        _str_enum(JobStatus, "job_status"), default=JobStatus.QUEUED, index=True
    )
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    images: Mapped[list[Image]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Image.id"
    )

    def image(self, role: ImageRole) -> Image | None:
        return next((img for img in self.images if img.role == role), None)


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
