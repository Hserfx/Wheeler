"""Model danych: zadania, ich obrazy i koła (tylko ścieżki i metadane, pliki leżą w `Storage`).

Zadanie (`Job`) to jedna edycja jednego zdjęcia, widoczna w Historii. Przechodzi przez kroki,
z których część wykonuje worker (`TaskKind`), a część użytkownik (status `draft`):

    wgranie --> queued/running (detect) --> draft (maski wykrytych kół jako nakładki)
    draft --klik obok masek--> queued/running (segment) --> draft (nowe koło dodane)
    draft --klik w maskę--> draft (koło usunięte, bez workera)
    draft --„Generuj”----> queued/running (generate) --> done | failed | cancelled
    done --„Ponów to koło”--> queued/running (regenerate_wheel) --> done (błąd/anulowanie
        też wracają do done z poprzednim wynikiem)

`Job.task` to rodzaj ostatnio zleconej pracy workera; razem ze statusem mówi UI, co się dzieje.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from wheeler_shared.db import Base


class JobStatus(enum.StrEnum):
    DRAFT = "draft"  # czeka na użytkownika (wybór koła, akceptacja maski)
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"  # anulowane przez użytkownika (w kolejce albo w trakcie generowania)

    @property
    def finished(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def busy(self) -> bool:
        """Worker ma coś do zrobienia (UI odpytuje o postęp)."""
        return self in (JobStatus.QUEUED, JobStatus.RUNNING)


class TaskKind(enum.StrEnum):
    """Rodzaj pracy workera; każdy ma swoją metodę w `wheeler_worker.processing.Processor`."""

    SEGMENT = "segment"  # kliknięcia -> maski kół (SAM 2), interaktywne: mają pierwszeństwo
    DETECT = "detect"  # zdjęcie -> wszystkie widoczne koła (Florence-2 + SAM 2), interaktywne
    GENERATE = "generate"  # podmiana felg we wszystkich kołach (mflux)
    REGENERATE_WHEEL = "regenerate_wheel"  # jedno koło od nowa z nowym seedem (ticket 07)

    @property
    def generates(self) -> bool:
        """Praca modelu edycji (mflux): długa, w kolejce za pracą interaktywną, z postępem."""
        return self in (TaskKind.GENERATE, TaskKind.REGENERATE_WHEEL)


class WheelSource(enum.StrEnum):
    DETECTED = "detected"  # znalezione przez detektor po wgraniu zdjęcia
    CLICK = "click"  # dodane kliknięciem użytkownika


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
    # Liczba kół znalezionych przez detektor (None = detekcja się nie odbyła albo nie udała).
    wheels_detected: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Moment ostatniego wstawienia do kolejki: kolejność FIFO (zadanie może długo być szkicem).
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Postęp generowania (`wheeler_shared.progress.Progress.to_json`), pisany przez worker.
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # „Anuluj” w trakcie pracy workera: worker sprawdza flagę po każdym kroku i przed kołem.
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())

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
    source: Mapped[WheelSource | None] = mapped_column(_str_enum(WheelSource, "wheel_source"))
    # Kliknięcie w pikselach zdjęcia (po obrocie EXIF); tylko dla `WheelSource.CLICK`.
    click_x: Mapped[float | None] = mapped_column(Float)
    click_y: Mapped[float | None] = mapped_column(Float)
    # Ścieżki względne w katalogu zadania (`jobs/<hex>/wheels/<id>/…`).
    mask_path: Mapped[str | None] = mapped_column(String(512))  # L 0/255: obszar do zmiany
    overlay_path: Mapped[str | None] = mapped_column(String(512))  # RGBA nakładka dla UI
    # Prostokąt obejmujący maskę (piksele zdjęcia, prawy/dolny wyłącznie); None bez maski.
    box_left: Mapped[int | None] = mapped_column(Integer)
    box_top: Mapped[int | None] = mapped_column(Integer)
    box_right: Mapped[int | None] = mapped_column(Integer)
    box_bottom: Mapped[int | None] = mapped_column(Integer)
    # Seed, z którym koło jest w bieżącym wyniku, gdy różni się od seeda zadania („ponów to
    # koło”); None = seed zadania (`params["seed"]`).
    seed: Mapped[int | None] = mapped_column(Integer)
    # Seed zleconego ponowienia tego koła (przenoszony do `seed` po sukcesie); None = brak.
    pending_seed: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="wheels")

    @property
    def box(self) -> tuple[int, int, int, int] | None:
        if self.box_left is None or self.box_top is None:
            return None
        if self.box_right is None or self.box_bottom is None:
            return None
        return self.box_left, self.box_top, self.box_right, self.box_bottom

    @box.setter
    def box(self, value: tuple[int, int, int, int] | None) -> None:
        self.box_left, self.box_top, self.box_right, self.box_bottom = value or (None,) * 4


class RimStatus(enum.StrEnum):
    """Felga w bibliotece: worker usuwa tło, użytkownik akceptuje podgląd (`review` -> `ready`)."""

    QUEUED = "queued"  # czeka na usunięcie tła
    RUNNING = "running"
    REVIEW = "review"  # wycięta, czeka na akceptację albo odrzucenie
    READY = "ready"  # w bibliotece, do wyboru przy nowym zdjęciu
    FAILED = "failed"

    @property
    def busy(self) -> bool:
        return self in (RimStatus.QUEUED, RimStatus.RUNNING)


class Rim(Base):
    """Felga w bibliotece. Pliki w `rims/<hex>/`: oryginał i wycięta felga (RGBA PNG).

    Zadanie dostaje KOPIĘ wyciętej felgi w swoim katalogu (`rim_library.use_rim`), więc
    usunięcie felgi z biblioteki nie psuje historii.
    """

    __tablename__ = "rims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[RimStatus] = mapped_column(
        _str_enum(RimStatus, "rim_status"), default=RimStatus.QUEUED, index=True
    )
    name: Mapped[str | None] = mapped_column(String(200))
    original_path: Mapped[str] = mapped_column(String(512))
    cut_path: Mapped[str | None] = mapped_column(String(512))
    # Klik w felgę na oryginale (piksele po obrocie EXIF): wycięcie przez SAM 2 zamiast BiRefNet.
    click_x: Mapped[float | None] = mapped_column(Float)
    click_y: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    use_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
