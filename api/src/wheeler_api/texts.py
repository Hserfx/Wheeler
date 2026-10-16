"""Wszystkie teksty interfejsu (po polsku) w jednym miejscu.

Szablony odwołują się do nich przez `t.<klucz>`; w szablonach nie wpisujemy tekstów na sztywno.
"""

from typing import Final

from wheeler_shared.models import Job, JobStatus, TaskKind
from wheeler_shared.progress import Progress

TEXTS: Final[dict[str, str]] = {
    "app_name": "Wheeler",
    "app_tagline": "Podmiana felg na zdjęciu samochodu",
    "nav_home": "Nowe zdjęcie",
    "nav_history": "Historia",
    # Historia
    "history_heading": "Historia",
    "history_empty": "Nie ma jeszcze żadnych zadań.",
    "history_job_label": "Zadanie nr {id}",
    "history_thumbnail_alt": "Miniatura zadania nr {id}",
    "history_select_job": "Zaznacz zadanie nr {id}",
    "history_select_all": "Zaznacz wszystkie",
    "history_select_none": "Odznacz wszystkie",
    "history_delete_selected": "Usuń zaznaczone",
    # Usuwanie
    "delete_confirm_heading": "Usunąć zadania?",
    "delete_confirm_hint": (
        "Zadania razem ze wszystkimi plikami (zdjęcie, maski, wynik) zostaną trwale "
        "usunięte z dysku. Tej operacji nie można cofnąć."
    ),
    "delete_confirm_count": "Do usunięcia: {count}",
    "delete_confirm_submit": "Usuń na stałe",
    "delete_cancel": "Anuluj",
    "delete_running_skipped": "Zadania w trakcie przetwarzania nie zostaną usunięte.",
    "delete_nothing": "Żadnego z wybranych zadań nie można teraz usunąć.",
    "delete_done": "Usunięto zadania: {count}.",
    "delete_skipped": "Pominięto zadania w trakcie przetwarzania: {count}.",
    # Strona główna
    "upload_heading": "Wgraj zdjęcie samochodu i felgi",
    "upload_hint": "Pliki JPEG, PNG lub WebP. Koło do podmiany wskażesz w następnym kroku.",
    "upload_photo_label": "Zdjęcie samochodu",
    "upload_rim_label": "Zdjęcie felgi",
    "upload_choose": "Wybierz plik",
    "upload_no_file_chosen": "Nie wybrano pliku",
    "upload_submit": "Dalej",
    "upload_sending": "Wysyłanie…",
    # Wybór koła i generowanie
    "wheel_click_hint": "Kliknij koło, w którym chcesz podmienić felgę.",
    "wheel_mask_hint": (
        "Zaznaczony obszar zostanie wygenerowany na nowo, reszta zdjęcia się nie zmieni. "
        "Kliknij inne miejsce, aby wybrać koło ponownie."
    ),
    "wheel_picker_alt": "Zdjęcie samochodu: kliknij koło",
    "wheel_mask_alt": "Maska wybranego koła",
    "generate_submit": "Generuj",
    "generate_hint": "Generowanie jednego koła trwa około 2 minut.",
    "job_rim": "Felga",
    # Zadanie
    "job_heading": "Zadanie nr {id}",
    "job_input": "Zdjęcie wejściowe",
    "job_result": "Wynik",
    "job_download": "Pobierz PNG",
    "job_error": "Błąd",
    "job_new": "Wgraj kolejne zdjęcie",
    "job_delete": "Usuń zadanie",
    "job_params": "Parametry",
    "job_params_default": "Użyto parametrów domyślnych.",
    "job_created": "Utworzono",
    "job_finished": "Zakończono",
    "status_label": "Status",
    "status_draft": "Wybór koła",
    "status_segmenting": "Wyznaczanie maski koła…",
    "status_queued": "W kolejce",
    "status_running": "Przetwarzanie…",
    "status_done": "Gotowe",
    "status_failed": "Niepowodzenie",
    # Postęp, anulowanie, kolejka, ponawianie (ticket 05)
    "progress_wheel": "Koło {wheel} z {wheels}: przygotowanie modelu…",
    "progress_step": "Koło {wheel} z {wheels}, krok {step} z {steps}",
    "status_cancelling": "Anulowanie…",
    "status_cancelled": "Anulowane",
    "job_cancel": "Anuluj",
    "queue_position": "Pozycja w kolejce: {position}",
    "job_retry": "Ponów",
    "job_retry_hint": "Nowe zadanie z tym samym zdjęciem, felgą, kołami i parametrami.",
    # Błędy
    "error_no_photo": "Wybierz zdjęcie samochodu.",
    "error_no_rim": "Wybierz zdjęcie felgi.",
    "error_no_wheel": "Najpierw kliknij koło i poczekaj na jego maskę.",
    "error_job_busy": "Zadanie jest teraz przetwarzane albo już zakończone.",
    "error_mask_failed": "Nie udało się zaznaczyć koła w tym miejscu. Kliknij bliżej środka koła.",
    "error_not_found": "Nie znaleziono strony.",
    "error_not_retryable": "Ponowić można tylko zadanie zakończone niepowodzeniem lub anulowane.",
    "error_generic": "Wystąpił błąd.",
    "error_back": "Wróć na stronę główną",
}

# Etykiety parametrów zadania (`Job.params`) w szczegółach zadania. Parametr bez etykiety
# wyświetla się pod swoim kluczem, więc nowe parametry są widoczne bez zmian w szablonach.
PARAM_LABELS: Final[dict[str, str]] = {}

STATUS_LABELS: Final[dict[JobStatus, str]] = {
    JobStatus.DRAFT: TEXTS["status_draft"],
    JobStatus.QUEUED: TEXTS["status_queued"],
    JobStatus.RUNNING: TEXTS["status_running"],
    JobStatus.DONE: TEXTS["status_done"],
    JobStatus.FAILED: TEXTS["status_failed"],
    JobStatus.CANCELLED: TEXTS["status_cancelled"],
}


def job_status_label(job: Job) -> str:
    """Status zadania dla użytkownika; praca workera nad maską ma własny opis."""
    if job.status.busy and job.task == TaskKind.SEGMENT:
        return TEXTS["status_segmenting"]
    if job.status.busy and job.cancel_requested:
        return TEXTS["status_cancelling"]
    return STATUS_LABELS[job.status]


def job_progress(job: Job) -> Progress | None:
    """Postęp generowania, gdy worker nad nim pracuje."""
    if job.status != JobStatus.RUNNING or job.task != TaskKind.GENERATE:
        return None
    return Progress.from_json(job.progress)


def progress_label(progress: Progress) -> str:
    if progress.steps is None:
        return TEXTS["progress_wheel"].format(wheel=progress.wheel, wheels=progress.wheels)
    return TEXTS["progress_step"].format(
        wheel=progress.wheel, wheels=progress.wheels, step=progress.step, steps=progress.steps
    )


def can_cancel(job: Job) -> bool:
    """Przycisk „Anuluj”: generowanie w kolejce albo w trakcie, bez wcześniejszej prośby."""
    return job.status.busy and job.task == TaskKind.GENERATE and not job.cancel_requested
