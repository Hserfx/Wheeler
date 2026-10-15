"""Wszystkie teksty interfejsu (po polsku) w jednym miejscu.

Szablony odwołują się do nich przez `t.<klucz>`; w szablonach nie wpisujemy tekstów na sztywno.
"""

from typing import Final

from wheeler_shared.models import JobStatus

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
    "upload_heading": "Wgraj zdjęcie samochodu",
    "upload_hint": "Wybierz plik JPEG, PNG lub WebP.",
    "upload_choose": "Wybierz zdjęcie",
    "upload_no_file_chosen": "Nie wybrano pliku",
    "upload_submit": "Wyślij",
    "upload_sending": "Wysyłanie…",
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
    "status_queued": "W kolejce",
    "status_running": "Przetwarzanie…",
    "status_done": "Gotowe",
    "status_failed": "Niepowodzenie",
    # Błędy
    "error_no_file": "Wybierz zdjęcie do wgrania.",
    "error_not_found": "Nie znaleziono strony.",
    "error_generic": "Wystąpił błąd.",
    "error_back": "Wróć na stronę główną",
}

# Etykiety parametrów zadania (`Job.params`) w szczegółach zadania. Parametr bez etykiety
# wyświetla się pod swoim kluczem, więc nowe parametry są widoczne bez zmian w szablonach.
PARAM_LABELS: Final[dict[str, str]] = {}

STATUS_LABELS: Final[dict[JobStatus, str]] = {
    JobStatus.QUEUED: TEXTS["status_queued"],
    JobStatus.RUNNING: TEXTS["status_running"],
    JobStatus.DONE: TEXTS["status_done"],
    JobStatus.FAILED: TEXTS["status_failed"],
}
