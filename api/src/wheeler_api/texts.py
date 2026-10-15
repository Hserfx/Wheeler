"""Wszystkie teksty interfejsu (po polsku) w jednym miejscu.

Szablony odwołują się do nich przez `t.<klucz>`; w szablonach nie wpisujemy tekstów na sztywno.
"""

from typing import Final

from wheeler_shared.models import JobStatus

TEXTS: Final[dict[str, str]] = {
    "app_name": "Wheeler",
    "app_tagline": "Podmiana felg na zdjęciu samochodu",
    "nav_home": "Nowe zdjęcie",
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

STATUS_LABELS: Final[dict[JobStatus, str]] = {
    JobStatus.QUEUED: TEXTS["status_queued"],
    JobStatus.RUNNING: TEXTS["status_running"],
    JobStatus.DONE: TEXTS["status_done"],
    JobStatus.FAILED: TEXTS["status_failed"],
}
