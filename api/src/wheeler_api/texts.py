"""Wszystkie teksty interfejsu (po polsku) w jednym miejscu.

Szablony odwołują się do nich przez `t.<klucz>`; w szablonach nie wpisujemy tekstów na sztywno.
"""

from typing import Final

from wheeler_shared import edit_params
from wheeler_shared.models import Job, JobStatus, TaskKind
from wheeler_shared.worker_state import ModelState

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
    "upload_hint": (
        "Pliki JPEG, PNG, WebP lub HEIC, do 50 MB i 40 MPx. Koło do podmiany wskażesz "
        "w następnym kroku."
    ),
    "upload_drop_hint": "Przeciągnij plik tutaj albo wklej go ze schowka (Ctrl+V / ⌘V).",
    "upload_photo_label": "Zdjęcie samochodu",
    "upload_rim_label": "Zdjęcie felgi",
    "upload_choose": "Wybierz plik",
    "upload_no_file_chosen": "Nie wybrano pliku",
    "upload_submit": "Dalej",
    "upload_sending": "Wysyłanie…",
    # Sekcja „Zaawansowane” (parametry edycji)
    "advanced_heading": "Zaawansowane",
    "advanced_hint": "Bez zmian użyte zostaną ustawienia domyślne.",
    "param_seed": "Seed",
    "param_seed_placeholder": "losowy",
    "param_seed_hint": (
        "Puste pole oznacza losowy seed. Użyty seed zobaczysz w szczegółach zadania."
    ),
    "param_resolution": "Rozdzielczość robocza (px)",
    "param_resolution_hint": (
        "Wielokrotność {step}, od {min} do {max}. Wyższa jest wolniejsza i zajmuje więcej pamięci."
    ),
    "param_mask_grow": "Poszerzenie maski (%)",
    "param_mask_grow_hint": "Procent boku koła, od {min} do {max}.",
    "param_prompt_suffix": "Dopisek do promptu",
    "param_prompt_suffix_placeholder": "np. Glossy black finish.",
    "param_prompt_suffix_hint": "Po angielsku; doklejany na końcu promptu.",
    "param_empty": "(brak)",
    "error_param_seed": "Seed musi być liczbą całkowitą od 0 do {seed_max} albo pustym polem.",
    "error_param_resolution": (
        "Rozdzielczość robocza musi być wielokrotnością {step} px, od {min} do {max} px."
    ),
    "error_param_mask_grow": "Poszerzenie maski musi być liczbą od {grow_min} do {grow_max}%.",
    "error_param_prompt_suffix": "Dopisek do promptu może mieć najwyżej {length} znaków.",
    # Wybór koła i generowanie
    "wheel_click_hint": "Kliknij koło, w którym chcesz podmienić felgę.",
    "wheel_mask_hint": (
        "Zaznaczone koła zostaną wygenerowane na nowo, reszta zdjęcia się nie zmieni. "
        "Kliknij maskę, aby usunąć koło, albo kliknij koło bez maski, aby je dodać."
    ),
    "wheels_none_detected": ("Nie wykryto żadnego koła. Kliknij koło na zdjęciu, aby je dodać."),
    "wheels_selected": "Wybrane koła: {count}",
    "wheel_picker_alt": "Zdjęcie samochodu: kliknij koło",
    "wheel_mask_alt": "Maska koła nr {number}",
    "generate_submit": "Generuj",
    "generate_hint": "Generowanie trwa około 2 minut na każde koło.",
    "job_rim": "Felga",
    # Zadanie
    "job_heading": "Zadanie nr {id}",
    "job_input": "Zdjęcie wejściowe",
    "job_result": "Wynik",
    "job_download": "Pobierz PNG",
    # Ekran wyniku (suwak przed/po, powiększanie, ponowienie koła)
    "compare_hint": "Przesuń suwak, aby porównać zdjęcie przed i po. Wybierz koło, aby je powiększyć.",
    "compare_before": "Przed",
    "compare_after": "Po",
    "compare_slider": "Suwak przed/po",
    "zoom_wheel": "Koło {number}",
    "zoom_reset": "Cały obraz",
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
    "status_detecting": "Wykrywanie kół…",
    "status_queued": "W kolejce",
    "status_running": "Przetwarzanie…",
    "status_done": "Gotowe",
    "status_failed": "Niepowodzenie",
    # Stan modelu w pamięci workera (nagłówek i zadanie w toku)
    "model_state_label": "Model",
    "model_state_asleep": "uśpiony",
    "model_state_loading": "ładowanie",
    "model_state_ready": "gotowy",
    "model_state_hint": (
        "Model ładuje się przy pierwszym zadaniu i zwalnia pamięć po okresie bezczynności."
    ),
    "model_loading_job": "Ładowanie modelu do pamięci (pierwsze zadanie po przerwie)…",
    # Błędy
    "error_no_photo": "Wybierz zdjęcie samochodu.",
    "error_no_rim": "Wybierz zdjęcie felgi.",
    "error_upload_format": (
        "Plik „{name}” nie jest obsługiwanym zdjęciem. Wgraj plik JPEG, PNG, WebP lub HEIC."
    ),
    "error_upload_too_large": "Plik „{name}” jest za duży. Limit to {limit} MB.",
    "error_upload_too_many_pixels": (
        "Zdjęcie „{name}” ma za dużą rozdzielczość. Limit to {limit} MPx (megapikseli)."
    ),
    "error_no_wheel": "Najpierw kliknij koło i poczekaj na jego maskę.",
    "error_job_busy": "Zadanie jest teraz przetwarzane albo już zakończone.",
    "error_mask_failed": "Nie udało się zaznaczyć koła w tym miejscu. Kliknij bliżej środka koła.",
    "error_detection_failed": (
        "Nie udało się automatycznie wykryć kół. Kliknij koło na zdjęciu, aby je dodać."
    ),
    "error_not_found": "Nie znaleziono strony.",
    "error_generic": "Wystąpił błąd.",
    "error_back": "Wróć na stronę główną",
}

# Etykiety parametrów zadania (`Job.params`) w szczegółach zadania. Parametr bez etykiety
# wyświetla się pod swoim kluczem, więc nowe parametry są widoczne bez zmian w szablonach.
PARAM_LABELS: Final[dict[str, str]] = {
    edit_params.SEED: TEXTS["param_seed"],
    edit_params.RESOLUTION: "Rozdzielczość robocza",
    edit_params.MASK_GROW: "Poszerzenie maski",
    edit_params.PROMPT_SUFFIX: TEXTS["param_prompt_suffix"],
}

STATUS_LABELS: Final[dict[JobStatus, str]] = {
    JobStatus.DRAFT: TEXTS["status_draft"],
    JobStatus.QUEUED: TEXTS["status_queued"],
    JobStatus.RUNNING: TEXTS["status_running"],
    JobStatus.DONE: TEXTS["status_done"],
    JobStatus.FAILED: TEXTS["status_failed"],
}


def job_status_label(job: Job) -> str:
    """Status zadania dla użytkownika; praca workera nad maską ma własny opis."""
    if job.status.busy and job.task == TaskKind.SEGMENT:
        return TEXTS["status_segmenting"]
    if job.status.busy and job.task == TaskKind.DETECT:
        return TEXTS["status_detecting"]
    return STATUS_LABELS[job.status]


MODEL_STATE_LABELS: Final[dict[ModelState, str]] = {
    ModelState.ASLEEP: TEXTS["model_state_asleep"],
    ModelState.LOADING: TEXTS["model_state_loading"],
    ModelState.READY: TEXTS["model_state_ready"],
}
