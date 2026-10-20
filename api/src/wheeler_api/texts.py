"""Wszystkie teksty interfejsu (po polsku) w jednym miejscu.

Szablony odwołują się do nich przez `t.<klucz>`; w szablonach nie wpisujemy tekstów na sztywno.
"""

from typing import Final

from wheeler_shared import edit_params
from wheeler_shared.failures import Failure
from wheeler_shared.models import JobStatus
from wheeler_shared.worker_state import ModelState

TEXTS: Final[dict[str, str]] = {
    "app_name": "Wheeler",
    "app_tagline": "Podmiana felg na zdjęciu samochodu",
    "nav_home": "Nowe zdjęcie",
    "nav_history": "Historia",
    "nav_rims": "Biblioteka felg",
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
    "upload_hint": (
        "Pliki JPEG, PNG, WebP lub HEIC, do 50 MB i 40 MPx. Koła wykryjemy automatycznie, "
        "a felgę wybierzesz w następnym kroku."
    ),
    "upload_drop_hint": "Przeciągnij plik tutaj albo wklej go ze schowka (Ctrl+V / ⌘V).",
    "upload_photo_label": "Zdjęcie samochodu",
    "upload_choose": "Wybierz plik",
    "upload_no_file_chosen": "Nie wybrano pliku",
    "upload_submit": "Dalej",
    "upload_sending": "Wysyłanie…",
    # Biblioteka felg
    "library_heading": "Biblioteka felg",
    "library_hint": (
        "Felgi dodane raz można wybierać przy kolejnych zdjęciach. Tło jest usuwane automatycznie."
    ),
    "library_empty": "Biblioteka jest pusta. Dodaj pierwszą felgę poniżej.",
    "rim_unnamed": "Felga nr {id}",
    "rim_new": "Nowa felga",
    "rim_used": "Użycia: {count}",
    "rim_last_used": "ostatnio",
    "rim_delete": "Usuń z biblioteki",
    "rim_delete_confirm": (
        "Usunąć felgę z biblioteki? Zadania, które jej użyły, zachowają swoją kopię felgi."
    ),
    "rim_add_heading": "Dodaj nową felgę",
    "rim_add_hint": "Zdjęcie felgi (JPEG, PNG, WebP lub HEIC). Tło usuniemy automatycznie.",
    "rim_name_placeholder": "Nazwa (opcjonalnie)",
    "rim_add_submit": "Dodaj felgę",
    "rim_cutting": "Usuwanie tła…",
    "rim_original": "Oryginał",
    "rim_cut": "Wycięta felga",
    "rim_review_hint": (
        "Sprawdź wyciętą felgę. Jeśli zostało coś poza nią (np. karoseria), kliknij felgę "
        "na oryginale, a wytniemy samo koło."
    ),
    "rim_cut_failed": "Nie udało się wyciąć felgi. Kliknij felgę na oryginale albo odrzuć zdjęcie.",
    "rim_accept": "Dodaj do biblioteki",
    "rim_reject": "Odrzuć",
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
    # Etykiety parametrów w szczegółach zadania (`PARAM_LABELS`)
    "job_param_seed": "Seed",
    "job_param_resolution": "Rozdzielczość robocza",
    "job_param_mask_grow": "Poszerzenie maski",
    "job_param_prompt_suffix": "Dopisek do promptu",
    "job_param_rim_size": "Rozmiar felgi",
    "job_param_current_rim": "Obecna felga",
    "error_param_seed": "Seed musi być liczbą całkowitą od 0 do {seed_max} albo pustym polem.",
    "error_param_resolution": (
        "Rozdzielczość robocza musi być wielokrotnością {step} px, "
        "od {resolution_min} do {resolution_max} px."
    ),
    "error_param_mask_grow": "Poszerzenie maski musi być liczbą od {grow_min} do {grow_max}%.",
    "error_param_prompt_suffix": "Dopisek do promptu może mieć najwyżej {length} znaków.",
    "error_param_rim_size": "Rozmiar felgi musi być stopniem od {rim_size_min} do {rim_size_max}.",
    "error_param_current_rim": (
        "Obecna felga musi być pełnym rozmiarem od {current_rim_min} do {current_rim_max}."
    ),
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
    "rim_choose_heading": "Felga",
    "rim_choose_hint": (
        "Wybierz felgę z biblioteki albo dodaj nową (sprawdzisz ją przed dodaniem). "
        "Felgę trzeba wybrać przed „Generuj”; możesz to zrobić już w trakcie wykrywania kół."
    ),
    # Rozmiar felgi (tickety 12 i 16): rośnie całe koło
    "rim_size_heading": "Rozmiar felgi",
    "rim_size_hint": (
        "Nowa felga względem obecnej, z tą samą oponą: całe koło rośnie albo maleje o tyle "
        "cali i chowa się za błotnikiem, a styk z ziemią zostaje. Cale są przybliżone."
    ),
    "rim_size_unchanged": "bez zmian",
    "rim_size_inches": "{sign}{inches}″",
    "result_rim_size": "Felga: {size}",
    # Obecna felga (ticket 14): skala cala przy zmianie rozmiaru (ticket 16)
    "current_rim_label": "Obecna felga",
    "current_rim_hint": (
        "Rozmiar felgi auta na zdjęciu (np. R17 z boku opony). Gdy go nie znasz, zostaw 17″."
    ),
    "current_rim_inches": "{inches}″",
    "result_current_rim": "obecna {size}",
    "sign_plus": "+",
    "sign_minus": "\N{MINUS SIGN}",  # minus typograficzny (U+2212), nie łącznik
    # Zadanie
    "job_heading": "Zadanie nr {id}",
    "job_input": "Zdjęcie wejściowe",
    "job_result": "Wynik",
    "job_download": "Pobierz PNG",
    # Ekran wyniku (suwak przed/po, powiększanie, ponowienie koła)
    "compare_hint": (
        "Przesuń suwak, aby porównać zdjęcie przed i po. Wybierz koło, aby je powiększyć."
    ),
    "compare_before": "Przed",
    "compare_after": "Po",
    "compare_slider": "Suwak przed/po",
    "zoom_wheel": "Koło {number}",
    "zoom_reset": "Cały obraz",
    "wheel_regenerate": "Ponów to koło",
    "wheel_regenerate_hint": (
        "Generuje to koło od nowa z innym seedem; reszta zdjęcia się nie zmienia."
    ),
    "wheel_seed": "seed",
    "error_not_regeneratable": (
        "Ponowić koło można tylko w gotowym wyniku, gdy nic innego nie jest generowane."
    ),
    "status_regenerating": "Ponowne generowanie koła…",
    "error_regenerate_failed": (
        "Nie udało się ponownie wygenerować koła. Poprzedni wynik został zachowany."
    ),
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
    # Postęp, anulowanie, kolejka, ponawianie (ticket 05)
    "progress_wheel": "Koło {wheel} z {wheels}: przygotowanie modelu…",
    "progress_step": "Koło {wheel} z {wheels}, krok {step} z {steps}",
    "status_cancelling": "Anulowanie…",
    "status_cancelled": "Anulowane",
    "job_cancel": "Anuluj",
    "queue_position": "Pozycja w kolejce: {position}",
    "job_retry": "Ponów",
    "job_retry_hint": "Nowe zadanie z tym samym zdjęciem, felgą, kołami i parametrami.",
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
    "error_no_rim": "Wybierz felgę z biblioteki albo dodaj nową.",
    "error_rim_file": "Wybierz zdjęcie felgi.",
    "error_rim_busy": "Felga jest teraz przetwarzana albo jej stan się zmienił. Spróbuj ponownie.",
    "error_upload_format": (
        "Plik „{name}” nie jest obsługiwanym zdjęciem. Wgraj plik JPEG, PNG, WebP lub HEIC."
    ),
    "error_upload_too_large": "Plik „{name}” jest za duży. Limit to {limit} MB.",
    "error_upload_mode": (
        "Zdjęcie „{name}” ma format pikseli, którego nie da się zachować bez zmian poza "
        "kołami (np. 16 bitów na kanał albo CMYK). Zapisz je jako 8-bitowe RGB i wgraj ponownie."
    ),
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
    "error_not_retryable": "Ponowić można tylko zadanie zakończone niepowodzeniem lub anulowane.",
    "error_generic": "Wystąpił błąd.",
    "error_back": "Wróć na stronę główną",
    # Przyczyny niepowodzenia pracy workera (`wheeler_shared.failures.Failure`, `failure_text`)
    "failure_internal": "Błąd wewnętrzny workera. Szczegóły są w jego logu (.run/worker.log).",
    "failure_interrupted": (
        "Przerwane: worker został zatrzymany albo uległ awarii w trakcie zadania."
    ),
    "failure_legacy_job": "Zadanie z poprzedniej wersji aplikacji (bez felgi i maski koła).",
    "failure_nothing_at_click": "W klikniętym miejscu nie znaleziono koła.",
    "failure_empty_mask": "Maska koła jest pusta.",
    "failure_rim_not_found": "Na zdjęciu nie znaleziono felgi.",
    "failure_no_rim": "Zadanie nie ma wybranej felgi.",
    "failure_rim_size_unavailable": (
        "Nie udało się wyznaczyć felgi albo zasłaniającej koło karoserii na jednym z kół, "
        "więc nie da się zmienić rozmiaru felgi. "
        "Wygeneruj bez zmiany rozmiaru."
    ),
    "failure_unsupported_photo": (
        "Zdjęcie ma format pikseli, którego nie da się zachować bez zmian (np. 16 bitów "
        "na kanał albo CMYK)."
    ),
}

# Etykiety parametrów zadania (`Job.params`) w szczegółach zadania. Parametr bez etykiety
# wyświetla się pod swoim kluczem, więc nowe parametry są widoczne bez zmian w szablonach.
PARAM_LABELS: Final[dict[str, str]] = {
    key: TEXTS[f"job_param_{key}"]
    for key in (
        edit_params.SEED,
        edit_params.RESOLUTION,
        edit_params.MASK_GROW,
        edit_params.PROMPT_SUFFIX,
        edit_params.RIM_SIZE,
        edit_params.CURRENT_RIM,
    )
}

STATUS_LABELS: Final[dict[JobStatus, str]] = {
    JobStatus.DRAFT: TEXTS["status_draft"],
    JobStatus.QUEUED: TEXTS["status_queued"],
    JobStatus.RUNNING: TEXTS["status_running"],
    JobStatus.DONE: TEXTS["status_done"],
    JobStatus.FAILED: TEXTS["status_failed"],
    JobStatus.CANCELLED: TEXTS["status_cancelled"],
}


def failure_text(code: str | None) -> str:
    """Komunikat dla kodu niepowodzenia z `Job.error`/`Rim.error`; nieznany kod (np. dawny
    tekst wyjątku) daje komunikat o błędzie wewnętrznym, nigdy surowy tekst z bazy."""
    try:
        failure = Failure(code or Failure.INTERNAL)
    except ValueError:
        failure = Failure.INTERNAL
    return TEXTS[f"failure_{failure.value}"]


MODEL_STATE_LABELS: Final[dict[ModelState, str]] = {
    ModelState.ASLEEP: TEXTS["model_state_asleep"],
    ModelState.LOADING: TEXTS["model_state_loading"],
    ModelState.READY: TEXTS["model_state_ready"],
}
