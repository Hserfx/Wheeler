# Wheeler

Lokalna aplikacja do podmiany felg na zdjęciu samochodu. Projekt: [docs/DESIGN.md](docs/DESIGN.md),
tickety: [docs/tickets.md](docs/tickets.md).

## Wymagania

macOS (Apple Silicon), [uv](https://docs.astral.sh/uv/), [just](https://just.systems/), Podman 6
z `docker-compose` (backend `podman compose`). Tailwind pobiera się sam (`just tailwind`, do `.bin/`).

## Uruchamianie

```sh
just up      # maszyna Podmana `wheels`, Postgres + API w kontenerach, worker w tle
             # → http://127.0.0.1:8000
just down    # zatrzymuje workera i kontenery (maszyna `wheels` zostaje)
just worker  # worker na pierwszym planie (zamiast tła z `just up`)
just logs api
```

- Maszyna `wheels` (2 CPU, 2 GB) jest tworzona automatycznie (`just machine`). Domyślnego
  połączenia Podmana nie zmieniamy, `justfile` ustawia `CONTAINER_CONNECTION=wheels`.
- Porty są publikowane wyłącznie na `127.0.0.1`: API `8000`, Postgres `55432` (dla natywnego
  workera). Zmiana: `WHEELER_API_PORT`, `WHEELER_DB_PORT`.
- Pliki leżą w `./data` (bind mount `/data` w kontenerze API). W bazie są tylko ścieżki względne
  wobec tego katalogu. Dane Postgresa są w wolumenie `wheeler_pgdata` na maszynie `wheels`.
- Wszystkie pliki zadania (wejście, maski, wycinki, wynik, miniatury) leżą w jego katalogu
  `data/jobs/<hex>` (`Storage.new_job_dir`). Usunięcie zadania w Historii
  (`wheeler_shared.history.delete_jobs`) kasuje ten katalog w całości; pliki poza nim (np. felgi
  z biblioteki) zostają. Automatycznego sprzątania nie ma.
- Felgi z biblioteki leżą w `data/rims/<hex>/` (`original.<ext>`, `cut.png`). Zadanie dostaje
  kopię wyciętej felgi (`jobs/<hex>/rim.png`), więc usunięcie felgi z biblioteki nie rusza historii.
- PID i log workera: `.run/worker.pid`, `.run/worker.log`.

## Struktura

| Katalog   | Zawartość |
|-----------|-----------|
| `shared/` | `wheeler_shared`: modele SQLAlchemy, migracje Alembica (`migrations/`), kolejka zadań (`queue.py`), magazyn plików (`storage.py`), konfiguracja |
| `api/`    | `wheeler_api`: FastAPI + Jinja2 + HTMX/Alpine; teksty UI w `texts.py`, stan zadania dla szablonów w `job_view.py`, zależności tras w `deps.py`; obraz `api/Containerfile` |
| `worker/` | `wheeler_worker`: natywny worker; `runner.py` (pętla kolejki), `processing.py` (procesory) |
| `tests/`  | pytest, w tym testy integracyjne na prawdziwym Postgresie |

Jeden `uv.lock` dla całego workspace'u. Katalog `spike/` (ticket 01) nie należy do workspace'u.

## Przepływ zadania i worker

Zadanie (`Job`) przechodzi przez kroki użytkownika (status `draft`) i pracę workera
(`queued`/`running`, rodzaj pracy w `Job.task`):

```
wgranie auta -> detect (Florence-2 + SAM 2) -> draft z maskami wykrytych kół (nakładki)
draft --klik w maskę---------> draft (koło usunięte od razu, bez workera)
draft --klik obok masek------> segment (SAM 2) -> draft z dodanym kołem
draft --felga + „Generuj”----> generate (mflux, każde koło osobno, ten sam seed i prompt) -> done | failed | cancelled
done --„Ponów to koło”-------> regenerate_wheel (jedno koło, nowy seed) -> done (błąd/anulowanie: poprzedni wynik)
```

- Detektor, SAM 2 i mflux działają wyłącznie w natywnym workerze, więc detekcja po wgraniu
  (`TaskKind.DETECT`) i maska po kliknięciu (`TaskKind.SEGMENT`) idą przez kolejkę. Obie mają
  pierwszeństwo przed generowaniem, w obrębie rodzaju obowiązuje FIFO po `queued_at` (świadome
  odstępstwo od FIFO, DESIGN.md „Kolejka zadań”; `queue_position` liczy trwającą pracę i felgi).
  Nieudana detekcja albo segmentacja wraca do szkicu z komunikatem (koło można dodać kliknięciem).
- Błędy workera: `Job.error`/`Rim.error` trzymają kod `wheeler_shared.failures.Failure`
  (procesor rzuca `ProcessingFailed(kod)`, każdy inny wyjątek to `internal`), a szablony pokazują
  tekst z `texts.py` (`failure_text`). Wyjątki bibliotek trafiają tylko do logu workera.
- Koła (`Wheel`): `source` (`detected` | `click`), maska, nakładka i `box_*` (prostokąt maski,
  wstępny filtr przy trafianiu kliknięciem). `Job.wheels_detected` = liczba kół z detekcji
  (0 -> UI mówi „Nie wykryto żadnego koła”). Klik trafia w maskę, gdy plik maski ma w tym punkcie
  piksel > 0 (`wheeler_api/wheel_masks.py`); przy nakładających się maskach usuwa koło dodane później.
- Działania użytkownika: `wheeler_shared/editing.py` (`request_detection`, `click_photo`,
  `start_generation`, `cancel_job`, `regenerate_wheel`).
  Kolejka: `wheeler_shared/queue.py`. Parametry edycji, ustawienia domyślne i dozwolone zakresy:
  `wheeler_shared/edit_params.py`. Sekcja „Zaawansowane” na stronie głównej
  (`wheeler_api/advanced.py`) zapisuje w `Job.params` użyte wartości już przy wgraniu zdjęcia;
  pusty seed jest losowany i zapisywany przy „Generuj”. Rozmiar felgi (`rim_size`, −2…+2″) nie należy
  do „Zaawansowane”: wybiera się go obok felgi na stronie zadania i zapisuje dopiero przy „Generuj”,
  tak samo jak obecną felgę (`current_rim`, 14…22″, brak klucza = 17″; od ticketu 16 skala cala).
  Steruje nim geometria, nie prompt: przy stopniu ≠ 0 worker przed mflux wyznacza elipsę felgi
  (SAM 2, `Segmenter.segment_rim`) i przeskalowuje felgę w wycinku radialnym warpem
  (`wheeler_worker/rim_geometry.py`). Razem z felgą wyznacza maskę karoserii zasłaniającej koło
  (`Segmenter.segment_body`, wybór i kryteria w `wheeler_worker/body_mask.py`; od ticketu 16 chroniona
  przy zmianie rozmiaru całego koła). Felga nie do wyznaczenia, zła maska karoserii albo obce
  zasłonięcie koła kończą zadanie kodem `rim_size_unavailable`.
- `wheeler_worker/processing.py` definiuje `Processor` z metodą na każdy rodzaj pracy
  (`detect`, `segment`, `generate`, `regenerate_wheel`; tylko pliki, bez bazy). Runner (`runner.py`, `HANDLERS`) pobiera zadanie
  (`FOR UPDATE SKIP LOCKED`), woła procesor i zapisuje wynik albo błąd.
- `WHEELER_PROCESSOR`: `wheel-swap` (domyślny: Florence-2-large + SAM 2.1 + mflux Qwen-Image-2.1-Turbo q8, wagi
  w `WHEELER_MFLUX_MODEL_PATH`, domyślnie `~/.cache/wheeler/qwen-image-2.1-turbo-mflux-q8`) albo
  `placeholder` (te same kroki z atrapami modeli, bez GPU; do pracy nad UI). Usuwanie tła felgi:
  BiRefNet (`ZhengPeng7/BiRefNet`).
- Modele ładują się leniwie przy pierwszym użyciu. Przed generowaniem Florence-2 i SAM 2 są zwalniane.
  Detekcja z ładowaniem modeli 4–6 s (pierwszy odczyt wag z dysku ~20 s), potem ~1 s na zdjęcie.
- Cykl życia modeli (`wheeler_worker/model_lifecycle.py`): po starcie workera modele nie są
  załadowane („uśpiony”), pierwsze zadanie je ładuje („ładowanie”, też w statusie zadania),
  potem zostają w pamięci („gotowy”) i są zwalniane po `WHEELER_MODEL_IDLE_TIMEOUT` sekundach
  bez zadań (domyślnie 600). Stan leży w tabeli `worker_state` (`wheeler_shared/worker_state.py`),
  a UI pokazuje go w nagłówku (`GET /model-state`, odświeżane co 2 s).
- Pliki koła w katalogu zadania: `wheels/<id>/mask.png` (dokładnie obszar, który może się zmienić;
  dla kół z detekcji maska i nakładka leżą w `wheels/detected-<n>/`),
  `overlay.png`, `crop.png`, `crop_mask.png`, `generated.png`; wynik `result.png`.
- Ekran wyniku (`_result_compare.html`): suwak przed/po na pełnym obrazie (warstwa „przed”
  przycinana `clip-path`), powiększanie okolicy koła (prostokąt `Wheel.box`, jedna transformacja
  CSS dla obu obrazów), pobieranie `GET /jobs/{id}/download` (PNG jako załącznik, rozdzielczość
  oryginału). „Ponów to koło” (`POST /jobs/{id}/wheels/{wheel_id}/regenerate`, tylko w `done`)
  zapisuje nowy seed w `Wheel.pending_seed`; worker (`TaskKind.REGENERATE_WHEEL`, w kolejce jak
  generowanie, z postępem i anulowaniem) generuje tylko to koło i składa je na BIEŻĄCYM wyniku,
  więc piksele poza maską tego koła (także inne koła) zostają bit w bit. Nowy wynik trafia do
  nowego pliku `result-<n>.png` (poprzedni zostaje; nowa nazwa omija cache przeglądarki),
  `Image(result)` wskazuje nowy plik, a koło dostaje `Wheel.seed` (pokazywany przy kole).
  Błąd, anulowanie albo awaria workera w trakcie wracają do `done` z poprzednim wynikiem.
- Postęp i anulowanie: procesor raportuje `Progress` (`wheeler_shared/progress.py`: koło, krok
  denoisingu) przed każdym kołem i po każdym kroku (`GenerateItem.on_progress`, w mflux przez
  `RimEditor.edit(on_step=...)`). Runner zapisuje go w `Job.progress` i przy okazji czyta
  `Job.cancel_requested`; ustawiona flaga przerywa pracę wyjątkiem `JobCancelled` (najpóźniej po
  bieżącym kroku), a zadanie dostaje status `cancelled`. Zadanie w kolejce jest anulowane od razu.
- Awaria lub restart workera: przy starcie worker oznacza zadania `running` jako `failed`
  (`recover_interrupted_jobs`; detekcja i segmentacja wracają do szkicu, wycinane felgi dostają `failed`), bez automatycznych ponowień. Zakłada
  jeden proces workera. „Ponów” (`wheeler_shared/retry.py`) tworzy nowe zadanie z kopiami plików,
  tymi samymi kołami i parametrami (z seedem) i od razu wstawia je do kolejki.
- Biblioteka felg (`wheeler_shared/rim_library.py`, UI: fragment `_rim_picker.html` na stronie
  zadania (przed „Generuj”) i na `/rims`, trasy w `wheeler_api/rim_routes.py`). Felga (`Rim`) ma
  własną kolejkę w tabeli `rims`, obsługiwaną przez worker przed zadaniami (`runner._cut_next_rim`,
  `Processor.cut_rim`):

  ```
  wgranie felgi -> queued -> worker: usuwanie tła -> review --„Dodaj do biblioteki”--> ready
  review/failed --klik w felgę na oryginale--> queued (wycięcie koła pod kliknięciem) -> review
  review/failed --„Odrzuć”, ready --„Usuń z biblioteki”--> wiersz i katalog felgi usunięte
  running --awaria/restart workera--> failed (przy starcie workera, jak zadania)
  ```

  Usuwanie tła (`WheelSwapProcessor.cut_rim`): Florence-2 wykrywa koła, a BiRefNet
  (`wheeler_worker/background.py`) wycina pierwszy plan z kadru wokół koła (największego albo
  tego pod kliknięciem; boxy obejmujące inne koło są pomijane). Bez kadru BiRefNet zostawia
  karoserię, gdy felga jest sfotografowana na aucie. Bez wykrytego koła: BiRefNet na całym zdjęciu
  albo SAM 2 z kliknięcia. Wynik: RGBA PNG przycięty do felgi (`rim_cut.cut_out`, maks. 1536 px).
  Model dostaje felgę na białym tle (`rim_cut.load_reference`). Felgę wybiera się na stronie zadania
  przed „Generuj” (krok 3: fragment `_rim_picker.html` poza odpytywanym `#job-status`, radio
  `rim_id` należy do formularza „Generuj” `#job-form`; `editing.choose_rim` kopiuje ją do zadania,
  `start_generation` bez felgi rzuca `NoRimSelected`). `POST /jobs` przyjmuje opcjonalnie `rim_id`.
  Wybór liczy użycie (`use_count`, `last_used_at`); biblioteka jest
  sortowana od ostatnio użytej (nowa, nieużyta felga liczy się od chwili dodania).
  Wagi BiRefNet (~0,9 GB, kod modelu z repozytorium HF, przypięta rewizja):
  `uv run hf download ZhengPeng7/BiRefNet --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`.
  Zmierzone na M2 Max: ładowanie + pierwsze zdjęcie ~4,5 s, potem ~0,9 s (z detekcją).
- Wejście (`wheeler_api/uploads.py`, `save_image_upload`, wspólne dla każdego wgrywanego obrazu):
  JPEG, PNG, WebP, HEIC, do 50 MB i 40 MPx; format rozpoznawany po zawartości. Tryby pikseli:
  `wheeler_shared/photo_modes.py` (16 bitów na kanał i CMYK są odrzucane, bo wynik nie byłby bit w bit). Dla HEIC i zdjęć
  z orientacją EXIF obok pliku powstaje obrócony podgląd `<plik>.preview.jpg` w pełnym rozmiarze
  (szablony: filtr `display_path`), bo kliknięcia są w układzie obróconego zdjęcia.
- Odczyt i zapis zdjęć w workerze: `wheeler_worker/photos.py` (`load_photo`: RGB dla modeli,
  `load_original`: oryginał w trybie wyniku do kompozycji). `result.png` dostaje profil ICC
  i EXIF oryginału (`read_metadata`: bez GPS i orientacji); maski i pliki robocze są bez metadanych.

## Migracje

Migracje uruchamiają się same przy starcie API i workera (`wheeler_shared.migrate.migrate`,
z blokadą doradczą Postgresa). Nowa migracja: plik w `shared/src/wheeler_shared/migrations/versions/`
(szablon `script.py.mako`), z kolejnym numerem rewizji.

## Testy i jakość

```sh
just test            # pytest (argumenty przekazywane dalej, np. `just test -k queue -x`)
just lint            # ruff check + ruff format --check + mypy (strict)
just check           # lint + test
just test-clean      # usuwa osierocone kontenery testowe z TEGO worktree
```

Testy integracyjne (`tests/conftest.py`):

- Fixture `pg_admin_url` (sesja) startuje **własny** kontener `postgres:17-alpine` na maszynie
  `wheels`, o unikalnej nazwie `wheeler-test-<hex>` i z **losowym portem hosta**
  (`-p 127.0.0.1::5432`). Kontener jest usuwany po sesji. Dzięki temu wiele worktree może
  równolegle uruchamiać `just test`. Kontenery mają etykiety `wheeler.test=1` i
  `wheeler.test.worktree=<hash ścieżki>`, po których `just test-clean` sprząta tylko własne.
- Fixture `database_url` tworzy na każdy test świeżą bazę i wykonuje na niej migracje.
  `session_factory`, `storage` (katalog danych w `tmp_path`) i `client` (`TestClient` API)
  budują się na nim.
- `WHEELER_TEST_PG_ADMIN_URL` pozwala wskazać istniejący serwer zamiast startować kontener.
- Test przepływu (`tests/test_upload_flow.py`) używa `TestClient` dla API i `run_once` workera
  z prawdziwym `WheelSwapProcessor` i atrapami modeli (`tests/fake_models.py`); GPU nie jest
  potrzebne. Kroki „wgraj, kliknij, generuj” są w `tests/helpers.py` (`upload` ze zdjęciem
  i felgą, `upload_photo` bez felgi, `drain`, `finish_job`, `image_src`/`fetch`). Po `upload` zadanie czeka w kolejce na detekcję kół: przed kliknięciem koła
  przetwórz kolejkę (`drain`; atrapa `fake_processor` nie wykrywa kół, `BoxDetector` zwraca zadane boxy).
- Logika wycinania i kompozycji: `tests/test_compositing.py`, bez modelu.
