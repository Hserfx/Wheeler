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
- PID i log workera: `.run/worker.pid`, `.run/worker.log`.

## Struktura

| Katalog   | Zawartość |
|-----------|-----------|
| `shared/` | `wheeler_shared`: modele SQLAlchemy, migracje Alembica (`migrations/`), kolejka zadań (`queue.py`), magazyn plików (`storage.py`), konfiguracja |
| `api/`    | `wheeler_api`: FastAPI + Jinja2 + HTMX/Alpine; teksty UI w `texts.py`; obraz `api/Containerfile` |
| `worker/` | `wheeler_worker`: natywny worker; `runner.py` (pętla kolejki), `processing.py` (procesory) |
| `tests/`  | pytest, w tym testy integracyjne na prawdziwym Postgresie |

Jeden `uv.lock` dla całego workspace'u. Katalog `spike/` (ticket 01) nie należy do workspace'u.

## Przepływ zadania i worker

Zadanie (`Job`) przechodzi przez kroki użytkownika (status `draft`) i pracę workera
(`queued`/`running`, rodzaj pracy w `Job.task`):

```
wgranie auta + felgi -> draft --klik w koło--> segment (SAM 2) -> draft z maską (nakładka)
                        draft --„Generuj”----> generate (mflux) -> done | failed
```

- SAM 2 i mflux działają wyłącznie w natywnym workerze, więc maska po kliknięciu też idzie przez
  kolejkę (`TaskKind.SEGMENT`). Segmentacja ma pierwszeństwo przed generowaniem, w obrębie rodzaju
  obowiązuje FIFO po `queued_at`. Nieudana segmentacja wraca do szkicu z komunikatem.
- Działania użytkownika: `wheeler_shared/editing.py` (`select_wheel`, `start_generation`).
  Kolejka: `wheeler_shared/queue.py`. Parametry edycji i ustawienia domyślne:
  `wheeler_shared/edit_params.py`.
- `wheeler_worker/processing.py` definiuje `Processor` z metodą na każdy rodzaj pracy
  (`segment`, `generate`; tylko pliki, bez bazy). Runner (`runner.py`, `HANDLERS`) pobiera zadanie
  (`FOR UPDATE SKIP LOCKED`), woła procesor i zapisuje wynik albo błąd.
- `WHEELER_PROCESSOR`: `wheel-swap` (domyślny: SAM 2.1 + mflux Qwen-Image-2.1-Turbo q8, wagi
  w `WHEELER_MFLUX_MODEL_PATH`, domyślnie `~/.cache/wheeler/qwen-image-2.1-turbo-mflux-q8`) albo
  `placeholder` (te same kroki z atrapami modeli, bez GPU; do pracy nad UI).
- Modele ładują się leniwie przy pierwszym użyciu. Przed generowaniem SAM 2 jest zwalniany.
- Pliki koła w katalogu zadania: `wheels/<id>/mask.png` (dokładnie obszar, który może się zmienić),
  `overlay.png`, `crop.png`, `crop_mask.png`, `generated.png`; wynik `result.png`.

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
  `session_factory` i `storage` (katalog danych w `tmp_path`) budują się na nim.
- `WHEELER_TEST_PG_ADMIN_URL` pozwala wskazać istniejący serwer zamiast startować kontener.
- Test przepływu (`tests/test_upload_flow.py`) używa `TestClient` dla API i `run_once` workera
  z prawdziwym `WheelSwapProcessor` i atrapami modeli (`tests/fake_models.py`); GPU nie jest
  potrzebne. Kroki „wgraj, kliknij, generuj” są w `tests/helpers.py` (`upload`, `finish_job`).
- Logika wycinania i kompozycji: `tests/test_compositing.py`, bez modelu.
