# Wheeler: zadania deweloperskie. `just` bez argumentów wypisuje listę.
#
# Podman: wszystko działa na maszynie `wheels` (2 CPU, 2 GB), tworzonej automatycznie.
# Domyślnego połączenia Podmana nie zmieniamy; wybieramy je przez CONTAINER_CONNECTION.
#
# Testy (`just test`) startują WŁASNY kontener PostgreSQL o unikalnej nazwie
# `wheeler-test-<hex>` z losowym portem hosta (127.0.0.1) i usuwają go po sesji
# (tests/conftest.py). Wiele worktree może więc równolegle uruchamiać `just test`.
# Osierocone kontenery testowe TEGO worktree usuwa `just test-clean`.

set shell := ["bash", "-euo", "pipefail", "-c"]

machine := env("WHEELER_PODMAN_CONNECTION", "wheels")

export CONTAINER_CONNECTION := machine
export WHEELER_PODMAN_CONNECTION := machine

tailwind_version := "v4.3.3"
tailwind_sha256 := "cdf646702987a743464dff4d9c60fd4480d1c1e73dd819a9a67f1078815dce9d"
tailwind := justfile_directory() / ".bin" / "tailwindcss-" + tailwind_version
run_dir := justfile_directory() / ".run"

default:
    @just --list

# Tworzy (jeśli trzeba) i uruchamia maszynę Podmana `wheels`.
machine:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! podman machine inspect {{ machine }} >/dev/null 2>&1; then
        echo "Tworzę maszynę Podmana {{ machine }} (2 CPU, 2 GB)…"
        podman machine init --cpus 2 --memory 2048 {{ machine }}
    fi
    if ! podman machine inspect {{ machine }} | grep -q '"State": "running"'; then
        podman machine start {{ machine }}
    fi

# Pobiera standalone CLI Tailwinda do .bin/ (bez Node.js), z weryfikacją SHA-256.
tailwind:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -x "{{ tailwind }}" ] && exit 0
    mkdir -p "$(dirname "{{ tailwind }}")"
    tmp="$(mktemp "{{ tailwind }}.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    curl -fsSL -o "$tmp" \
        "https://github.com/tailwindlabs/tailwindcss/releases/download/{{ tailwind_version }}/tailwindcss-macos-arm64"
    echo "{{ tailwind_sha256 }}  $tmp" | shasum -a 256 -c - >/dev/null
    chmod +x "$tmp"
    mv "$tmp" "{{ tailwind }}"

# Buduje api/src/wheeler_api/static/app.css z szablonów.
css: tailwind
    cd api && "{{ tailwind }}" -i tailwind.css -o src/wheeler_api/static/app.css --minify

# Instaluje zależności całego workspace'u (api, worker, shared + dev).
sync:
    uv sync --all-packages --frozen

# Startuje maszynę `wheels`, kontenery (Postgres, API) i workera w tle. UI: http://127.0.0.1:8000
up: machine css sync
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p data "{{ run_dir }}"
    podman compose up -d --build --wait
    if [ -f "{{ run_dir }}/worker.pid" ] && kill -0 "$(cat "{{ run_dir }}/worker.pid")" 2>/dev/null; then
        echo "Worker już działa (pid $(cat "{{ run_dir }}/worker.pid"))."
    else
        nohup .venv/bin/wheeler-worker >> "{{ run_dir }}/worker.log" 2>&1 &
        echo $! > "{{ run_dir }}/worker.pid"
        echo "Worker uruchomiony w tle (pid $!), log: .run/worker.log"
    fi
    echo "Wheeler działa: http://127.0.0.1:${WHEELER_API_PORT:-8000}"

# Zatrzymuje workera i kontenery (maszyna `wheels` zostaje uruchomiona).
down:
    #!/usr/bin/env bash
    set -euo pipefail
    pidfile="{{ run_dir }}/worker.pid"
    if [ -f "$pidfile" ]; then
        pid="$(cat "$pidfile")"
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid"
            for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
            echo "Worker zatrzymany."
        fi
        rm -f "$pidfile"
    fi
    if podman machine inspect {{ machine }} 2>/dev/null | grep -q '"State": "running"'; then
        podman compose down
    fi

# Uruchamia workera na pierwszym planie (Ctrl+C kończy po bieżącym zadaniu).
worker: sync
    .venv/bin/wheeler-worker

# Logi kontenerów (np. `just logs api`).
logs *services:
    podman compose logs -f {{ services }}

# Testy: prawdziwy Postgres w efemerycznym kontenerze na `wheels` (unikalna nazwa, losowy port).
test *args: machine sync
    uv run --frozen pytest {{ args }}

# ruff (lint + format) i mypy.
lint: sync
    uv run --frozen ruff check .
    uv run --frozen ruff format --check .
    uv run --frozen mypy

# Formatowanie i autopoprawki ruff.
fmt:
    uv run --frozen ruff check --fix .
    uv run --frozen ruff format .

# Wszystkie kontrole: lint + testy.
check: lint test

# Usuwa osierocone kontenery testowe z TEGO worktree (np. po przerwanym pytest).
test-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    label="wheeler.test.worktree=$(printf '%s' "{{ justfile_directory() }}" | shasum -a 1 | cut -c1-12)"
    ids="$(podman ps -aq --filter "label=$label")"
    [ -n "$ids" ] && podman rm -f -t 0 $ids || echo "Brak kontenerów testowych."
