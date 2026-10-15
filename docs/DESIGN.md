# Wheeler — projekt

Lokalna aplikacja webowa do podmiany felg na zdjęciu samochodu na felgę ze zdjęcia referencyjnego, z użyciem lokalnego modelu Qwen-Image-2.1. Poza kołami obraz pozostaje nienaruszony.

Dokument zbiera ustalenia z sesji projektowej z 2026-10-15.

## Cel i zakres

- Użytkownik wgrywa zdjęcie auta i zdjęcie felgi (albo wybiera felgę z biblioteki), a dostaje to samo zdjęcie z podmienionymi felgami.
- **Zmieniają się tylko koła.** Wynik ma tę samą rozdzielczość co oryginał, a każdy piksel poza maskami kół jest identyczny bit w bit z oryginałem. Bez upscalingu auta ani tła.
- Brak uwierzytelniania. Aplikacja działa **wyłącznie na tym Macu** i nasłuchuje tylko na `127.0.0.1`.
- Priorytetem jest dobre UX, wygląd ma drugorzędne znaczenie.

## Środowisko docelowe

- Apple M2 Max, 32 GB pamięci zunifikowanej, macOS.
- Podman 6.0.2 (VM libkrun), uv 0.12.15. Dostępny `docker-compose`, którego `podman compose` używa jako backendu.
- Wagi w `~/.cache/huggingface/hub`:
  - `Qwen/Qwen-Image-2.1-Turbo`: komplet w bf16, ok. 32,4 GB. Transformer 14,2 GB, text encoder Qwen3-VL-8B 17,5 GB, VAE 0,7 GB. 8 kroków.
  - `Qwen/Qwen-Image-2.1`: tylko metadane, bez wag.

## Architektura

```
przeglądarka ──► FastAPI (Podman) ──► PostgreSQL (Podman)
                      │                    ▲
                      ▼                    │ kolejka zadań
                 ./data (wolumen) ◄── worker AI (natywnie, macOS, MLX)
```

- **Worker AI działa natywnie na macOS** w uv venv. Kontenery Podmana na macOS działają w VM z Linuksem bez dostępu do GPU/Metal, więc model w kontenerze liczyłby się na CPU.
- **FastAPI + PostgreSQL** działają w Podmanie, na osobnej maszynie `wheels` (2 CPU, 2 GB RAM, żeby VM nie zabierała pamięci modelowi). Orkiestracja przez `podman compose`. Port publikowany jako `127.0.0.1:8000:8000`.
- **Komunikacja API ↔ worker** odbywa się przez kolejkę zadań w PostgreSQL (`SELECT … FOR UPDATE SKIP LOCKED`).
- **Pliki** (zdjęcia, maski, wyniki) leżą na wolumenie dyskowym `./data`, montowanym w kontenerze API i dostępnym dla workera. W bazie trzymamy tylko ścieżki i metadane.
- **Uruchamianie:** `just up` startuje maszynę Podmana, kontenery i workera w tle. `just down` zatrzymuje wszystko.

## Worker AI

### Silnik inferencji

- **mflux (MLX)** jako biblioteka Pythona instalowana przez uv. Model: **Qwen-Image-2.1-Turbo w q8**, skonwertowany jednorazowo z lokalnych wag bf16.
- Powody wyboru:
  - bf16 w diffusers nie mieści się w 32 GB w całości (szczyt 46–48 GB).
  - PyTorch MPS ma błąd `F.pad` w enkoderze VAE (diffusers PR #14902, niezmergowany), który psuje edycję.
  - mflux działa natywnie na Apple Silicon, obsługuje kwantyzację, maskę i wiele obrazów referencyjnych, a błąd MPS go nie dotyczy.
- **Zarządzanie pamięcią:** model ładuje się przy pierwszym zadaniu i zwalnia po 10 min bezczynności (wartość konfigurowalna).

### Modele pomocnicze (w workerze)

- **Detekcja kół:** lekki detektor, Florence-2 albo Grounding DINO. Wybór rozstrzygnie spike.
- **Maski:** SAM 2, zarówno do precyzyjnego kształtu wykrytych kół, jak i do „dodaj koło” jednym kliknięciem (prompt punktowy).
- **Usuwanie tła z felgi referencyjnej:** BiRefNet lub rembg, uruchamiane przy dodawaniu felgi do biblioteki.

### Pipeline edycji

1. **Wejście:** JPEG/PNG/WebP/HEIC (`pillow-heif`), obrót według EXIF. Limit 40 MPx / 50 MB.
2. **Detekcja kół** (kilka sekund) i maski SAM 2. Użytkownik akceptuje lub koryguje je w UI.
3. **Maska kompozycji = całe koło** (opona + felga), lekko poszerzona. Prompt każe zachować oponę i zmienić tylko felgę. Opona jest generowana na nowo, ale ma wyglądać prawie identycznie. Dzięki temu nowa felga może mieć inną średnicę wizualną.
4. **Każde koło edytujemy osobno:** wycinek wokół koła przeskalowany do rozdzielczości roboczej (~1024 px, Turbo traci detal poniżej). Wywołanie z `image=[felga, wycinek]`, przy czym ostatni obraz wyznacza proporcje wyjścia. Ten sam seed i prompt dla wszystkich kół jednego zdjęcia.
5. **Kompozycja:** wynik skalujemy z powrotem do rozmiaru wycinka i wklejamy w oryginał z wygładzoną krawędzią maski. Piksele poza maską pozostają bit w bit identyczne.
6. **Wyjście:** zawsze PNG (bezstratny, żeby zachować gwarancję z punktu 5), z profilem ICC (np. Display P3 z iPhone'a) i EXIF bez danych GPS.

## Kolejka zadań

- Zadania przetwarzane są pojedynczo w kolejności FIFO, bo GPU jest jedno.
- Zadanie można anulować, a przerwanie następuje między krokami lub kołami.
- Zadanie przerwane przez awarię lub restart workera dostaje status `failed`, a UI pokazuje przycisk „ponów”. Bez automatycznych ponowień.
- Postęp raportowany per koło i per krok denoisingu.

## Baza danych (PostgreSQL)

- **Zadania:** status, parametry (seed, rozdzielczość, poszerzenie maski, dopisek do promptu), czasy, błąd.
- **Obrazy:** metadane i ścieżki do plików (oryginał, maski, wycinki, wynik).
- **Biblioteka felg:** wycięta felga, oryginał, opcjonalna nazwa, data dodania, licznik użyć i data ostatniego użycia. Sortowanie po ostatnim użyciu, bez filtrów.
- **Retencja:** historia i pliki są trzymane bezterminowo. Ręczne usuwanie (pojedyncze i zbiorcze) kasuje także pliki z dysku.

## Interfejs (UI)

- **Język:** polski, wszystkie teksty w jednym pliku.
- **Stack:** Jinja2 renderowane przez FastAPI + HTMX/Alpine.js. Tailwind przez standalone CLI, bez Node.js.
- **Strona główna** prowadzi przez kroki:
  1. Wgranie zdjęcia auta (drag & drop, wklejenie ze schowka).
  2. Automatyczna detekcja kół z nakładką masek. Klik w maskę usuwa ją, a klik w puste miejsce dodaje koło.
  3. Wybór felgi z biblioteki albo wgranie nowej, z podglądem wyciętej felgi do akceptacji.
  4. „Generuj”: postęp per koło i per krok (polling HTMX co ~1 s), z możliwością anulowania.
  5. Wynik z suwakiem przed/po, przyciskiem „ponów to koło” (inny seed) i pobieraniem PNG.
- **Sekcja „Zaawansowane”** jest domyślnie zwinięta i zawiera seed, rozdzielczość roboczą, poszerzenie maski i dopisek do promptu. Użyte wartości zapisujemy w historii zadania.
- **Podstrona „Historia”:** miniatury, podgląd, ręczne usuwanie.

## Konwencje inżynierskie

- Monorepo: `api/`, `worker/`, `shared/` jako workspace uv z jednym `uv.lock`.
- SQLAlchemy 2 + Alembic.
- pytest:
  - testy logiki kompozycji, w tym test, że piksele poza maską są identyczne,
  - testy API na prawdziwym PostgreSQL w Podmanie, z mockiem modelu AI.
- ruff, mypy, `justfile` (`up`, `down`, `worker`, `test`).
- `git init`, bez CI.

## Kolejność prac

### Krok 0: spike pomiarowy

Przed budową aplikacji wykonujemy krótki skrypt na przykładowych zdjęciach z darmowych źródeł (Wikimedia Commons / Unsplash). Potrzebne są 2–3 auta w różnych ujęciach (bok, 3/4 przodu, koło częściowo zasłonięte) i 2–3 felgi.

Spike mierzy czas i jakość dla:

- mflux Turbo q8 vs q4,
- rozdzielczości roboczej 1024 vs 1536 px,
- Florence-2 vs Grounding DINO jako detektora kół,
- dla porównania: istniejący pipeline diffusers bf16 z `../image-gen-model-test/generate.py`. To dwuetapowe ładowanie (text encoder → zwolnienie → transformer + VAE) z obejściem błędu `F.pad`. Zmierzone tam na M2 Max przy edycji 1024 px: ładowanie text encodera 19 s, kodowanie promptu 18 s, ładowanie transformera 9 s, generowanie 95 s (~10 s/krok), szczyt ~29 GB.

Wyniki spike'a ustalą domyślną kwantyzację, rozdzielczość roboczą i detektor.

### Krok 1+: aplikacja

Budowa zgodnie z powyższym projektem.

## Znane ryzyka

- **Brak natywnej maski w diffusers** dla Qwen-Image-2.1. Kompozycję robimy sami, a mflux ma `--mask-image`.
- **Spójność felg między kołami:** każde koło edytujemy osobno, więc felgi mogą się minimalnie różnić. Mitygacja: ten sam seed i prompt plus „ponów to koło”.
- **Artefakty Turbo w małych rozdzielczościach** (siatka pikseli poniżej ~1024 px). Wycinki skalujemy w górę do rozdzielczości roboczej.
- **Szum przy siatce dokładnie 1024 px** zgłaszany w ComfyUI (ComfyUI #16435). Do sprawdzenia w spike'u, alternatywnie 992/1056.
- **„Noise replay”:** ten sam seed i rozmiar co przy generowaniu obrazu wejściowego psuje edycję (diffusers #14824). W praktyce dotyczy tylko obrazów wygenerowanych tym samym modelem.
