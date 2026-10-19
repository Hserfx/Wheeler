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

### Ustawienia domyślne (wynik spike'a 01)

Pomiary i rysunki: [docs/spike/README.md](spike/README.md).

- **mflux z gita**, commit `b534021a9c8c45bb064959d754870fd98322e447` (Turbo jest na `main` od PR #859, nie ma go
  w wydanym 0.22.0). Model `QwenImage21Edit` z `ModelConfig.qwen_image_21_turbo()`: 8 kroków, guidance 1.
- **Kwantyzacja q8**, wagi skonwertowane w `~/.cache/wheeler/qwen-image-2.1-turbo-mflux-q8` (17,8 GB).
  q4 (`…-q4`, 10 GB) jako ustawienie zapasowe przy braku pamięci. q4 nie jest szybsze, tylko lżejsze.
- **`compute_precision=mx.float16`**, **`model.tiling_config = TilingConfig()`**, `mx.set_cache_limit(2 GB)`.
- **Rozdzielczość robocza 1024 px** (pole 1024², boki zaokrąglone do 32 px). 1536 px jest 3× wolniejsze i nie
  mieści się w pamięci przy q8. Szumu siatki przy 1024 nie stwierdzono (porównanie z 992/1056).
- **Referencje `[wycinek, felga]`** z `mask_image` (maska koła w rozmiarze roboczym) i jawnymi `width`/`height`.
  Maska mflux działa na pierwszej referencji, a proporcje wyjścia bierze z ostatniej.
- **Wycinek:** kwadrat wokół koła z marginesem 25% boku koła z każdej strony, przycięty do zdjęcia.
- **Maska kompozycji:** maska SAM poszerzona o 3% boku koła. Krawędź wygładzana do wewnątrz na pasie ~1,5% boku koła,
  więc piksele poza maską pozostają bit w bit.
- **Szablon promptu** (obraz 1 = wycinek, obraz 2 = felga):
  „Replace the rim of the car wheel in image 1 with the alloy rim design shown in image 2. Keep the tire, the car
  body, the wheel arch, the brake caliper position, the lighting, the shadows and the camera angle of image 1
  unchanged. The new rim must follow the perspective of the wheel in image 1 and fill the tire opening exactly.
  Photorealistic.” Dopisek użytkownika doklejamy na końcu.
- **Detektor: Florence-2-large** (`florence-community/Florence-2-large`, fp16 na MPS), zadanie
  `<CAPTION_TO_PHRASE_GROUNDING>` z frazą `wheel`. **Maski: SAM 2.1 hiera-large** (`facebook/sam2.1-hiera-large`
  przez transformers) z promptem box.
- **Zmierzone na M2 Max:** jedno koło ~110 s (q8, 1024 px), szczyt pamięci MLX 26,9 GB. Ładowanie wag ~4 s.
  Detekcja + maski poniżej 1 s na zdjęcie.

### Modele pomocnicze (w workerze)

- **Detekcja kół:** Florence-2-large (phrase grounding „wheel”), wybrany w spike'u zamiast Grounding DINO.
- **Maski:** SAM 2, zarówno do precyzyjnego kształtu wykrytych kół, jak i do „dodaj koło” jednym kliknięciem (prompt punktowy).
- **Usuwanie tła z felgi referencyjnej:** BiRefNet lub rembg, uruchamiane przy dodawaniu felgi do biblioteki.

### Pipeline edycji

1. **Wejście:** JPEG/PNG/WebP/HEIC (`pillow-heif`), obrót według EXIF. Limit 40 MPx / 50 MB.
   Gwarancja „bit w bit” obejmuje każde przyjęte wejście (ustalone po review): wynik zachowuje tryb
   oryginału (RGB, RGBA, L, LA; alfa bez zmian w całym obrazie), paleta jest rozwijana do RGB/RGBA,
   obraz 1-bitowy do L. Wejścia, których PNG wyniku nie odda wiernie (16 bitów na kanał, CMYK itp.),
   są odrzucane przy wgraniu z komunikatem.
2. **Detekcja kół** (kilka sekund) i maski SAM 2. Użytkownik akceptuje lub koryguje je w UI.
3. **Maska kompozycji = całe koło** (opona + felga), lekko poszerzona. Prompt każe zachować oponę i zmienić tylko felgę. Opona jest generowana na nowo, ale ma wyglądać prawie identycznie. Dzięki temu nowa felga może mieć inną średnicę wizualną.
4. **Każde koło edytujemy osobno:** wycinek wokół koła przeskalowany do rozdzielczości roboczej (1024 px, Turbo traci detal poniżej). Wywołanie z `image_paths=[wycinek, felga]` i maską koła, bo maska mflux działa na pierwszej referencji. Rozmiar wyjścia podajemy jawnie (`width`/`height`), bo domyślnie wyznacza go ostatni obraz. Ten sam seed i prompt dla wszystkich kół jednego zdjęcia.
5. **Kompozycja:** wynik skalujemy z powrotem do rozmiaru wycinka i wklejamy w oryginał z wygładzoną krawędzią maski. Piksele poza maską pozostają bit w bit identyczne.
6. **Wyjście:** zawsze PNG (bezstratny, żeby zachować gwarancję z punktu 5), z profilem ICC (np. Display P3 z iPhone'a) i EXIF bez danych GPS.

## Kolejka zadań

- Zadania przetwarzane są pojedynczo, bo GPU jest jedno. W obrębie rodzaju pracy obowiązuje FIFO
  (po chwili wstawienia do kolejki).
- **Świadome odstępstwo od czystego FIFO (ustalone po review):** krótka praca, na którą użytkownik
  czeka przy ekranie, idzie przed generowaniem. Kolejność: najpierw wycinanie felg do biblioteki
  (~1 s), potem detekcja kół i maski z kliknięcia (~1 s), na końcu generowanie i „ponów to koło”
  (~2 min na koło). Trwającej pracy nic nie przerywa. Bez tego klik w koło czekałby na wszystkie
  wcześniej zlecone generowania.
- Pozycja w kolejce pokazywana w UI liczy wszystko, co worker zrobi przed danym zadaniem: trwającą
  pracę, felgi w kolejce i zadania przed nim w powyższej kolejności (praca interaktywna zlecona
  później może je jeszcze wyprzedzić).
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
- **Szum przy siatce dokładnie 1024 px** zgłaszany w ComfyUI (ComfyUI #16435). Sprawdzone w spike'u na edycji wycinków: przy 1024 px brak szumu, wynik jak przy 992/1056.
- **Zapas pamięci:** q8 przy 1024 px ma szczyt 26,9 GB z 32 GB. Jeśli pojawi się swap, przełączamy na q4 (19 GB).
- **Przenoszenie zacisku hamulcowego** ze zdjęcia felgi na auto. Mitygacja: felga z usuniętym tłem (ticket 06), ewentualnie dopisek w prompcie.
- **„Noise replay”:** ten sam seed i rozmiar co przy generowaniu obrazu wejściowego psuje edycję (diffusers #14824). W praktyce dotyczy tylko obrazów wygenerowanych tym samym modelem.
