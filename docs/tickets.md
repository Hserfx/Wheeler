# Wheeler — tickety

Tickety to pionowe przekroje (tracer bullets) wynikające z [DESIGN.md](DESIGN.md), ułożone w kolejności zależności. Ticket można zacząć, gdy wszystkie blokujące go tickety są zrobione.

```
01 ─┐
    ├─► 03 ─┬─► 04 ─► 07
02 ─┘       ├─► 05, 06, 09, 10, 11
02 ─────────┴─► 08
06, 07, 09 ─► 12 ─► 13 ─┬─► 14 ─┐
                        └─► 15 ─┴─► 16 ─► 17
```

---

# 01: Spike — pomiar modelu i detektora

**What to build:** Krótki, jednorazowy eksperyment, który na prawdziwym M2 Max ustala, jakich ustawień domyślnych użyje aplikacja. Pobieramy przykładowe zdjęcia z darmowych źródeł (Wikimedia Commons / Unsplash): 2–3 auta w różnych ujęciach (bok, 3/4 przodu, koło częściowo zasłonięte) i 2–3 felgi. Mierzymy na nich czas, szczytowe zużycie pamięci i jakość podmiany felgi na wycinku wokół koła.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Przykładowe zdjęcia są w repo wraz ze źródłem i licencją każdego z nich
- [x] Wagi Qwen-Image-2.1-Turbo są skonwertowane do mflux q8 i q4 z lokalnego cache bf16
- [x] Zmierzone czasy i szczytowa pamięć dla kombinacji: q8 / q4 × 1024 / 1536 px, z rozbiciem na ładowanie modelu i generowanie
- [x] Ta sama edycja wykonana bazowym pipeline'em diffusers bf16 (dwuetapowe ładowanie z `image-gen-model-test`) jako punkt odniesienia
- [x] Porównanie Florence-2 i Grounding DINO pod kątem trafności i czasu wykrywania kół na zdjęciach przykładowych
- [x] Sprawdzone, czy siatka dokładnie 1024 px daje szum (porównanie z 992/1056)
- [x] Wyniki i obrazy porównawcze w raporcie, a wybrane ustawienia domyślne (kwantyzacja, rozdzielczość robocza, detektor, szablon promptu) dopisane do DESIGN.md

**Comments:**

- 2026-10-16: Raport w `docs/spike/README.md`, kod w `spike/`. Ustawienia domyślne: mflux z gita (commit `b534021a`,
  bo Turbo nie ma w wydanym 0.22.0), q8 + fp16 compute + kafelkowy VAE, 1024 px, `[wycinek, felga]` z maską,
  Florence-2-large + SAM 2.1 large. Kolejność `[felga, wycinek]` z pierwotnego DESIGN w 2 z 3 przypadków zwracała
  zdjęcie felgi, więc DESIGN poprawiony. Kod wycinania i kompozycji (`spike/src/wheeler_spike/wheel_crop.py`,
  z testami) do przeniesienia w tickecie 03.

---

# 02: Szkielet end-to-end (bez AI)

**What to build:** Najcieńsza działająca ścieżka przez wszystkie warstwy. `just up` uruchamia maszynę Podmana `wheels`, kontenery PostgreSQL i FastAPI oraz natywnego workera w tle. Pod `127.0.0.1:8000` jest polska strona (Jinja2 + HTMX/Alpine, Tailwind ze standalone CLI), na której wgrywam zdjęcie. Powstaje wtedy zadanie w kolejce w Postgresie, worker je odbiera, wykonuje atrapę przekształcenia, zapisuje wynik na wolumenie i oznacza zadanie jako gotowe, a strona pokazuje wynik. `just down` zatrzymuje wszystko.

**Blocked by:** None (can start immediately; może iść równolegle z 01)

**Status:** done

- [x] Monorepo `api/`, `worker/`, `shared/` jako workspace uv z jednym lockfile'em, `git init`
- [x] Maszyna Podmana `wheels` (2 CPU, 2 GB) tworzona automatycznie, jeśli nie istnieje
- [x] Port API opublikowany wyłącznie na `127.0.0.1`, niedostępny z sieci lokalnej
- [x] Schemat bazy zarządzany przez Alembic, migracje uruchamiane przy starcie
- [x] Worker pobiera zadania przez `FOR UPDATE SKIP LOCKED` i przetwarza jedno naraz
- [x] Pliki leżą na wolumenie dyskowym wspólnym dla API i workera, a w bazie są tylko ścieżki
- [x] Wszystkie teksty UI po polsku, w jednym pliku
- [x] `just up`, `just down`, `just worker` i `just test` działają, a ruff i mypy przechodzą
- [x] Test integracyjny: wgranie zdjęcia → zadanie → wynik, na prawdziwym Postgresie w Podmanie

**Comments:**

- 2026-10-15 (agent): zrobione na gałęzi `ticket-02`. Uruchamianie i testy opisane w `README.md` i `justfile`. Testy integracyjne startują własny kontener Postgresa na `wheels` (unikalna nazwa, losowy port 127.0.0.1), więc `just test` można odpalać równolegle w wielu worktree. Punkt rozszerzenia dla 03: `worker/src/wheeler_worker/processing.py` (`Processor`, `PROCESSORS`, `WHEELER_PROCESSOR`). Sprawdzone ręcznie: `just up` → wgranie zdjęcia przez HTTP → wynik PNG; port 8000 niedostępny z adresu LAN; `just down` zatrzymuje workera i kontenery.

---

# 03: Podmiana jednego koła

**What to build:** Pierwsza prawdziwa podmiana felgi. Wgrywam zdjęcie auta i zdjęcie felgi, a potem klikam w jedno koło na zdjęciu auta. SAM 2 wyznacza maskę całego koła (opona + felga, lekko poszerzoną). Wycinek wokół koła trafia do mflux Turbo razem z felgą referencyjną, z poleceniem podmiany felgi przy zachowaniu opony. Wynik jest skalowany z powrotem i wklejany w oryginał z wygładzoną krawędzią maski. Dostaję PNG w rozdzielczości oryginału, w którym zmieniło się tylko to koło.

**Blocked by:** 01, 02

**Status:** done

- [x] Kliknięcie w koło tworzy maskę i pokazuje ją jako nakładkę przed generowaniem
- [x] Wywołanie modelu przyjmuje obraz felgi i wycinek koła, a ustawienia domyślne pochodzą z wyników spike'a
- [x] Wynik ma dokładnie tę samą rozdzielczość co zdjęcie wejściowe
- [x] Test automatyczny: każdy piksel poza maską jest identyczny bit w bit z oryginałem
- [x] Wynik zapisywany zawsze jako PNG
- [x] Testy API używają mocka modelu AI, a logika kompozycji jest testowana bez modelu

**Comments:**

- 2026-10-16 (agent): zrobione na gałęzi `ticket-03`. Maska po kliknięciu też idzie przez kolejkę, bo SAM 2
  działa tylko w natywnym workerze: `Job.task` (`segment` / `generate`), nowy status `draft` (czeka na
  użytkownika), tabela `wheels` (klik, ścieżki maski i nakładki), migracja `0002`. Segmentacja ma pierwszeństwo
  przed generowaniem, FIFO po `queued_at`. `Processor` ma metody `segment` i `generate` (tylko pliki), modele
  są za interfejsami `Segmenter` (SAM 2.1) i `RimEditor` (mflux), ładowane leniwie. Zapisana maska jest już
  poszerzona o 3% boku koła, więc nakładka pokazuje dokładnie obszar, który może się zmienić. Kod kompozycji
  przeniesiony ze spike'a do `wheeler_worker/compositing.py`. Testy: `test_compositing.py`, `test_wheel_swap.py`
  (procesor z atrapami modeli), `test_upload_flow.py` (HTTP + worker), `test_queue.py`.
- Weryfikacja na prawdziwym modelu (SAM 2.1 large + mflux q8, `car_side_golf.jpg` 1920×1080, przez HTTP i
  worker pod `gpu-run.sh`): maska po kliknięciu 15 s (zimny start z importem torch), generowanie 121 s
  (z ładowaniem mflux; 8 kroków po ~9 s), szczyt pamięci procesu 28,6 GB. Wynik PNG 1920×1080, wszystkie
  2 009 257 pikseli poza maską identyczne bit w bit. Felga podmieniona poprawnie; model przeniósł też żółty
  zacisk ze zdjęcia felgi (znane ryzyko, ticket 06).

---

# 04: Automatyczna detekcja i edycja masek, wiele kół

**What to build:** Po wgraniu zdjęcia auta aplikacja sama znajduje wszystkie widoczne koła i pokazuje ich maski jako nakładkę. Klik w maskę ją usuwa, a klik w puste miejsce dodaje nowe koło (SAM 2 z promptem punktowym). Po zatwierdzeniu wszystkie wybrane koła są podmieniane, każde osobnym wywołaniem, z tym samym seedem i promptem.

**Blocked by:** 03

**Status:** done

- [x] Detekcja (detektor wybrany w spike'u) + SAM 2 kończy się w ciągu kilku sekund
- [x] Gdy nie wykryto żadnego koła, UI jasno o tym informuje i zachęca do dodania koła kliknięciem
- [x] Usuwanie i dodawanie masek działa przed uruchomieniem generowania
- [x] Wszystkie wybrane koła trafiają do jednego wyniku, a piksele poza wszystkimi maskami pozostają nienaruszone
- [x] Wszystkie koła jednego zadania używają tego samego seeda i promptu

**Comments:**

- Detekcja: Florence-2-large (`wheeler_worker/detector.py`) -> SAM 2.1 z promptem box (`Segmenter.segment_boxes`),
  jako `TaskKind.DETECT` w kolejce zaraz po wgraniu. Na prawdziwych modelach (M2 Max, `docs/spike/samples/`):
  ciepło 0,8–1,0 s na zdjęcie, zimny start z ładowaniem modeli 4–6 s; pierwszy odczyt wag z dysku (pusty cache
  stron) ~20 s. Maski obejmują oponę i felgę, jak w spike'u (Golf 2/2, Sorento 2/2 z kawałkiem cienia, Vios 3 koła
  aut + koło skutera do usunięcia kliknięciem).
- Klik w zdjęcie: w masce usuwa koło od razu (bez workera), obok masek dodaje koło (SAM 2, dwustopniowo jak w 03).

---

# 05: Postęp, anulowanie i obsługa awarii

**What to build:** W trakcie generowania widzę, które koło jest przetwarzane i na którym kroku denoisingu jest model. Mogę anulować zadanie. Jeśli worker padnie albo zostanie zrestartowany w trakcie zadania, zadanie dostaje status `failed`, a w UI pojawia się przycisk „Ponów”.

**Blocked by:** 03

**Status:** done

- [x] Postęp per koło i per krok odświeżany pollingiem HTMX co ~1 s
- [x] „Anuluj” zatrzymuje zadanie najpóźniej po bieżącym kroku, a zadanie dostaje status „anulowane”
- [x] Zadanie przerwane przez awarię lub restart workera ma status `failed`, bez automatycznego ponawiania
- [x] „Ponów” tworzy nowe zadanie z tymi samymi danymi wejściowymi i parametrami
- [x] Kolejne zadania czekają w kolejce FIFO, a UI pokazuje pozycję w kolejce

**Comments:**

- Postęp: `Job.progress` (JSONB, `wheeler_shared.progress.Progress`), flaga `Job.cancel_requested`,
  migracja `t05_progress`, nowy status `cancelled`. Anulowanie działa dla generowania (w kolejce: od razu,
  w trakcie: po bieżącym kroku albo przed kolejnym kołem). „Ponów” jest też dla zadań anulowanych.
- Awaria workera jest wykrywana przy jego ponownym starcie (`recover_interrupted_jobs`); dopóki worker
  nie wystartuje, zadanie przerwane twardym zabiciem procesu pokazuje się jako „Przetwarzanie…”.
- Weryfikacja na prawdziwym modelu (SAM 2 + mflux q8, Golf, 1024 px): postęp krok 1/8 po 55 s, potem co
  ~10 s; „Anuluj” w kroku 4 zatrzymał generowanie 11 s później (po tym kroku), status `cancelled`,
  bez `result.png`.

---

# 06: Biblioteka felg

**What to build:** Raz wgraną felgę mogę wybierać przy kolejnych zdjęciach bez ponownego wgrywania. Przy dodawaniu nowej felgi tło jest usuwane automatycznie, a ja widzę podgląd wyciętej felgi i mogę go zaakceptować albo odrzucić. Felgi mogą mieć opcjonalną nazwę i są sortowane po ostatnim użyciu.

**Blocked by:** 03

**Status:** done

- [x] Usuwanie tła (BiRefNet lub rembg) z podglądem do akceptacji przed zapisaniem do biblioteki
- [x] Przechowywane dane: wycięta felga, oryginał, opcjonalna nazwa, data dodania, licznik użyć, data ostatniego użycia
- [x] Wybór felgi z biblioteki na stronie głównej jako alternatywa dla wgrania nowej
- [x] Do modelu trafia wycięta felga, nie oryginał z tłem
- [x] Felgę można usunąć z biblioteki, a usunięcie nie psuje historii zadań, które jej używały

**Comments:**

- BiRefNet (`ZhengPeng7/BiRefNet`, fp16 na MPS) w kadrze wokół koła wykrytego przez Florence-2. Sam BiRefNet
  na całym zdjęciu felgi zamontowanej w aucie zostawiał karoserię (2 z 3 próbek ze spike'a). Klik w felgę na
  oryginale wybiera koło (gdy jest ich kilka) i wycina ponownie. Zweryfikowane na prawdziwych modelach na
  `docs/spike/samples/rim_*.jpg`: wszystkie trzy wycięte do samego koła (opona + felga), ~0,9 s na zdjęcie
  po załadowaniu (~4,5 s z ładowaniem). Na `rim_mesh_silver_angle` zostaje wąski pasek błotnika nad oponą.
- Zacisk hamulcowy widoczny przez szprychy należy do koła i zostaje na wyciętej feldze (zob. ryzyko „Przenoszenie
  zacisku hamulcowego” w DESIGN). Usunięcie tła ogranicza ryzyko, ale go nie eliminuje.
- Dodanie nowej felgi odbywa się zawsze przez bibliotekę (wgranie, podgląd, akceptacja); `POST /jobs` przyjmuje
  `rim_id` zamiast pliku felgi. Zadanie dostaje kopię wyciętej felgi w swoim katalogu, więc usunięcie felgi
  z biblioteki nie zmienia historii. Model widzi felgę na białym tle.

---

# 07: Ekran wyniku — przed/po i „ponów to koło”

**What to build:** Gotowy wynik oglądam na suwaku przed/po. Jeśli jedno koło wyszło gorzej, mogę wygenerować ponownie tylko to koło z innym seedem, a pozostałe koła zostają bez zmian. Wynik pobieram jako PNG.

**Blocked by:** 04

**Status:** done

- [x] Suwak przed/po działa na pełnym obrazie, z możliwością powiększenia okolicy koła
- [x] „Ponów to koło” generuje tylko wybrane koło z nowym seedem i podmienia je w wyniku
- [x] Piksele poza maską ponawianego koła, łącznie z pozostałymi kołami, pozostają bez zmian
- [x] Pobieranie zwraca PNG w rozdzielczości oryginału

**Comments:**

- Ekran wyniku: `_result_compare.html` (suwak przed/po na pełnym obrazie, przyciski „Koło N”
  powiększające okolicę koła według `Wheel.box`, „Cały obraz”), `GET /jobs/{id}/download`
  (PNG jako załącznik, rozdzielczość oryginału, ICC/EXIF bez GPS). Suwak i powiększanie są
  sprawdzone testami na poziomie HTML; w przeglądarce nie były klikane (brak przeglądarki
  sterowanej z agenta), matematyka powiększenia sprawdzona ręcznie na boxach z prawdziwej detekcji.
- „Ponów to koło”: `POST /jobs/{id}/wheels/{wheel_id}/regenerate` (tylko `done`),
  `TaskKind.REGENERATE_WHEEL`, migracja `t07_wheel_seed` (`wheels.seed`, `wheels.pending_seed`).
  Koło jest składane na bieżącym wyniku, nowy plik `result-<n>.png`; błąd, anulowanie albo awaria
  workera wracają do poprzedniego wyniku. Test bit w bit: `tests/test_wheel_swap.py`
  i `tests/test_result_screen.py`.
- Prawdziwe modele (Golf 1920x1080, felga multispoke, q8, 1024 px, przed merge'em 06): detekcja
  2 kół 16 s, generowanie obu 250 s, ponowienie koła 2 130 s (seed 42 -> 849551686); poza maską
  ponawianego koła 0 zmienionych pikseli (koło 1 identyczne), w masce zmienione 61 366 z 64 241,
  ICC zachowany.

---

# 08: Historia

**What to build:** Osobna podstrona „Historia” pokazuje wszystkie zadania jako miniatury ze statusem i datą. Mogę otworzyć dowolne zadanie, żeby zobaczyć wejście, wynik i parametry, oraz usuwać zadania pojedynczo lub zbiorczo, razem z ich plikami na dysku.

**Blocked by:** 02

**Status:** done

- [x] Lista zadań od najnowszych, z miniaturą, statusem i datą
- [x] Widok szczegółów zadania z wejściem, wynikiem i użytymi parametrami
- [x] Usuwanie pojedyncze i zbiorcze (zaznaczanie), z potwierdzeniem
- [x] Usunięcie kasuje pliki zadania z wolumenu, a test to weryfikuje
- [x] Brak automatycznego sprzątania, historia jest trzymana bezterminowo

**Comments:**

- 2026-10-15 (agent): zrobione na gałęzi `ticket-08`, testy w `tests/test_history.py`. `/history` to siatka miniatur (generowane leniwie przez `/jobs/{id}/thumbnail` jako `<plik>.thumb.jpg` w katalogu zadania; Pillow doszedł do zależności `api`). Szczegóły zadania (`/jobs/{id}`) pokazują wszystkie klucze z `Job.params` bez listy na sztywno; etykiety można dopisać w `PARAM_LABELS` (`texts.py`). Potwierdzenie usuwania to osobna strona `GET /history/delete?ids=…` (wspólna dla pojedynczego i zbiorczego), a `POST /history/delete` usuwa. Logika w `wheeler_shared/history.py` (`delete_jobs`): kasuje cały katalog `jobs/<hex>` (także pliki, których baza nie zna), pliki spoza niego zostają, zadania `running` są pomijane. Konwencja dla 03/09: wszystkie pliki zadania zapisuj w jego katalogu.

---

# 09: Sekcja „Zaawansowane”

**What to build:** Na stronie głównej jest zwijana sekcja „Zaawansowane”, domyślnie zwinięta, w której mogę ustawić seed, rozdzielczość roboczą, poszerzenie maski i dopisek do promptu. Bez jej otwierania wszystko działa na rozsądnych ustawieniach domyślnych. Użyte wartości są zapisywane w zadaniu.

**Blocked by:** 03

**Status:** done

- [x] Sekcja domyślnie zwinięta, z ustawieniami domyślnymi z wyników spike'a
- [x] Puste pole seed oznacza losowy seed, a wylosowana wartość jest zapisywana w zadaniu
- [x] Walidacja zakresów (rozdzielczość jako wielokrotność 32, rozsądne granice poszerzenia maski) z czytelnym komunikatem
- [x] Użyte wartości widoczne w szczegółach zadania w historii

**Comments:**

- Zakresy (`wheeler_shared/edit_params.py`): rozdzielczość 768–1152 px co 32 (1536 px przy q8 wchodzi w swap; 1152 px ma pole 1,27 razy większe niż 1024, szacunkowo ~29 GB), poszerzenie maski 0–10% boku koła (w formularzu w procentach, w `Job.params` jako ułamek), seed 0–2³¹−1, dopisek do 300 znaków. Walidacja w przeglądarce (atrybuty `min`/`max`/`step`) i na serwerze (400 z komunikatem, sekcja otwarta, wpisane wartości zachowane).
- Zadanie zapisuje w `params` użyte wartości (także domyślne) już przy wgraniu zdjęcia, więc historia nie zależy od późniejszych zmian ustawień domyślnych. Komunikat „Użyto parametrów domyślnych” zostaje dla zadań bez zapisanych parametrów.
- Weryfikacja testami z atrapami modeli (bez prawdziwego modelu), w tym przepływ przez workera: rozdzielczość, seed i dopisek trafiają do edytora, poszerzenie maski zmienia maskę przy segmentacji.

---

# 10: Obsługa wejścia i metadanych

**What to build:** Mogę wgrać zdjęcie w typowym formacie, łącznie z HEIC z iPhone'a, przez przeciągnięcie albo wklejenie ze schowka. Zdjęcie jest poprawnie obrócone według EXIF. Za duże pliki są odrzucane z czytelnym komunikatem. Wynik zachowuje profil kolorów oryginału i jego EXIF, ale bez lokalizacji GPS.

**Blocked by:** 03

**Status:** done

- [x] Akceptowane formaty: JPEG, PNG, WebP i HEIC, a inne są odrzucane z komunikatem
- [x] Obrót według orientacji EXIF stosowany przed detekcją i kompozycją
- [x] Limity 40 MPx / 50 MB sprawdzane z czytelnym komunikatem
- [x] Wgrywanie przez drag & drop, wybór pliku i wklejenie ze schowka
- [x] Wynikowy PNG zawiera profil ICC oryginału (np. Display P3) i EXIF bez danych GPS
- [x] Test: zdjęcie HEIC z orientacją i GPS daje poprawnie obrócony PNG z ICC i bez GPS

**Comments:**

- Walidacja: `wheeler_api/uploads.py` (`save_image_upload`), wspólna dla zdjęcia auta i felgi (i biblioteki felg z 06). Format po zawartości, limity 50 MB (strumieniowo) i 40 MPx (z nagłówka, bez dekodowania).
- HEIC i zdjęcia z orientacją EXIF dostają obrócony podgląd `<plik>.preview.jpg` w pełnym rozmiarze (filtr `display_path` w szablonach), więc kliknięcia są w tym samym układzie co w workerze niezależnie od tego, jak przeglądarka traktuje EXIF.
- Wynik: `photos.read_metadata` + `save_png(..., metadata=)` (ICC 1:1, EXIF bez GPS i orientacji). Test bit w bit po obrocie EXIF: `tests/test_photo_input.py`. Testowe pliki z metadanymi generuje `tests/photo_files.py`.
- Drag & drop i wklejanie (Alpine, `DataTransfer` do zwykłego pola pliku) sprawdzone testem obecności w HTML, bez ręcznego testu w przeglądarce.

---

# 11: Cykl życia modelu w pamięci

**What to build:** Worker nie zajmuje pamięci, dopóki nie jest potrzebny. Model ładuje się przy pierwszym zadaniu, zostaje w pamięci na kolejne zadania i zwalnia się po 10 minutach bezczynności (czas konfigurowalny). W UI widzę stan modelu: „uśpiony”, „ładowanie” albo „gotowy”.

**Blocked by:** 03

**Status:** done

- [x] Po starcie workera model nie jest załadowany
- [x] Pierwsze zadanie ładuje model, a UI pokazuje stan „ładowanie” zamiast zablokowanego postępu
- [x] Po skonfigurowanym czasie bezczynności model jest zwalniany, a pamięć faktycznie wraca do systemu
- [x] Czas bezczynności ustawiany przez konfigurację, domyślnie 10 min
- [x] Stan modelu (uśpiony / ładowanie / gotowy) widoczny w UI

**Comments:**

- Weryfikacja na prawdziwym modelu (M2 Max, SAM 2.1 + mflux q8, generowanie 512 px, czas bezczynności 5 s):
  `phys_footprint` procesu 0,05 GB po starcie → 3,2 GB po segmentacji → 21,3 GB po generowaniu
  (MLX active 17,2 GB + cache 2,1 GB) → **1,6 GB po zwolnieniu** (MLX active 0, cache 0; MPS driver 1,0 GB).
  Ponowne zadanie ładuje SAM 2 znowu (3,8 GB), drugie zwolnienie → 1,4 GB.
  Stany: asleep → loading → ready → loading (mflux) → ready → asleep → loading → ready → asleep.
- Stan modelu w tabeli `worker_state`; po awarii workera zostaje ostatni stan aż do restartu
  (start workera zapisuje „uśpiony”). Brak heartbeatu.

---

# 12: Rozmiar felgi (plus-sizing)

**What to build:** Przed „Generuj”, obok wyboru felgi, wybieram rozmiar nowej felgi względem obecnej: `−2″ · −1″ · bez zmian · +1″ · +2″`. Zmienia się tylko średnica felgi, a zewnętrzna średnica koła zostaje: większa felga oznacza niższy profil opony, mniejsza wyższy (jak przy prawdziwym plus-sizingu). Całość mieści się w obecnej masce koła (opona + felga), więc piksele poza maską pozostają bit w bit.

Rozmiarem steruje geometria, nie prompt (sam prompt model ignoruje, zob. komentarze). Przy stopniu ≠ 0 worker przed wywołaniem modelu wyznacza elipsę felgi (SAM 2) i przeskalowuje wycinek koła promieniowo: felga rośnie albo maleje o ~4% średnicy koła na cal, bok opony się dopasowuje, a zewnętrzna krawędź opony zostaje. Model dostaje wycinek z felgą już w docelowym rozmiarze i podmienia tylko jej wzór.

**Blocked by:** 06, 07, 09

**Status:** done

- [x] Wybór stopnia w sekcji wyboru felgi na stronie zadania, w formularzu „Generuj” (`#job-form`); domyślnie „bez zmian”. `POST /jobs` przy wgraniu go nie przyjmuje
- [x] Stopień zapisany w `Job.params` (nowy klucz w `wheeler_shared/edit_params.py`), z walidacją zakresu −2…+2 po stronie serwera; brak klucza oznacza „bez zmian”
- [x] Jedno ustawienie na całe zadanie: wszystkie koła, „Ponów” (`retry.py`) i „Ponów to koło” używają tej samej wartości
- [x] Przy „bez zmian” prompt jest identyczny z obecnym szablonem (istniejące zadania i testy bez zmian)
- [x] Prompt jest identyczny dla każdego stopnia (obecny szablon „bez zmian”); osobny szablon plus-sizingu znika
- [x] Ekran wyniku pokazuje wybrany stopień (np. „Felga: +2″”) obok seeda, tylko gdy ≠ „bez zmian”; siatka Historii bez zmian (szczegóły zadania i tak pokazują wszystkie klucze `Job.params`, etykieta w `PARAM_LABELS`)
- [x] Przy stopniu ≠ 0, przed załadowaniem mflux, worker wyznacza dla każdego koła zadania elipsę felgi: maska samej felgi z SAM 2 (punkt w środku koła + punkty negatywne na oponie albo box ~0,8 bboxa koła), elipsa dopasowana (`cv2.fitEllipse`) tylko do tej części konturu felgi, która graniczy z oponą (odporność na zasłonięcie). Potem SAM 2 jest zwalniany, jak dotąd przed mflux
- [x] Zewnętrzna krawędź opony = elipsa felgi × k (ten sam środek i obrót; felga i ściana opony są koncentryczne także w 3/4), k z profilu maski koła; gdy k wypada poza 1,15–1,6 (np. zasłonięcie), przyjmowane jest typowe k = 1,32
- [x] Radialny warp w układzie tej (obróconej) elipsy, odwzorowanie z prototypu: dla ρ = promień znormalizowany do zewnętrznej krawędzi opony, ρ_f = obecna felga, ρ_c = ρ_f + stopień × 0,04: `[0, ρ_f] → [0, ρ_c]` liniowo (felga), `[ρ_f, 1] → [ρ_c, 1]` liniowo (bok opony), ρ ≥ 1 bez zmian. Docelowy rozmiar nie jest przycinany ani korygowany
- [x] Przy „bez zmian” wycinek trafia do modelu bez warpu i bez wyznaczania elipsy (zachowanie i czas jak dotąd)
- [x] Wycinek dla modelu i pomiar felgi zawsze z oryginalnego zdjęcia, także przy „Ponów to koło” (bez podwójnego przeskalowania); składanie jak dotąd na bieżącym wyniku, piksele poza maską bit w bit
- [x] Gdy felgi nie da się wyznaczyć (brak maski albo średnica felgi poza 40–95% średnicy koła), zadanie kończy się błędem z nowym kodem `rim_size_unavailable` i czytelnym komunikatem (np. „Nie udało się wyznaczyć felgi na kole N — wygeneruj bez zmiany rozmiaru”)
- [x] Obrazy diagnostyczne (elipsa felgi i opony na wycinku, wycinek po warpie) w katalogu koła; bez zmian w bazie i w UI
- [x] `docs/DESIGN.md` opisuje sterowanie geometryczne zamiast szablonu plus-sizingu
- [x] Testy na atrapach modeli: geometria warpu (piksele poza elipsą koła i poza maską bit w bit, felga po warpie ma docelowy promień, przy 0 brak warpu), przekazanie warpowanego wycinka do edytora przez workera (także przy „Ponów to koło” — wycinek z oryginału), błąd `rim_size_unavailable`; dotychczasowe testy parametru i ekranu wyniku zostają

**Comments:**

- 2026-10-19 (grilling): ustalone z użytkownikiem. Zmiana średnicy całego koła (wejście w nadkole, domalowanie szczeliny) świadomie poza zakresem, bo wymaga maski obejmującej karoserię. Cale są przybliżone: z samego zdjęcia nie znamy obecnego rozmiaru felgi, sterujemy proporcją średnicy felgi do średnicy koła. Weryfikacja tylko testami na atrapach; jakość na prawdziwym modelu użytkownik ocenia sam w trakcie używania. Jeśli model będzie ignorował polecenie, następny krok to osobny ticket na sterowanie geometryczne (np. wskazówka z docelowym okręgiem felgi).
- 2026-10-19 (agent): zrobione na gałęzi `ticket-12`. Parametr `rim_size` (`wheeler_shared/edit_params.py`:
  `RIM_SIZE`, `EditParams.rim_size`, `parse_rim_size`) przychodzi tylko z `POST /jobs/{id}/generate` (radia
  `form="job-form"` w sekcji felgi w `job.html`); poza zakresem albo nie-liczba: 400 ze stroną zadania. Zapisywany
  także `0`, brak pola zostawia zapisaną wartość. Prompt: `build_prompt(suffix, rim_size=…)` i
  `PLUS_SIZING_TEMPLATE` w `wheeler_worker/wheel_swap.py`; przy stopniu ≠ 0 z szablonu znika też „Keep the tire”
  (tak samo sprzeczne z pocienianiem boku opony jak „fill the tire opening exactly”). „Ponów” i „Ponów to koło”
  dostają wartość z `Job.params` bez dodatkowego kodu. Ekran wyniku: `data-role="result-rim-size"` w liście kół
  `_result_compare.html`; etykieta w `PARAM_LABELS`, wartość jak „+2″”/„−1″”/„bez zmian”. Testy:
  `tests/test_rim_size.py` (atrapy modeli). Na prawdziwym modelu nie sprawdzane (zgodnie z grillingiem).
- 2026-10-19 (agent, poprawki po code review, gałąź `review-fixes-12`): walidacja `rim_size` jak pola
  „Zaawansowane” (`advanced.parse_rim_size` -> `InvalidParam` -> `error_message`; placeholdery rozdzielczości
  przemianowane na `{resolution_min}`/`{resolution_max}`), komunikat z minusem typograficznym („od −2″ do +2″”).
  Niezmienniki stopnia w `edit_params.py`: `RIM_SIZES` (zakres) i `rim_size_change` -> `RimSizeChange(inches,
  larger)` albo `None` („bez zmian”); korzystają z nich `advanced.rim_size_text` i `build_prompt`. Znaki
  `sign_plus`/`sign_minus` w `texts.py`. Kontekst szablonów: `saved_rim_size` (zapisany, liczony raz)
  i `chosen_rim_size` (zaznaczony w pickerze). Ekran wyniku: „Felga: +2″” przy seedzie każdego koła.
  `PLUS_SIZING_TEMPLATE` (i DESIGN) dostał „Keep the tire tread and the outer edge of the tire unchanged.”;
  prompt przy 0 bez zmian. README: wzmianka o `rim_size`.
- 2026-10-19 (test użytkownika): zadanie 6 (Clio, 2 koła, widok z boku) z +2″ — felga bez zmiany średnicy,
  zmienił się tylko wzór. `Job.params` miał `rim_size: 2`, worker na nowym kodzie, prompt +2″ poprawnie
  zbudowany: model (Turbo, 8 kroków, maska całego koła) ignoruje polecenie rozmiaru i trzyma geometrię wycinka.
- 2026-10-19 (prototyp): gałąź `prototype/plus-sizing-warp` (`worker/prototype_plus_sizing/`: `run.py`,
  `VERDICT.md`, arkusze). Wstępny radialny warp felgi działa: ρ felgi (średnica felgi / koła) kontrola 0,725;
  warp +2″ → 0,825 (≈ +2,5″) z promptem bazowym i 0,835 z promptem +2″; warp −2″ → 0,675 (≈ −1,3″, model
  częściowo wraca do oryginału); koło 12 warp +2″ → 0,805 (+2,0″). Jedno auto, jeden seed, widok z boku.
- 2026-10-19 (rozpoznanie elipsy felgi, ta sama gałąź, `rim_ellipse/NOTES.md` + nakładki): metoda jasności
  z prototypu zawodzi w 3/4 i przy ciemnych felgach (środek bboxa koła przesunięty o 0,2–0,4 półosi od środka
  felgi); SAM 2 dał maskę samej felgi na 10/10 kół (także 3/4 i syntetycznie czarne felgi), reguła wyboru
  kandydata multimask w notatkach; stała ρ = 0,7 myli się średnio o ~1,8″. Etap SAM ~20 s z ładowaniem,
  szczyt ~3,8 GB.
- 2026-10-19 (grilling, ponowne otwarcie): ticket wraca do `ready-for-agent` zamiast osobnego ticketu.
  Ustalone: jeden prompt dla wszystkich stopni; brak korekty niedostrzału przy zmniejszaniu (zapisany jako
  znane zachowanie) i brak przycinania docelowego rozmiaru; elipsa liczona w workerze przy generowaniu, bez
  bazy i bez podglądu w UI; wycinek zawsze z oryginału; brak felgi = błąd zadania; typowe k = 1,32 jako
  zapas przy niepewnej krawędzi opony. Weryfikacja tylko testami na atrapach, jakość na prawdziwym modelu
  ocenia użytkownik.
- 2026-10-19 (agent, sterowanie geometryczne, gałąź `ticket-12-warp`): szablon plus-sizingu usunięty,
  `build_prompt(suffix)` daje ten sam prompt dla każdego stopnia. Geometria w nowym
  `wheeler_worker/rim_geometry.py` (numpy + OpenCV, `opencv-python` dopisany do zależności workera, i tak
  przychodził z mflux): `measure_rim` (reguła wyboru kandydata z `rim_ellipse/NOTES.md`, elipsa
  `cv2.fitEllipse` tylko z brzegu felgi graniczącego z oponą, k = 20. percentyl profilu maski koła,
  poza 1,15–1,6 `TYPICAL_K` = 1,32, średnica pionowa felgi 40–95% wysokości koła), `warp_rim`
  (odwzorowanie z prototypu, `cv2.remap` Lanczos, piksele ρ ≥ 1 kopiowane z wejścia bit w bit),
  `ungrown_wheel_mask` (maska koła bez `mask_grow`, jako przybliżenie maski SAM) i `outline_rim`
  (diagnostyka). SAM 2: nowa metoda `Segmenter.segment_rim` (punkt w centroidzie + 8 negatywów na
  oponie oraz box 0,8 bboxa + punkt, oba multimask; 6 kandydatów); atrapa `placeholder` też ją ma.
  `WheelSwapProcessor.generate`/`regenerate_wheel`: przy stopniu ≠ 0 `_measure_rims` na oryginale
  przed `_prepare_edit` (tam zwolnienie SAM 2 przed mflux); `_WheelEdit.apply(original, base, …)`
  wycina z oryginału, warpuje i składa na `base` (bieżący wynik przy „Ponów to koło”). Pliki
  diagnostyczne `rim_ellipse.png` i `warped.png` w `wheels/<id>/`; `crop.png` zostaje wycinkiem bez
  warpu. Nowy kod `Failure.RIM_SIZE_UNAVAILABLE` (`rim_size_unavailable`) z tekstem w `texts.py`.
  Testy: `tests/test_rim_geometry.py` (warp, pomiar, zasłonięcie, k zapasowe, odrzucenie felgi),
  `tests/test_rim_size.py` (procesor z atrapami i HTTP: warpowany wycinek trafia do modelu przy
  generowaniu, „Ponów” i „Ponów to koło” z oryginału bez podwójnego warpu, przy 0 brak pomiaru,
  błąd `rim_size_unavailable` na stronie zadania). DESIGN i README zaktualizowane.
  Odstępstwa: komunikat błędu nie podaje numeru koła (w `Job.error` leży sam kod, bez parametrów;
  numer koła trafia tylko do logu workera): „Nie udało się wyznaczyć felgi na jednym z kół, więc nie
  da się zmienić jej rozmiaru. Wygeneruj bez zmiany rozmiaru.” „Ponów to koło” także przy „bez zmian”
  wycina teraz z oryginału (wcześniej z bieżącego wyniku), bo to jedna ścieżka kodu. Adapter SAM 2
  (`Sam2Segmenter.segment_rim`) nie był uruchamiany na prawdziwym modelu (zgodnie z grillingiem:
  tylko atrapy).
- 2026-10-19 (agent, poprawki po code review, gałąź `review-fixes-12-warp`): wybór kandydata SAM tylko
  z zakresu reguły z `rim_ellipse/NOTES.md` (ρ_y 45–92%, `RIM_CANDIDATE_RANGE`); 40–95%
  (`RIM_DIAMETER_RANGE`) zostaje jako ostateczny warunek elipsy dopasowanej do brzegu przy oponie
  (`rim_size_unavailable`). Warp tylko w masce koła, którą zmienia model (`Image.composite` z `crop_mask`
  w rozdzielczości roboczej): krawędź opony = felga × k potrafiła wyjść poza maskę, a model widział tam
  przeskalowany bok opony, który kompozycja odcinała (szew). Nowe testy: kandydat w pasie 40–45% / 92–95%
  przegrywa z typową felgą mimo wyższej oceny SAM; poza maską wycinek dla modelu = wycinek bez warpu;
  SAM mierzy felgę i jest zwalniany przed pierwszym krokiem mflux (atrapa `DiscSegmenter.log`); zgodność
  `Ellipse` z konwencją OpenCV przy obrocie. Refaktory: `RimGeometry.k` -> `tire_ratio` (stałe
  `TIRE_RATIO_RANGE`, `TYPICAL_TIRE_RATIO`, `TIRE_RATIO_PERCENTILE`), `_WheelEdit.apply(..., rim_geometry=)`,
  `_measure_rims(..., rim_size=, mask_grow=)` zamiast całego `EditParams`; `grow_wheel_mask`
  i `ungrown_wheel_mask` w `compositing.py` ze wspólnym wzorem poszerzenia; `Ellipse.to_unit`/`from_unit`
  (obrót i normalizacja w jednym miejscu) i `wheel_ellipse` (elipsa wpisana w bbox maski koła) używane
  przez pomiar, `Sam2Segmenter`, placeholder i atrapy; `RimCandidate` zamiast `tuple[Image.Image, float]`;
  wspólne `ellipse_inside`/`wheel_photo` w `tests/helpers.py`. Bez łączenia rozmiaru felgi z geometrią
  w jeden typ: stopień jest jeden na zadanie (`_WheelEdit.rim_size`), geometria na koło, więc wspólny typ
  tylko by je powielał. Punkty negatywne i box felgi dla SAM przesunięte o pół piksela (środek bboxa
  w układzie środków pikseli, jak centroid) — pomijalne.
- 2026-10-19 (test użytkownika + grilling): efekt po warpie oceniony jako nienaturalny. Felga rośnie,
  a opona stoi w miejscu, a +2″ na zdjęciu zmienia się tylko trochę. Kierunek: rośnie całe koło (felga
  i zewnętrzna krawędź opony), a dzisiejszy plus-sizing ze stałą średnicą koła zostanie zastąpiony.
  Najpierw prototyp (ticket 13), ticket implementacyjny dopiero po jego werdykcie. Ten ticket zostaje
  `done`.

---

# 13: Prototyp: powiększanie całego koła

**What to build:** Prototyp (gałąź `prototype/…`, kod poza produkcyjnym workerem, jak `prototype/plus-sizing-warp`) sprawdzający na prawdziwych zdjęciach i prawdziwym modelu nowe znaczenie stopnia rozmiaru felgi: przy „+N″” rośnie **całe koło**, nie tylko felga. Wynik to arkusze przed/po i `VERDICT.md`, na których podstawie powstanie ticket implementacyjny. Kodu produkcyjnego (worker, UI, `Job.params`) ten ticket nie zmienia.

Ustalenia z grillingu (2026-10-19), które prototyp realizuje:

- **„+N″” = felga +N″ i zewnętrzna średnica koła +N″:** bok opony zostaje tej samej grubości w rzeczywistości („ta sama opona na większej feldze”). „−N″” analogicznie.
- **Skala cala ze znanego rozmiaru felgi:** 1″ = średnica widocznej felgi w px / (obecny rozmiar felgi + 1,4″). 1,4″ to rant felgi (kołnierz ~17–18 mm z każdej strony, wymiary ETRTO), widoczny na zdjęciu, a nieliczony w rozmiarze nominalnym. Stała nie jest weryfikowana na zdjęciach ze znanym rozmiarem. Obecny rozmiar to później opcjonalne pole „Obecna felga” (14″–22″ co 1″, domyślnie 17″), a w prototypie parametr skryptu z domyślnym 17″.
- **Punkt styku z ziemią stoi w miejscu:** najniższy punkt elipsy opony, także w ujęciu 3/4. Koło rośnie w górę i na boki, w nadkole, a przy zmniejszaniu opada do tego punktu. Opona nie wchodzi w podłoże i nie zawisa nad nim.
- **Karoseria zasłania koło i zostaje bit w bit:** maska karoserii z SAM 2 (punkt na błotniku nad nadkolem, ewentualnie z punktami negatywnymi na kole). Powiększone koło zajmuje nową elipsę opony poza maską karoserii, więc góra opony chowa się za błotnikiem.
- **Zmniejszanie:** odsłonięty obszar (stara maska koła minus nowe koło) w wycinku dla modelu jest wstępnie wypełniony, np. rozmytym, przyciemnionym kolorem z okolicy nadkola, a model domalowuje go w obrębie maski edycji. Maska edycji = (stare koło ∪ nowe koło) minus karoseria.
- **Felga** mierzona jak w tickecie 12 (SAM 2, elipsa z brzegu felgi graniczącego z oponą). Prompt bez zmian (ten sam dla każdego stopnia).
- **Zła maska karoserii** (pusta, obejmująca koło albo cały wycinek) kończy później zadanie błędem `rim_size_unavailable`, wspólnym z brakiem felgi. Prototyp ustala, po czym poznać złą maskę.

**Blocked by:** 12

**Status:** done

- [x] Skrypt prototypu uruchomiony na prawdziwym modelu (SAM 2 + mflux, ustawienia jak w produkcji) na zdjęciach Golf (bok), Sorento (3/4), Vios (zasłonięte) i Clio (zadanie użytkownika), dla stopni −2″, −1″, +1″, +2″ i kontroli „bez zmian”, przy obecnej feldze 17″ — **zawężone decyzją użytkownika (2026-10-19):** mflux tylko na Golfie i Sorento (wszystkie stopnie); Vios i Clio pominięte, bo użytkownik przetestuje je sam (dla nich SAM 2: felga, maska karoserii, wejścia po warpie)
- [x] Arkusze przed/po na każde zdjęcie: oryginał, wycinek po warpie (z zaznaczoną elipsą felgi, nową elipsą opony i maską karoserii), wynik modelu (Vios i Clio bez kolumny wyniku, zob. wyżej)
- [x] Pomiar wyniku: średnica felgi i koła po modelu (SAM 2 na wyniku) w calach według tej samej skali, porównana z docelową; kryterium sukcesu odchylenie ≤ ~0,5″ przy ±2″ — **pomiar wykonany; kryterium spełnia Golf (felga ≤ 0,48″, koło z sylwetki ±1,75″), felga Sorento (3/4) nie (odchylenie 0,4–2,0″)**
- [x] Automatyczne sprawdzenie, że piksele maski karoserii i piksele poza maską edycji są bit w bit z oryginałem
- [x] Maska karoserii poprawna na wszystkich zdjęciach zestawu (bok, 3/4, zasłonięte), z nakładkami do oceny; opisane przypadki, w których zawodzi, i proponowane kryteria „zła maska” dla `rim_size_unavailable`
- [x] `VERDICT.md`: wynik każdego kryterium (rozmiar, karoseria bit w bit, maska karoserii), miejsce na ocenę wyglądu przez użytkownika (koło naturalnie wypełnia nadkole, wnętrze nadkola przy −2″ wiarygodne), czasy i szczyt pamięci, rekomendacja dalszych kroków. Jeśli model „cofa” rozmiar przy −N″: propozycja korekty niedostrzału albo rezygnacji z wartości ujemnych. Proponowany podział ticketu implementacyjnego (pole „Obecna felga” w UI i `Job.params`, maska karoserii z błędem, nowy warp zastępujący warp z ticketu 12, zmniejszanie)

**Comments:**

- 2026-10-19 (grilling): weryfikacja tym razem na prawdziwym modelu, bo implementacja ticketu 12 sprawdzona
  tylko atrapami trafiła na model, który ignorował prompt. Ticket implementacyjny powstaje dopiero po
  werdykcie, bo jego zakres zależy od wyników (np. czy zostają wartości ujemne).
- 2026-10-19 (agent, prototyp): gałąź `prototype/whole-wheel-resize`, `worker/prototype_whole_wheel/`:
  `whole_wheel.py` (geometria i kryteria, testy na syntetycznych kołach `test_whole_wheel.py`), `run.py`
  (etapy detect/sam/plan/gen/measure/silhouette/report na SAM 2.1 i mflux q8 jak w produkcji,
  reużywa `rim_geometry`, `compositing`, adapterów SAM 2 i mflux), `VERDICT.md`, arkusze
  `out/sheet_*.jpg`, nakładki karoserii `out/body_*.jpg`, liczby `out/*.json`. Werdykt: całe koło
  rośnie i maleje zgodnie z celem; zewnętrzna średnica z sylwetki ±1,75–2,0″ przy ±2″ na Golfie
  i Sorento (kandydaci sylwetki co 0,25″, więc odchylenie 0 albo 0,25″ to rozdzielczość siatki,
  nie zmierzona dokładność), punkt styku stoi (≤ 1 px). Felga: Golf (bok) odchylenie ≤ 0,48″ przy
  ±2″; Sorento (3/4) nie spełnia (felga mniejsza już w kontroli: 15,7″ i 16,4″ zamiast 17″, zmiana
  względem kontroli 22–65% stopnia przy ±2″ i 7–63% przy ±1″; zachowanie modelu w 3/4, nie
  warpu). Chroniona maska karoserii (`body_effective`: surowa maska SAM minus wnętrze nadkola, po
  czyszczeniu, bez pikseli koła) i piksele poza maską edycji: 0 zmienionych pikseli w 16 przebiegach
  ze zmianą rozmiaru; surowa maska SAM: 0–2508 px na koło (wnętrze nadkola, które plan oddaje
  modelowi). Model nie cofa zmniejszania jak w tickecie 12 (−2″ → −1,75″), wartości ujemne zostają
  bez korekty. Maska karoserii poprawna na 4 zdjęciach
  dopiero z punktami ujemnymi w szczelinie nadkola i odjętym wnętrzem nadkola (bez tego SAM wciąga
  nadkole do karoserii i koło nie może w nie wejść, `out/v1/`). Kryteria złej maski: pusta, > 5% koła,
  > 85% wycinka poza kołem, < 25% strefy błotnika, a także obce zasłonięcie opony > 5% (Vios koło 1:
  zderzak drugiego auta, 30%; reszta ≤ 1%). mflux 115–134 s na koło (z logów `gen_all`/`gen_rest2`),
  szczyt 27,7 GB (zmierzony w przebiegu Golfa pierwszego podejścia, te same ustawienia); SAM ~1,5 s
  na koło, szczyt 3,6 GB zmierzony w etapie `measure` (`/usr/bin/time -l`). Odstępstwa: Vios i Clio
  bez mflux (decyzja użytkownika, przetestuje je sam; mflux ruszył na Viosie, 3 edycje, przebieg
  przerwany, wyniki odrzucone); średnica koła mierzona z sylwetki (IoU z przeskalowaną maską starego koła), bo k z profilu maski jest w 3/4
  niewiarygodne; ocena wyglądu czeka na użytkownika (pola w `VERDICT.md`). Proponowany podział
  ticketu implementacyjnego w `VERDICT.md`.
- 2026-10-19 (agent, poprawki po code review): `visible_tire_ratio` i `arch_gap_points` odrzucają
  promienie wychodzące poza zdjęcie (marsz promieni wspólny z `rim_geometry.wheel_rays`), stopień
  jako `rim_size: float`, dane zadania 6 z głównego checkoutu, stałe z `EditParams`, kryteria złej
  maski na surowej masce SAM (po czyszczeniu udział koła był zawsze 0), brak dobrego kandydata
  karoserii → `rim_size_unavailable` z powodem zamiast złej maski. Przeliczone plan/measure/
  silhouette/report bez mflux: wejścia modelu bit w bit te same, liczby rozmiaru bez zmian.
  Poprawione wnioski (wyżej i w `VERDICT.md`): zakres zmiany felgi Sorento, rozdzielczość siatki
  sylwetki, co obejmuje sprawdzenie bit w bit, źródła czasów i pamięci (`out/timing.json`), Vios.
- 2026-10-19 (użytkownik): wygląd, werdykt i rekomendacja z `VERDICT.md` zatwierdzone (powiększanie
  całego koła, wartości ujemne bez korekty, proponowany podział ticketu implementacyjnego).

---

# 14: Pole „Obecna felga”

**What to build:** W sekcji rozmiaru felgi na stronie zadania, obok wyboru stopnia `−2″ … +2″`, wybieram obecny rozmiar felgi auta (14″–22″ co 1″, domyślnie 17″). Wartość jest zapisywana w zadaniu i od ticketu 16 wyznacza skalę cala przy zmianie rozmiaru (1″ = średnica felgi w px / (obecna felga + 1,4″), zob. ticket 13). Ten ticket dodaje samo pole, a zachowanie generowania się nie zmienia.

**Blocked by:** 13

**Status:** needs-triage

- [x] Pole „Obecna felga” w sekcji felgi, w formularzu „Generuj” (`#job-form`), jak wybór `rim_size`; domyślnie 17″. `POST /jobs` przy wgraniu go nie przyjmuje
- [x] Nowy klucz w `Job.params` (`wheeler_shared/edit_params.py`, pole w `EditParams`), walidacja 14–22 po stronie serwera jak `rim_size` (`advanced.parse_…` → `InvalidParam` → czytelny komunikat); brak klucza oznacza 17″
- [x] Jedno ustawienie na całe zadanie: „Ponów” i „Ponów to koło” dostają tę samą wartość
- [x] Ekran wyniku pokazuje obecną felgę obok „Felga: +2″”, tylko gdy stopień ≠ „bez zmian”; szczegóły zadania w historii przez `PARAM_LABELS`
- [x] Teksty po polsku w `texts.py`; testy jak w `tests/test_rim_size.py` (walidacja, zapis, retry)

**Comments:**

- 2026-10-19: z rekomendacji w `VERDICT.md` (gałąź `prototype/whole-wheel-resize`, `worker/prototype_whole_wheel/` (`VERDICT.md`, `whole_wheel.py`, `test_whole_wheel.py`, `run.py`)), zatwierdzonej przez użytkownika (ticket 13). Stała `DEFAULT_CURRENT_RIM` = 17 jest dziś w prototypie (`whole_wheel.py`); jej miejsce to `edit_params.py`.
- 2026-10-20 (agent, gałąź `ticket-14`): klucz `current_rim` w `wheeler_shared/edit_params.py`
  (`CURRENT_RIM`, `CURRENT_RIMS` 14…22, `DEFAULT_CURRENT_RIM` = 17, `EditParams.current_rim`).
  Pole to `<select name="current_rim" form="job-form">` pod stopniem rozmiaru w `fieldset`
  `data-role="rim-size"` (`job.html`); walidacja `advanced.parse_current_rim` -> `InvalidParam`
  -> „Obecna felga musi być pełnym rozmiarem od 14″ do 22″.”, 400 ze stroną zadania. Zapis jak
  `rim_size` w `start_generation(..., current_rim=)`: zapisywane przy każdym „Generuj” z polem (także
  przy „bez zmian”), brak pola (albo puste pole, które FastAPI traktuje jak brak) zostawia zapisaną
  wartość; wybór zostaje zaznaczony po odrzuconym „Generuj” (409). „Ponów” i „Ponów to koło” biorą
  wartość z `Job.params` bez dodatkowego kodu. Ekran wyniku: `data-role="result-current-rim"`
  („obecna 17″”) za „Felga: +2″”, tylko przy stopniu ≠ 0; etykieta w `PARAM_LABELS`. Worker
  i generowanie bez zmian (`EditParams.current_rim` czeka na ticket 16). Testy:
  `tests/test_current_rim.py`. DESIGN i README: wzmianka o polu. Odstępstwa: stała rantu 1,4″
  zostaje w prototypie (przeniesie ją ticket 16 razem z `inch_px`); podpowiedź pod stopniem
  rozmiaru („zewnętrzna średnica koła zostaje”) bez zmian, bo zmienia się ona w tickecie 16.
- 2026-10-20 (użytkownik): zmiany cofnięte na `integration` — poprzednia wersja (plus-sizing z ticketu 12, stan `1b9e595`) oceniona jako lepsza. Kod ticketów 14–17 został na gałęziach `ticket-14`…`ticket-17` i `review-fixes-14-17` (merge `c3a4917`). Dalszy kierunek do ustalenia.

---

# 15: Maska karoserii przed zmianą rozmiaru

**What to build:** Przy stopniu ≠ „bez zmian” worker, zaraz po pomiarze felgi i przed załadowaniem mflux, wyznacza dla każdego koła maskę karoserii, która zasłania koło (błotnik, nakładka nadkola). Zła maska albo obce zasłonięcie koła kończy zadanie błędem `rim_size_unavailable`, tym samym co brak felgi. Maska trafia do diagnostyki koła; warp się jeszcze nie zmienia (ticket 16 jej użyje).

**Blocked by:** 13

**Status:** needs-triage

- [x] `Segmenter.segment_body` (publiczna metoda protokołu; adapter SAM 2.1 i atrapy `placeholder`/testowe), prompt jak w prototypie: punkty dodatnie nad oponą (`body_prompt`, `up_angle`), ujemne w środku felgi i na oponie oraz ujemne w szczelinie nadkola (`arch_gap_points`), multimask w dwóch wariantach; maska wnętrza nadkola z SAM
- [x] Kolejność z `VERDICT.md`: (a) kryteria `BodyMaskStats.problem` na **surowych** kandydatach SAM (pusta, > 5% koła, > 85% wycinka poza kołem, < 25% strefy błotnika) i wybór `choose_body`; brak dobrego kandydata → `rim_size_unavailable`, bez brania najlepszego ze złych; (b) potem odjęcie wnętrza nadkola i `clean_body_mask`; (c) `foreign_occlusion` > 5% na masce po czyszczeniu → `rim_size_unavailable`
- [x] Wspólny marsz promieni `rim_geometry.wheel_rays` (refaktor `_tire_ratio` z gałęzi prototypu, commit `2eeffbd`, bez zmiany zachowania): promienie wychodzące poza zdjęcie są odrzucane, nie przycinane
- [x] Liczone razem z felgą w `_measure_rims` (wycinek z oryginału, także przy „Ponów to koło”); SAM zwalniany przed mflux jak dziś; przy „bez zmian” brak pomiaru (czas jak dotąd)
- [x] Diagnostyka w katalogu koła (np. `body.png` i nakładka na `rim_ellipse.png`); bez zmian w bazie i w UI
- [x] Testy na syntetycznych kołach (przeniesione z `test_whole_wheel.py`) i na atrapach przez workera: dobra maska przechodzi, każde kryterium złej maski i obce zasłonięcie dają `rim_size_unavailable`, SAM zwalniany przed pierwszym krokiem mflux

**Comments:**

- 2026-10-19: z rekomendacji w `VERDICT.md` (gałąź `prototype/whole-wheel-resize`, `worker/prototype_whole_wheel/` (`VERDICT.md`, `whole_wheel.py`, `test_whole_wheel.py`, `run.py`)). Na zestawie z prototypu kryteria przepuszczają wszystkie wybrane maski (surowe: strefa błotnika 56–98%, koło ≤ 1,9%, wycinek 3–50%), a obce zasłonięcie zatrzymuje Vios koło 1 (zderzak drugiego auta, 30%). Pierwsze podejście bez punktów w szczelinie nadkola (`out/v1/`) wciągało nadkole do karoserii.
- 2026-10-20 (agent, gałąź `ticket-15`): nowy `wheeler_worker/body_mask.py` (kryteria `BodyMaskStats.problem`,
  `choose_body`, `body_prompt`/`BodyPrompt`, `arch_gap_points`, `choose_liner`, `clean_body_mask`,
  `foreign_occlusion`, `outline_body`; całość w kolejności z `VERDICT.md` w `body_for_resize`, wynik: maska albo
  `BodyMaskProblem`, w tym nowe `FOREIGN_OCCLUSION`). `Segmenter.segment_body(image, wheel_mask, rim)` ->
  `BodySegments(with_gap, without_gap, liner)`: adapter SAM 2.1 liczy prompt z `body_prompt` (jak `segment_rim`
  swój), jeden embedding na trzy prompty; atrapy `placeholder` (pas nad kołem) i testowe (`DiscSegmenter.body`,
  domyślnie `fender_mask`). `rim_geometry.wheel_rays(ellipse, wheel, angles=, steps=)` (refaktor `_tire_ratio`
  bez zmiany zachowania; `arch_gap_points` z niego korzysta i odrzuca promienie wychodzące poza zdjęcie).
  `_measure_rims` zwraca `_MeasuredWheel(rim, body)`; kryteria w wycinku `_wheel_crop` (ten sam co dla modelu).
  Diagnostyka: `body.png` (maska w wycinku) i czerwona poświata karoserii na `rim_ellipse.png`. Powód odrzucenia
  tylko w `ProcessingFailed.detail` (log). Tekst `failure_rim_size_unavailable` obejmuje teraz też karoserię;
  DESIGN (krok 4) i README zaktualizowane. Testy: `tests/test_body_mask.py` (karoseria przeniesiona z `test_whole_wheel.py`; geometria
  nowego koła zostaje dla ticketu 16; + `body_prompt`, `body_for_resize`), `tests/test_rim_size.py` (dobra maska i diagnostyka, każde kryterium
  i obce zasłonięcie -> `rim_size_unavailable` z właściwym powodem, kolejność `rim`, `body`, `release` przed
  mflux, „bez zmian” bez pomiaru, „Ponów to koło” mierzy karoserię na oryginale). Odstępstwa: dwa warianty
  karoserii to „z ujemnymi w szczelinie” i „bez nich” (prototyp miał jeszcze trzeci, same dodatnie — pominięty,
  zgodnie z „dwoma wariantami” w tickecie); test ticketu 12 „warp tylko w masce koła” (felga 45 px, opona
  z typowego k wystaje z maski) dostał atrapę nakładki nadkola wokół koła, bo bez niej teraz słusznie kończy
  się obcym zasłonięciem. Adapter SAM 2 nie uruchomiony na prawdziwym modelu (ticket tego nie wymaga).
- 2026-10-20 (agent, poprawki po code review (gałąź `review-fixes-14-17`)): prompt karoserii liczony raz
  (`Segmenter.segment_body` zwraca go w `BodySegments.prompt`, `body_for_resize` go nie przelicza);
  `_bool`/`_image` i ręczne wycinki `Box` zastąpione wspólnymi `compositing.mask_array`/`mask_image`
  i `Box.slices`, sprawdzenie „punkt w obrazie i w masce” w `_in_mask`; `_measure_rims` →
  `_measure_wheels` (mierzy felgę i karoserię). Bez zmiany zachowania.
- 2026-10-20 (użytkownik): zmiany cofnięte na `integration` — poprzednia wersja (plus-sizing z ticketu 12, stan `1b9e595`) oceniona jako lepsza. Kod ticketów 14–17 został na gałęziach `ticket-14`…`ticket-17` i `review-fixes-14-17` (merge `c3a4917`). Dalszy kierunek do ustalenia.

---

# 16: Zmiana rozmiaru całego koła

**What to build:** Stopień `−2″ … +2″` zmienia całe koło, nie tylko felgę: felga i zewnętrzna średnica koła rosną (albo maleją) o N cali według skali z „Obecnej felgi”, bok opony zostaje tej samej grubości, a najniższy punkt opony (styk z ziemią) stoi w miejscu. Powiększone koło wchodzi w nadkole i chowa się za karoserią, która zostaje bit w bit. Ten warp zastępuje warp felgi z ticketu 12.

**Blocked by:** 14, 15

**Status:** needs-triage

- [x] `resized_geometry` + `plan_resize` z prototypu w workerze zamiast `warp_rim`: skala cala z obecnej felgi (`inch_px`, dłuższa oś elipsy felgi / (obecna + 1,4″)), jednokładność wokół `lowest_point`, odwzorowanie felga/bok opony jak w `VERDICT.md`
- [x] Maska edycji = (stare koło ∪ nowe koło) poszerzone o `mask_grow`, minus karoseria z ticketu 15; wycinek dla modelu wokół maski edycji; kompozycja przez maskę edycji zamiast zapisanej maski koła
- [x] Piksele karoserii i wszystko poza maską edycji bit w bit z oryginałem (test na atrapach); przy „Ponów to koło” wycinek i pomiar z oryginału, składanie na bieżącym wyniku
- [x] Przy „bez zmian” ścieżka bez zmian (maska koła + `mask_grow`, bez pomiaru, bez karoserii)
- [x] Prompt ten sam dla każdego stopnia (`build_prompt`)
- [x] Diagnostyka jak dziś: `rim_ellipse.png` (elipsa felgi, nowa elipsa opony, karoseria) i `warped.png`
- [x] Usunięty warp felgi z ticketu 12 (`warp_rim` i jego testy) tam, gdzie go zastępuje nowy kod
- [x] `docs/DESIGN.md` opisuje rozmiar całego koła zamiast plus-sizingu ze stałą średnicą koła; README zaktualizowane
- [x] Testy na atrapach (geometria, punkt styku, bit w bit, przekazanie wycinka do edytora przez workera, także „Ponów to koło”) oraz jedno uruchomienie na prawdziwym modelu (Golf, +2″, obecna felga 17″) porównane z liczbami z `VERDICT.md` (felga ~19″, sylwetka koła ~+1,75–2″, styk ≤ 1 px)

**Comments:**

- 2026-10-19: z rekomendacji w `VERDICT.md` (gałąź `prototype/whole-wheel-resize`, `worker/prototype_whole_wheel/` (`VERDICT.md`, `whole_wheel.py`, `test_whole_wheel.py`, `run.py`)). Przy wartościach ujemnych ten ticket robi samą geometrię; wstępne wypełnienie odsłoniętego nadkola dochodzi w tickecie 17. Felga w ujęciu 3/4 wychodzi mniejsza już w kontroli „bez zmian” (zachowanie modelu, nie warpu) — poza zakresem.
- 2026-10-20 (agent, gałąź `ticket-16`): nowy `wheeler_worker/wheel_resize.py` (`RIM_FLANGE_INCHES`, `inch_px`,
  `lowest_point`, `resized_geometry`, `plan_resize` -> `ResizePlan(geometry, wheel, edit_mask, model_input)`;
  odwzorowanie z prototypu, liczone w oknie wokół starego koła i jego obrazu w jednokładności, nie na całym
  zdjęciu). `_WheelEdit.apply`: przy stopniu ≠ 0 plan na oryginale (skala z `EditParams.current_rim`), wycinek
  `_wheel_crop` wokół maski edycji, maska modelu i kompozycja = maska edycji; poza maską modelu wycinek to
  dokładnie oryginał (`Image.composite`, jak w tickecie 12). `_MeasuredWheel` niesie maskę koła bez
  poszerzenia. `warp_rim`, `RIM_FRACTION_PER_INCH` i testy warpu usunięte; `outline_rim` dostał kolory
  (stara felga i opona białe, nowe zielona i żółta). `crop.png`/`crop_mask.png` przy zmianie rozmiaru to
  wycinek oryginału i maska edycji wokół maski edycji. Testy: `tests/test_wheel_resize.py` (skala cala,
  geometria, styk także w 3/4, plan: nowe elipsy za karoserią, maska edycji, bit w bit, koło przy brzegu
  zdjęcia), `tests/test_rim_size.py` (procesor i HTTP z atrapami: wynik mierzony względem „bez zmian” —
  felga, środek, szerokość, styk; skala z obecnej felgi 14″/22″; karoseria i wszystko poza starym ∪ nowym
  kołem bit w bit; „Ponów to koło” z oryginału, ten sam wycinek, składanie na bieżącym wyniku), smoke test
  atrapy `placeholder` przy +2″. DESIGN (sekcja „Rozmiar koła”), README, podpowiedź pod stopniem rozmiaru
  i komentarze w `edit_params.py` zaktualizowane.
  Prawdziwy model (SAM 2.1 large + mflux q8, 1024 px, `car_side_golf.jpg`, 2 koła, +2″, obecna 17″, seed
  863048696, felga z zadania 6; produkcyjny `WheelSwapProcessor.detect` + `generate`, pomiar jak `run.py
  measure`/`silhouette` prototypu): felga 19,19″ i 19,03″ (prototyp 19,01″ / 18,85″, cel 19″), sylwetka koła
  +1,75″ na obu kołach (prototyp +1,75″), styk −1 px na obu (prototyp 0 / −1 px); oryginał tą metodą 17,00″,
  sylwetka 0. Bit w bit: 0 zmienionych pikseli poza sumą masek edycji i 0 w karoserii każdego koła w jego
  wycinku (41 i 20 tys. px). Czas: detekcja 17 s, generowanie 234 s dla 2 kół (pierwszy krok koła 1 po 52 s
  z ładowaniem), szczyt `peak memory footprint` 28,9 GB (MLX 25,1 GB), pomiar SAM 3,6 GB. Wygląd jak
  w prototypie (opona chowa się za błotnikiem, styk stoi). Odstępstwa: skrypt `gpu-run.sh` z poprzednich
  ticketów nie istniał (zaginął ze scratchpadu), więc odtworzony jako prosta blokada GPU +
  `/usr/bin/time -l`; uruchomienie przez procesor, nie przez HTTP i kolejkę; przy wartościach ujemnych
  odsłonięte nadkole ma w wycinku piksele starej opony (wypełnienie: ticket 17), więc testy szerokości koła
  tylko dla wartości dodatnich.
- 2026-10-20 (agent, poprawki po code review (gałąź `review-fixes-14-17`)): okno warpu ma zapas
  `LANCZOS_REACH` + 1 px (Lanczos czyta 4 px sąsiedztwa; przy `mask_grow` = 0 zapas 2 px dawał przy
  brzegu okna odbicie zamiast zdjęcia), a współrzędne źródła są zaokrąglane do siatki `cv2.remap`
  (1/32 px) jeszcze w float64, bo float32 w układzie okna i całego zdjęcia zaokrąglał je inaczej
  (pojedyncze piksele różniły się o kilka poziomów). Test: plan w oknie = plan na całym zdjęciu przy
  `mask_grow` = 0 (tło z pełnym szumem; przy lekkim szumie odbicie nie różni się od zdjęcia). Refaktory
  bez zmiany zachowania: `MeasuredWheel` przeniesiony do `wheel_resize` i podawany `plan_resize` jednym
  argumentem (`ResizePlan.old` niesie go do diagnostyki, więc `_WheelEdit.apply` sprawdza tylko
  `plan`), zapis diagnostyki w `_save_resize_diagnostics`, `mask_box` liczony raz (`_wheel_crop` bierze
  `Box`), bez `ResizePlan.wheel` (testy biorą nowe koło z elipsy nowej opony z pikselem zapasu albo
  z pikseli koła w `model_input`), `rim_size: int` w `plan_resize` i `resized_geometry` (ułamek 0,25″
  był tylko w prototypie, do siatki sylwetki w `run.py`; testy workera go nie używają), stałe kolorów
  na górze `rim_geometry`.
- 2026-10-20 (użytkownik): zmiany cofnięte na `integration` — poprzednia wersja (plus-sizing z ticketu 12, stan `1b9e595`) oceniona jako lepsza. Kod ticketów 14–17 został na gałęziach `ticket-14`…`ticket-17` i `review-fixes-14-17` (merge `c3a4917`). Dalszy kierunek do ustalenia.

---

# 17: Zmniejszanie koła — wypełnienie odsłoniętego nadkola

**What to build:** Przy „−1″” i „−2″” odsłonięty obszar (stare koło minus nowe) jest w wycinku dla modelu wstępnie wypełniony przyciemnionym kolorem z okolicy nadkola z rozmytą krawędzią, a model domalowuje wnętrze nadkola w obrębie maski edycji. Wartości ujemne zostają bez korekty niedostrzału (prototyp: −2″ → −1,75″).

**Blocked by:** 16

**Status:** needs-triage

- [x] Wypełnienie jak `_prefill` w prototypie: mediana koloru pasa nad górną połową starego koła × 0,5, krawędź rozmyta (σ = 2% boku koła)
- [x] Test: odsłonięty obszar leży w masce edycji i nie zawiera pikseli starej opony; poza maską edycji bit w bit
- [x] Bez korekty niedostrzału dla wartości ujemnych (decyzja z ticketu 13)

**Comments:**

- 2026-10-19: z rekomendacji w `VERDICT.md` (gałąź `prototype/whole-wheel-resize`, `worker/prototype_whole_wheel/` (`VERDICT.md`, `whole_wheel.py`, `test_whole_wheel.py`, `run.py`)). Wygląd wnętrza nadkola przy −1″/−2″ zatwierdzony przez użytkownika na arkuszach prototypu.
- 2026-10-20 (agent, gałąź `ticket-17`): `wheel_resize.plan_resize` przy zmniejszaniu wypełnia
  odsłonięte nadkole (stare koło minus nowe, poza karoserią) w `model_input` przez `_prefill`
  z prototypu: mediana koloru oryginału w pasie 6% boku koła wokół górnej połowy starego koła (bez
  karoserii; gdy pusty, cała okolica koła) × 0,5, gauss σ = 2% boku koła tylko na pikselach
  odsłoniętych (stałe `PREFILL_BAND`, `PREFILL_DARKEN`, `PREFILL_BLUR`). Liczone w oknie wokół starego
  koła, nie na całym zdjęciu. Maska edycji bez zmian (obejmowała już stare koło), poza nią bit w bit;
  `warped.png` pokazuje wypełnienie. Bez korekty niedostrzału. Testy: `tests/test_wheel_resize.py`
  (odsłonięty obszar w masce edycji, jasność ok. 0,5 tła, > 95% pikseli jaśniejszych od starej opony,
  poza maską edycji bit w bit; przy +2″ nic nie jest wypełniane), `tests/test_rim_size.py`: szerokość
  koła sprawdzana teraz także dla −1″/−2″ (także obecna felga 14″/22″ i ponowione zadanie z −2″);
  pomiar `_measure` liczy brzeg koła jako ostatni piksel ciemny jak opona w szarym obszarze, bo
  wypełnienie na zaszumionym tle atrapy też jest szare (~64). DESIGN (krok 5 „Rozmiaru koła”)
  zaktualizowany. Odstępstwa: „pas nad górną połową” to jak w `_prefill` prototypu pierścień wokół starego
  koła powyżej środka opony; kolor z oryginału zamiast z obrazu po wklejeniu nowego koła (to samo,
  bo pierścień leży poza starym kołem); prawdziwy model nieuruchamiany (ticket tego nie wymaga).
- 2026-10-20 (agent, poprawki po code review (gałąź `review-fixes-14-17`)): test wypełnienia wymaga
  zera pikseli starej opony w odsłoniętym obszarze (było: ≤ 5% ciemnych). Bez wyłączania rozmytego
  brzegu: najciemniejszy piksel wypełnienia ~46 (rozmycie przy nowej oponie), opona 10. Odsłonięty
  obszar w teście to stare koło poza elipsą nowej opony powiększoną o 1% (~1 px), bo nowe koło to
  stara maska po warpie (najbliższy sąsiad) i jego brzeg może wyjść o piksel poza elipsę.
- 2026-10-20 (użytkownik): zmiany cofnięte na `integration` — poprzednia wersja (plus-sizing z ticketu 12, stan `1b9e595`) oceniona jako lepsza. Kod ticketów 14–17 został na gałęziach `ticket-14`…`ticket-17` i `review-fixes-14-17` (merge `c3a4917`). Dalszy kierunek do ustalenia.
