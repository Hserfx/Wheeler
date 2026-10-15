# Wheeler — tickety

Tickety to pionowe przekroje (tracer bullets) wynikające z [DESIGN.md](DESIGN.md), ułożone w kolejności zależności. Ticket można zacząć, gdy wszystkie blokujące go tickety są zrobione.

```
01 ─┐
    ├─► 03 ─┬─► 04 ─► 07
02 ─┘       ├─► 05, 06, 09, 10, 11
02 ─────────┴─► 08
```

---

# 01: Spike — pomiar modelu i detektora

**What to build:** Krótki, jednorazowy eksperyment, który na prawdziwym M2 Max ustala, jakich ustawień domyślnych użyje aplikacja. Pobieramy przykładowe zdjęcia z darmowych źródeł (Wikimedia Commons / Unsplash): 2–3 auta w różnych ujęciach (bok, 3/4 przodu, koło częściowo zasłonięte) i 2–3 felgi. Mierzymy na nich czas, szczytowe zużycie pamięci i jakość podmiany felgi na wycinku wokół koła.

**Blocked by:** None (can start immediately)

**Status:** claimed

- [ ] Przykładowe zdjęcia są w repo wraz ze źródłem i licencją każdego z nich
- [ ] Wagi Qwen-Image-2.1-Turbo są skonwertowane do mflux q8 i q4 z lokalnego cache bf16
- [ ] Zmierzone czasy i szczytowa pamięć dla kombinacji: q8 / q4 × 1024 / 1536 px, z rozbiciem na ładowanie modelu i generowanie
- [ ] Ta sama edycja wykonana bazowym pipeline'em diffusers bf16 (dwuetapowe ładowanie z `image-gen-model-test`) jako punkt odniesienia
- [ ] Porównanie Florence-2 i Grounding DINO pod kątem trafności i czasu wykrywania kół na zdjęciach przykładowych
- [ ] Sprawdzone, czy siatka dokładnie 1024 px daje szum (porównanie z 992/1056)
- [ ] Wyniki i obrazy porównawcze w raporcie, a wybrane ustawienia domyślne (kwantyzacja, rozdzielczość robocza, detektor, szablon promptu) dopisane do DESIGN.md

---

# 02: Szkielet end-to-end (bez AI)

**What to build:** Najcieńsza działająca ścieżka przez wszystkie warstwy. `just up` uruchamia maszynę Podmana `wheels`, kontenery PostgreSQL i FastAPI oraz natywnego workera w tle. Pod `127.0.0.1:8000` jest polska strona (Jinja2 + HTMX/Alpine, Tailwind ze standalone CLI), na której wgrywam zdjęcie. Powstaje wtedy zadanie w kolejce w Postgresie, worker je odbiera, wykonuje atrapę przekształcenia, zapisuje wynik na wolumenie i oznacza zadanie jako gotowe, a strona pokazuje wynik. `just down` zatrzymuje wszystko.

**Blocked by:** None (can start immediately; może iść równolegle z 01)

**Status:** claimed

- [ ] Monorepo `api/`, `worker/`, `shared/` jako workspace uv z jednym lockfile'em, `git init`
- [ ] Maszyna Podmana `wheels` (2 CPU, 2 GB) tworzona automatycznie, jeśli nie istnieje
- [ ] Port API opublikowany wyłącznie na `127.0.0.1`, niedostępny z sieci lokalnej
- [ ] Schemat bazy zarządzany przez Alembic, migracje uruchamiane przy starcie
- [ ] Worker pobiera zadania przez `FOR UPDATE SKIP LOCKED` i przetwarza jedno naraz
- [ ] Pliki leżą na wolumenie dyskowym wspólnym dla API i workera, a w bazie są tylko ścieżki
- [ ] Wszystkie teksty UI po polsku, w jednym pliku
- [ ] `just up`, `just down`, `just worker` i `just test` działają, a ruff i mypy przechodzą
- [ ] Test integracyjny: wgranie zdjęcia → zadanie → wynik, na prawdziwym Postgresie w Podmanie

---

# 03: Podmiana jednego koła

**What to build:** Pierwsza prawdziwa podmiana felgi. Wgrywam zdjęcie auta i zdjęcie felgi, a potem klikam w jedno koło na zdjęciu auta. SAM 2 wyznacza maskę całego koła (opona + felga, lekko poszerzoną). Wycinek wokół koła trafia do mflux Turbo razem z felgą referencyjną, z poleceniem podmiany felgi przy zachowaniu opony. Wynik jest skalowany z powrotem i wklejany w oryginał z wygładzoną krawędzią maski. Dostaję PNG w rozdzielczości oryginału, w którym zmieniło się tylko to koło.

**Blocked by:** 01, 02

**Status:** ready-for-agent

- [ ] Kliknięcie w koło tworzy maskę i pokazuje ją jako nakładkę przed generowaniem
- [ ] Wywołanie modelu przyjmuje obraz felgi i wycinek koła, a ustawienia domyślne pochodzą z wyników spike'a
- [ ] Wynik ma dokładnie tę samą rozdzielczość co zdjęcie wejściowe
- [ ] Test automatyczny: każdy piksel poza maską jest identyczny bit w bit z oryginałem
- [ ] Wynik zapisywany zawsze jako PNG
- [ ] Testy API używają mocka modelu AI, a logika kompozycji jest testowana bez modelu

---

# 04: Automatyczna detekcja i edycja masek, wiele kół

**What to build:** Po wgraniu zdjęcia auta aplikacja sama znajduje wszystkie widoczne koła i pokazuje ich maski jako nakładkę. Klik w maskę ją usuwa, a klik w puste miejsce dodaje nowe koło (SAM 2 z promptem punktowym). Po zatwierdzeniu wszystkie wybrane koła są podmieniane, każde osobnym wywołaniem, z tym samym seedem i promptem.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Detekcja (detektor wybrany w spike'u) + SAM 2 kończy się w ciągu kilku sekund
- [ ] Gdy nie wykryto żadnego koła, UI jasno o tym informuje i zachęca do dodania koła kliknięciem
- [ ] Usuwanie i dodawanie masek działa przed uruchomieniem generowania
- [ ] Wszystkie wybrane koła trafiają do jednego wyniku, a piksele poza wszystkimi maskami pozostają nienaruszone
- [ ] Wszystkie koła jednego zadania używają tego samego seeda i promptu

---

# 05: Postęp, anulowanie i obsługa awarii

**What to build:** W trakcie generowania widzę, które koło jest przetwarzane i na którym kroku denoisingu jest model. Mogę anulować zadanie. Jeśli worker padnie albo zostanie zrestartowany w trakcie zadania, zadanie dostaje status `failed`, a w UI pojawia się przycisk „Ponów”.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Postęp per koło i per krok odświeżany pollingiem HTMX co ~1 s
- [ ] „Anuluj” zatrzymuje zadanie najpóźniej po bieżącym kroku, a zadanie dostaje status „anulowane”
- [ ] Zadanie przerwane przez awarię lub restart workera ma status `failed`, bez automatycznego ponawiania
- [ ] „Ponów” tworzy nowe zadanie z tymi samymi danymi wejściowymi i parametrami
- [ ] Kolejne zadania czekają w kolejce FIFO, a UI pokazuje pozycję w kolejce

---

# 06: Biblioteka felg

**What to build:** Raz wgraną felgę mogę wybierać przy kolejnych zdjęciach bez ponownego wgrywania. Przy dodawaniu nowej felgi tło jest usuwane automatycznie, a ja widzę podgląd wyciętej felgi i mogę go zaakceptować albo odrzucić. Felgi mogą mieć opcjonalną nazwę i są sortowane po ostatnim użyciu.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Usuwanie tła (BiRefNet lub rembg) z podglądem do akceptacji przed zapisaniem do biblioteki
- [ ] Przechowywane dane: wycięta felga, oryginał, opcjonalna nazwa, data dodania, licznik użyć, data ostatniego użycia
- [ ] Wybór felgi z biblioteki na stronie głównej jako alternatywa dla wgrania nowej
- [ ] Do modelu trafia wycięta felga, nie oryginał z tłem
- [ ] Felgę można usunąć z biblioteki, a usunięcie nie psuje historii zadań, które jej używały

---

# 07: Ekran wyniku — przed/po i „ponów to koło”

**What to build:** Gotowy wynik oglądam na suwaku przed/po. Jeśli jedno koło wyszło gorzej, mogę wygenerować ponownie tylko to koło z innym seedem, a pozostałe koła zostają bez zmian. Wynik pobieram jako PNG.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Suwak przed/po działa na pełnym obrazie, z możliwością powiększenia okolicy koła
- [ ] „Ponów to koło” generuje tylko wybrane koło z nowym seedem i podmienia je w wyniku
- [ ] Piksele poza maską ponawianego koła, łącznie z pozostałymi kołami, pozostają bez zmian
- [ ] Pobieranie zwraca PNG w rozdzielczości oryginału

---

# 08: Historia

**What to build:** Osobna podstrona „Historia” pokazuje wszystkie zadania jako miniatury ze statusem i datą. Mogę otworzyć dowolne zadanie, żeby zobaczyć wejście, wynik i parametry, oraz usuwać zadania pojedynczo lub zbiorczo, razem z ich plikami na dysku.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Lista zadań od najnowszych, z miniaturą, statusem i datą
- [ ] Widok szczegółów zadania z wejściem, wynikiem i użytymi parametrami
- [ ] Usuwanie pojedyncze i zbiorcze (zaznaczanie), z potwierdzeniem
- [ ] Usunięcie kasuje pliki zadania z wolumenu, a test to weryfikuje
- [ ] Brak automatycznego sprzątania, historia jest trzymana bezterminowo

---

# 09: Sekcja „Zaawansowane”

**What to build:** Na stronie głównej jest zwijana sekcja „Zaawansowane”, domyślnie zwinięta, w której mogę ustawić seed, rozdzielczość roboczą, poszerzenie maski i dopisek do promptu. Bez jej otwierania wszystko działa na rozsądnych ustawieniach domyślnych. Użyte wartości są zapisywane w zadaniu.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Sekcja domyślnie zwinięta, z ustawieniami domyślnymi z wyników spike'a
- [ ] Puste pole seed oznacza losowy seed, a wylosowana wartość jest zapisywana w zadaniu
- [ ] Walidacja zakresów (rozdzielczość jako wielokrotność 32, rozsądne granice poszerzenia maski) z czytelnym komunikatem
- [ ] Użyte wartości widoczne w szczegółach zadania w historii

---

# 10: Obsługa wejścia i metadanych

**What to build:** Mogę wgrać zdjęcie w typowym formacie, łącznie z HEIC z iPhone'a, przez przeciągnięcie albo wklejenie ze schowka. Zdjęcie jest poprawnie obrócone według EXIF. Za duże pliki są odrzucane z czytelnym komunikatem. Wynik zachowuje profil kolorów oryginału i jego EXIF, ale bez lokalizacji GPS.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Akceptowane formaty: JPEG, PNG, WebP i HEIC, a inne są odrzucane z komunikatem
- [ ] Obrót według orientacji EXIF stosowany przed detekcją i kompozycją
- [ ] Limity 40 MPx / 50 MB sprawdzane z czytelnym komunikatem
- [ ] Wgrywanie przez drag & drop, wybór pliku i wklejenie ze schowka
- [ ] Wynikowy PNG zawiera profil ICC oryginału (np. Display P3) i EXIF bez danych GPS
- [ ] Test: zdjęcie HEIC z orientacją i GPS daje poprawnie obrócony PNG z ICC i bez GPS

---

# 11: Cykl życia modelu w pamięci

**What to build:** Worker nie zajmuje pamięci, dopóki nie jest potrzebny. Model ładuje się przy pierwszym zadaniu, zostaje w pamięci na kolejne zadania i zwalnia się po 10 minutach bezczynności (czas konfigurowalny). W UI widzę stan modelu: „uśpiony”, „ładowanie” albo „gotowy”.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Po starcie workera model nie jest załadowany
- [ ] Pierwsze zadanie ładuje model, a UI pokazuje stan „ładowanie” zamiast zablokowanego postępu
- [ ] Po skonfigurowanym czasie bezczynności model jest zwalniany, a pamięć faktycznie wraca do systemu
- [ ] Czas bezczynności ustawiany przez konfigurację, domyślnie 10 min
- [ ] Stan modelu (uśpiony / ładowanie / gotowy) widoczny w UI
