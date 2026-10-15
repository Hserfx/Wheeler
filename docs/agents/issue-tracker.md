# Issue tracker: docs/tickets.md

Spec i tickety żyją jako markdown w `docs/`, bez zewnętrznego trackera i bez PR-ów (repo nie ma remote'a).

## Conventions

- Spec: `docs/<NAZWA>.md` (obecnie `docs/DESIGN.md`).
- Wszystkie tickety w jednym pliku `docs/tickets.md`, każdy jako sekcja `# NN: Tytuł`, rozdzielone `---`. Numeracja od `01`.
- Każdy ticket ma linie `**Blocked by:**` (numery ticketów albo `None`) i `**Status:**`.
- Wartości `Status`: role triage z `triage-labels.md`, plus `claimed` (ktoś nad nim pracuje) i `done` (zamknięty).
- Kryteria akceptacji to checkboxy `- [ ]` pod ticketem.
- Komentarze i historia dopisywane na końcu sekcji ticketu pod `**Comments:**`.

## When a skill says "publish to the issue tracker"

Dopisz nową sekcję `# NN: Tytuł` na końcu `docs/tickets.md` z kolejnym numerem i zaktualizuj diagram zależności na górze pliku.

## When a skill says "fetch the relevant ticket"

Przeczytaj sekcję o danym numerze w `docs/tickets.md`.

## Claiming and closing work

- **Claim:** ustaw `**Status:** claimed` przed rozpoczęciem pracy.
- **Close:** odhacz spełnione checkboxy (`- [x]`) i ustaw `**Status:** done` w tym samym commicie co ostatnia zmiana ticketu. Praca jest zamknięta po merge'u na gałąź integracyjną / `main`; PR-ów nie ma.
- **Frontier:** tickety ze statusem `ready-for-agent`, których wszystkie `Blocked by` mają `Status: done`.
