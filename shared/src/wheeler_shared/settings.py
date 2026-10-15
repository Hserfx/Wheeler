"""Konfiguracja wspólna dla API i workera (zmienne środowiskowe z prefiksem `WHEELER_`)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Domyślnie: Postgres z `compose.yaml`, opublikowany na hoście dla natywnego workera.
DEFAULT_DATABASE_URL = "postgresql+psycopg://wheeler:wheeler@127.0.0.1:55432/wheeler"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WHEELER_")

    database_url: str = DEFAULT_DATABASE_URL
    # Wolumen danych: `./data` na hoście, `/data` w kontenerze API.
    data_dir: Path = Path("data")
