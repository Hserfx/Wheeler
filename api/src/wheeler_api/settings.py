from wheeler_shared.settings import Settings


class ApiSettings(Settings):
    # Migracje Alembica przy starcie aplikacji (wymaganie: schemat aktualny po `just up`).
    run_migrations: bool = True
