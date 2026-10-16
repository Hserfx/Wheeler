"""`wheeler-worker` / `just worker`: natywny worker przetwarzający kolejkę zadań."""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType

from wheeler_shared.db import make_engine, make_session_factory
from wheeler_shared.migrate import migrate
from wheeler_shared.storage import Storage
from wheeler_worker.processing import build_processor
from wheeler_worker.runner import recover_interrupted_jobs, run_forever
from wheeler_worker.settings import WorkerSettings

log = logging.getLogger("wheeler_worker")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = WorkerSettings()
    processor = build_processor(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.data_dir)

    migrate(settings.database_url)
    engine = make_engine(settings.database_url)
    stop = threading.Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("Sygnał %s: kończę po bieżącym zadaniu", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    session_factory = make_session_factory(engine)
    recover_interrupted_jobs(session_factory)
    log.info("Worker gotowy (procesor=%s, dane=%s)", settings.processor, storage.root)
    try:
        run_forever(
            session_factory,
            storage,
            processor,
            poll_interval=settings.poll_interval,
            stop=stop,
        )
    finally:
        engine.dispose()
    log.info("Worker zatrzymany")


if __name__ == "__main__":
    main()
