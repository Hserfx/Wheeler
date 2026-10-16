"""Kolejka zadań w PostgreSQL (prawdziwy Postgres w Podmanie)."""

from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.models import Job, TaskKind
from wheeler_shared.queue import claim_next_job, create_job, enqueue


def _job(session_factory: sessionmaker[Session], name: str, params: dict[str, int]) -> int:
    with session_factory.begin() as session:
        return create_job(
            session,
            input_path=f"jobs/{name}/input.jpg",
            rim_path=f"jobs/{name}/rim.png",
            params=params,
        )


def _enqueue(session_factory: sessionmaker[Session], job_id: int, task: TaskKind) -> None:
    with session_factory.begin() as session:
        enqueue(session.get_one(Job, job_id), task)


def _claim(session_factory: sessionmaker[Session]) -> tuple[int, TaskKind] | None:
    with session_factory.begin() as session:
        claimed = claim_next_job(session)
    return None if claimed is None else (claimed.id, claimed.task)


def test_claim_returns_none_when_queue_is_empty(session_factory: sessionmaker[Session]) -> None:
    assert _claim(session_factory) is None


def test_draft_jobs_are_not_claimed(session_factory: sessionmaker[Session]) -> None:
    _job(session_factory, "a", {})
    assert _claim(session_factory) is None


def test_jobs_are_claimed_once_in_order_they_were_queued_not_created(
    session_factory: sessionmaker[Session],
) -> None:
    older = _job(session_factory, "a", {"seed": 1})
    newer = _job(session_factory, "b", {})
    _enqueue(session_factory, newer, TaskKind.GENERATE)
    _enqueue(session_factory, older, TaskKind.GENERATE)

    with session_factory.begin() as session:
        claimed = claim_next_job(session)
        assert claimed is not None
        assert (claimed.id, claimed.task) == (newer, TaskKind.GENERATE)
        assert claimed.input_path == "jobs/b/input.jpg"
        assert claimed.rim_path == "jobs/b/rim.png"
    assert _claim(session_factory) == (older, TaskKind.GENERATE)
    assert _claim(session_factory) is None


def test_mask_requests_jump_ahead_of_queued_generation(
    session_factory: sessionmaker[Session],
) -> None:
    generating = _job(session_factory, "a", {})
    clicked = _job(session_factory, "b", {})
    _enqueue(session_factory, generating, TaskKind.GENERATE)
    _enqueue(session_factory, clicked, TaskKind.SEGMENT)

    assert _claim(session_factory) == (clicked, TaskKind.SEGMENT)
    assert _claim(session_factory) == (generating, TaskKind.GENERATE)


def test_job_locked_by_another_transaction_is_skipped(
    session_factory: sessionmaker[Session],
) -> None:
    first = _job(session_factory, "a", {})
    second = _job(session_factory, "b", {})
    _enqueue(session_factory, first, TaskKind.GENERATE)
    _enqueue(session_factory, second, TaskKind.GENERATE)

    # Pierwszy worker ma otwartą transakcję i trzyma blokadę na pierwszym zadaniu.
    with session_factory.begin() as worker_a:
        claimed_a = claim_next_job(worker_a)
        with session_factory.begin() as worker_b:
            claimed_b = claim_next_job(worker_b)

    assert claimed_a is not None and claimed_a.id == first
    assert claimed_b is not None and claimed_b.id == second
