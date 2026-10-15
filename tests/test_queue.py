"""Kolejka zadań w PostgreSQL (prawdziwy Postgres w Podmanie)."""

from sqlalchemy.orm import Session, sessionmaker

from wheeler_shared.queue import claim_next_job, enqueue_job


def test_claim_returns_none_when_queue_is_empty(session_factory: sessionmaker[Session]) -> None:
    with session_factory.begin() as session:
        assert claim_next_job(session) is None


def test_jobs_are_claimed_in_fifo_order_and_only_once(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory.begin() as session:
        first = enqueue_job(session, input_path="jobs/a/input.jpg", params={"seed": 1})
        second = enqueue_job(session, input_path="jobs/b/input.jpg", params={})

    with session_factory.begin() as session:
        claimed = claim_next_job(session)
        assert claimed is not None
        assert claimed.id == first
        assert claimed.input_path == "jobs/a/input.jpg"
        assert claimed.params == {"seed": 1}

    with session_factory.begin() as session:
        claimed = claim_next_job(session)
        assert claimed is not None
        assert claimed.id == second

    with session_factory.begin() as session:
        assert claim_next_job(session) is None


def test_job_locked_by_another_transaction_is_skipped(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory.begin() as session:
        first = enqueue_job(session, input_path="jobs/a/input.jpg", params={})
        second = enqueue_job(session, input_path="jobs/b/input.jpg", params={})

    # Pierwszy worker ma otwartą transakcję i trzyma blokadę na pierwszym zadaniu.
    with session_factory.begin() as worker_a:
        claimed_a = claim_next_job(worker_a)
        with session_factory.begin() as worker_b:
            claimed_b = claim_next_job(worker_b)

    assert claimed_a is not None and claimed_a.id == first
    assert claimed_b is not None and claimed_b.id == second
