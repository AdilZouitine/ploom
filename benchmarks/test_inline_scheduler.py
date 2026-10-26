"""Measure the in-process scheduler and modeled primitives without worker processes."""

import pytest

import ploom
from ploom.execution import Execution, Passed
from ploom.exploration import explore
from ploom.thread_scheduler import ThreadScheduler
from ploom.trace import Config

CONFIG = Config(max_executions=100_000, timeout=3600.0)
WITHDRAWAL = 100


class InlineRunner:
    """Run each schedule in a fresh in-process scheduler instead of a subprocess."""

    def run(
        self,
        required_prefix: tuple[int, ...],
        _timeout_seconds: float,
        expected_execution: Execution | None = None,
        /,
    ) -> Execution:
        """Execute one schedule of ``locked_withdrawals``."""
        return ThreadScheduler(CONFIG.max_steps, CONFIG.max_tasks).execute(
            locked_withdrawals, required_prefix, expected_execution
        )


def locked_withdrawals() -> None:
    """Two threads withdraw from one account under a lock; never overdrawn."""
    balance = ploom.SharedCell(WITHDRAWAL)
    lock = ploom.Lock()

    def withdraw() -> None:
        with lock:
            if balance.read() >= WITHDRAWAL:
                balance.write(balance.read() - WITHDRAWAL)

    first = ploom.Thread(target=withdraw)
    second = ploom.Thread(target=withdraw)
    first.start()
    second.start()
    first.join()
    second.join()
    assert balance.read() >= 0


def yielding_tasks() -> None:
    """Run three tasks that each hit 20 scheduling points."""

    def worker() -> None:
        for _ in range(20):
            ploom.yield_now()

    workers = [ploom.Thread(target=worker) for _ in range(3)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()


@pytest.mark.benchmark
def test_single_execution() -> None:
    """Measure one default schedule with 60 modeled thread hand-offs."""
    execution = ThreadScheduler(CONFIG.max_steps, CONFIG.max_tasks).execute(
        yielding_tasks, (), None
    )
    assert execution.outcome == Passed()


@pytest.mark.benchmark
def test_locked_withdrawals_exploration() -> None:
    """Measure exhaustive in-process exploration of the README example with a lock."""
    result = explore(InlineRunner(), CONFIG)
    assert result.execution.outcome == Passed()
    assert result.statistics.executions > 1
