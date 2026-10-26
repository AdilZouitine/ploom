import threading
from dataclasses import replace
from pathlib import Path
from time import monotonic

import pytest
from hypothesis import given
from hypothesis import strategies as st

import ploom
from ploom.execution import Failed
from ploom.primitives import validate_immutable
from ploom.thread_scheduler import ThreadScheduler


def test_lost_update_and_replay(tmp_path: Path) -> None:
    def scenario() -> None:
        counter = ploom.SharedCell(0)

        def increment() -> None:
            counter.write(counter.read() + 1)

        first = ploom.Thread(target=increment)
        second = ploom.Thread(target=increment)
        first.start()
        second.start()
        first.join()
        second.join()
        expected = 2
        assert counter.read() == expected

    with pytest.raises(ploom.ModelCheckError) as caught:
        ploom.check(scenario, trace_dir=tmp_path, config=ploom.Config(minimize_timeout=0.3))
    error = caught.value
    assert error.trace_path.exists()
    assert error.minimized_path.exists()
    assert error.trace.outcome == Failed(error.failure)
    assert ploom.replay(scenario, error.trace_path) == error.trace
    assert ploom.replay(scenario, error.minimized_path) == error.minimized_trace
    assert error.reduction.status in ("completed", "budget_exhausted")

    changed_step = replace(error.trace.steps[0], operation="changed")
    changed = error.trace.model_copy(update={"steps": (changed_step, *error.trace.steps[1:])})
    with pytest.raises(ploom.ReplayDivergenceError, match="operation"):
        ploom.replay(scenario, changed)


def test_protected_increment_exhaustive() -> None:
    def scenario() -> None:
        counter = ploom.SharedCell(0)
        lock = ploom.Lock()

        def increment() -> None:
            with lock:
                counter.write(counter.read() + 1)

        task = ploom.Thread(target=increment)
        task.start()
        increment()
        task.join()
        expected = 2
        assert counter.read() == expected

    result = ploom.check(scenario)
    assert result.executions > 1
    assert result.steps > result.executions


def test_deadlock(tmp_path: Path) -> None:
    def scenario() -> None:
        lock = ploom.Lock()
        lock.acquire()
        lock.acquire()

    with pytest.raises(ploom.DeadlockError) as caught:
        ploom.check(scenario, trace_dir=tmp_path)
    assert caught.value.reduction.status == "completed"
    assert ploom.replay(scenario, caught.value.trace_path).outcome == caught.value.trace.outcome


@pytest.mark.parametrize(
    ("configuration", "limit"),
    [
        (ploom.Config(max_steps=1), "steps"),
        (ploom.Config(max_tasks=1), "tasks"),
        (ploom.Config(max_executions=1), "schedules"),
    ],
)
def test_limits(configuration: ploom.Config, limit: str) -> None:
    def scenario() -> None:
        thread = ploom.Thread(target=ploom.yield_now)
        thread.start()
        ploom.yield_now()
        thread.join()

    with pytest.raises(ploom.IncompleteExplorationError) as caught:
        ploom.check(scenario, config=configuration)
    assert caught.value.limit == limit
    assert caught.value.statistics.executions >= 1


def test_hard_timeout_and_recovery(tmp_path: Path) -> None:
    entered = tmp_path / "entered-native-wait"

    def stuck() -> None:
        entered.touch()
        threading.Event().wait()

    started = monotonic()
    with pytest.raises(ploom.IncompleteExplorationError) as caught:
        ploom.check(stuck, config=ploom.Config(timeout=3.0))
    assert caught.value.limit == "wall_time"
    assert entered.exists()
    timeout_margin = 10
    assert monotonic() - started < timeout_margin

    def healthy() -> None:
        ploom.yield_now()

    assert ploom.check(healthy).executions == 1


def test_thread_and_lock_contract() -> None:
    def scenario() -> None:
        lock = ploom.Lock()
        assert not lock.locked()
        assert lock.acquire(blocking=False)
        assert lock.locked()
        assert not lock.acquire(blocking=False)
        lock.release()
        seen = ploom.SharedCell(0)

        def target(value: int, *, offset: int) -> None:
            seen.write(value + offset)

        task = ploom.Thread(target=target, args=(3,), kwargs={"offset": 4})
        assert not task.is_alive()
        task.start()
        task.join()
        assert not task.is_alive()
        task.join()
        expected = 7
        assert seen.read() == expected

    assert ploom.check(scenario).executions > 0


@pytest.mark.parametrize("value", [[], {}, set(), (0, [])])
def test_reject_mutable_values(value: object) -> None:
    with pytest.raises(TypeError, match="immutable"):
        validate_immutable(value)


@given(st.recursive(st.one_of(st.none(), st.integers(), st.text()), st.tuples))
def test_immutable_values(value: object) -> None:
    validate_immutable(value)


def test_primitives_require_execution() -> None:
    with pytest.raises(RuntimeError, match="inside"):
        ploom.Lock()


def test_scheduler_joins_owned_threads() -> None:
    def scenario() -> None:
        task = ploom.Thread(target=ploom.yield_now)
        task.start()

    scheduler = ThreadScheduler(ploom.Config(max_steps=100, max_tasks=16))
    execution = scheduler.execute(scenario, (), None)
    assert execution.outcome.kind == "passed"
    assert not any(thread.name.startswith("ploom-") for thread in threading.enumerate())
