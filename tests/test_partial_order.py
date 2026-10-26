from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ploom
from ploom.execution import (
    Diverged,
    Execution,
    ExplorationLimit,
    Incomplete,
    Passed,
    replay_identity,
)
from ploom.exploration import ScheduleSearch, explore
from ploom.partial_order import PendingAction, independent
from ploom.thread_scheduler import ThreadScheduler
from tests.helpers import failure_of

WORKER_COUNT = 2


@dataclass
class InlineRunner:
    scenario: Callable[[], object]
    config: ploom.Config

    def run(
        self,
        required_prefix: tuple[int, ...],
        _timeout_seconds: float,
        expected_execution: Execution | None = None,
    ) -> Execution:
        return ThreadScheduler(
            self.config.max_steps,
            self.config.max_tasks,
            partial_order_reduction=self.config.partial_order_reduction,
        ).execute(self.scenario, required_prefix, expected_execution)


def independent_scenario() -> None:
    cells = (ploom.SharedCell(0), ploom.SharedCell(0))

    def update(index: int) -> None:
        cells[index].write(cells[index].read() + 1)
        assert cells[index].read() == 1

    workers = [ploom.Thread(target=update, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert cells[0].read() == cells[1].read() == 1


def config(*, reduced: bool, bound: int | None = None) -> ploom.Config:
    return ploom.Config(
        partial_order_reduction=reduced,
        max_preemptions=bound,
        max_executions=50_000,
        timeout=60.0,
        minimize_executions=0,
    )


def test_independent_operations_reduce_executions() -> None:
    exhaustive = config(reduced=False)
    reduced = config(reduced=True)
    baseline = explore(InlineRunner(independent_scenario, exhaustive), exhaustive)
    result = explore(InlineRunner(independent_scenario, reduced), reduced)
    assert baseline.execution.outcome == result.execution.outcome == Passed()
    assert result.statistics.executions < baseline.statistics.executions


@pytest.mark.parametrize("bound", [0, 1, 2])
def test_preemption_bounds_fall_back_to_exhaustive_search(bound: int) -> None:
    exhaustive = config(reduced=False, bound=bound)
    reduced = config(reduced=True, bound=bound)
    baseline = explore(InlineRunner(independent_scenario, exhaustive), exhaustive)
    result = explore(InlineRunner(independent_scenario, reduced), reduced)
    assert baseline.execution.outcome == result.execution.outcome == Passed()
    assert result.statistics.executions == baseline.statistics.executions


Operation = st.tuples(st.sampled_from(["read", "write"]), st.integers(0, 1))


@given(st.lists(Operation, min_size=1, max_size=2), st.lists(Operation, min_size=1, max_size=2))
@settings(max_examples=12, deadline=None)
def test_generated_programs_preserve_every_terminal_observation(
    left: list[tuple[str, int]], right: list[tuple[str, int]]
) -> None:
    def scenario() -> None:
        cells = (ploom.SharedCell(0), ploom.SharedCell(0))

        def execute(program: list[tuple[str, int]], value: int) -> None:
            local = value
            for kind, target in program:
                if kind == "read":
                    local = cells[target].read()
                else:
                    cells[target].write(local)

        workers = [
            ploom.Thread(target=execute, args=(left, 1)),
            ploom.Thread(target=execute, args=(right, 2)),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        observation = (cells[0].read(), cells[1].read())
        raise AssertionError(str(observation))

    assert terminal_observations(scenario, False) == terminal_observations(scenario, True)


def terminal_observations(scenario: Callable[[], None], reduced: bool) -> set[tuple[str, str]]:
    limits = config(reduced=reduced)
    search = ScheduleSearch(
        InlineRunner(scenario, limits),
        monotonic() + 60,
        50_000,
        lambda: None,
        partial_order_reduction=reduced,
    )
    result: set[tuple[str, str]] = set()
    for execution in search:
        failure = failure_of(execution)
        result.add((failure.category, failure.message))
    return result


def test_data_dependent_next_operations_preserve_observations() -> None:
    def scenario() -> None:
        cells = (ploom.SharedCell(0), ploom.SharedCell(0), ploom.SharedCell(0))

        def conditional() -> None:
            target = 1 if cells[0].read() else 2
            cells[target].write(7)

        def publish() -> None:
            cells[0].write(1)

        workers = [ploom.Thread(target=conditional), ploom.Thread(target=publish)]
        for worker in workers:
            worker.start()
        # The root also interacts with workers while they are runnable.
        cells[0].write(0)
        for worker in workers:
            worker.join()
        raise AssertionError(str((cells[1].read(), cells[2].read())))

    assert terminal_observations(scenario, False) == terminal_observations(scenario, True)


def test_lock_enabledness_and_deadlocks_preserve_observations() -> None:
    def scenario() -> None:
        lock = ploom.Lock()
        cell = ploom.SharedCell(0)

        def attempt() -> None:
            if lock.acquire(blocking=False):
                cell.write(1)
                lock.release()
            else:
                cell.write(2)

        def hold() -> None:
            lock.acquire()
            # Finishing without release creates some genuinely deadlocked orders.
            if cell.read():
                lock.release()

        workers = [ploom.Thread(target=attempt), ploom.Thread(target=hold)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        lock.acquire()
        raise AssertionError(str(cell.read()))

    exhaustive = terminal_observations(scenario, False)
    assert exhaustive == terminal_observations(scenario, True)
    assert any(category == "deadlock" for category, _ in exhaustive)


def test_three_workers_preserve_observations() -> None:
    def scenario() -> None:
        cells = (ploom.SharedCell(0), ploom.SharedCell(0))

        def write(target: int, value: int) -> None:
            cells[target].write(value)

        workers = [ploom.Thread(target=write, args=args) for args in ((0, 1), (1, 2), (0, 3))]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        raise AssertionError(str((cells[0].read(), cells[1].read())))

    assert terminal_observations(scenario, False) == terminal_observations(scenario, True)


def lost_update() -> None:
    cell = ploom.SharedCell(0)

    def increment() -> None:
        cell.write(cell.read() + 1)

    workers = [ploom.Thread(target=increment) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert cell.read() == WORKER_COUNT


def deadlock() -> None:
    locks = (ploom.Lock(), ploom.Lock())

    def lock_both(first: int, second: int) -> None:
        with locks[first], locks[second]:
            pass

    workers = [
        ploom.Thread(target=lock_both, args=(0, 1)),
        ploom.Thread(target=lock_both, args=(1, 0)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


@pytest.mark.parametrize("scenario", [lost_update, deadlock])
def test_reduced_search_finds_races_and_deadlocks(scenario: Callable[[], None]) -> None:
    limits = config(reduced=True)
    result = explore(InlineRunner(scenario, limits), limits)
    failure_of(result.execution)
    runner = InlineRunner(scenario, limits)
    replayed = runner.run(tuple(step.task for step in result.execution.steps), 60, result.execution)
    assert replayed.steps == result.execution.steps
    assert replay_identity(replayed) == replay_identity(result.execution)


def test_public_check_replay_and_minimization(tmp_path: Path) -> None:
    def scenario() -> None:
        cell = ploom.SharedCell(0)

        def increment() -> None:
            cell.write(cell.read() + 1)

        workers = [ploom.Thread(target=increment) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        assert cell.read() == WORKER_COUNT

    limits = ploom.Config(
        partial_order_reduction=True,
        max_executions=5000,
        timeout=60,
        minimize_executions=100,
        minimize_timeout=10,
    )
    with pytest.raises(ploom.ModelCheckError) as caught:
        ploom.check(scenario, config=limits, trace_dir=tmp_path)
    error = caught.value
    assert ploom.replay(scenario, error.trace_path) == error.trace
    assert error.trace.config.partial_order_reduction
    assert error.reduction.executions > 0


@pytest.mark.parametrize("allocate_thread", [False, True])
def test_reduction_rejects_dynamic_allocation(allocate_thread: bool) -> None:
    def scenario() -> None:
        worker = ploom.Thread()
        worker.start()
        if allocate_thread:
            ploom.Thread()
        else:
            ploom.SharedCell(0)
        worker.join()

    limits = config(reduced=True)
    result = explore(InlineRunner(scenario, limits), limits)
    assert "constructed before starting" in failure_of(result.execution).message


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("0:cell.read:0", "1:cell.read:0", True),
        ("0:cell.read:0", "1:cell.write:0", False),
        ("0:lock.acquire:0", "1:lock.release:0", False),
        ("0:lock.acquire:0", "1:lock.release:1", True),
        ("0:thread.join:1", "1:cell.write:0", False),
        ("0:yield", "1:cell.read:0", False),
        ("0:future.operation:0", "1:cell.read:1", False),
        ("0:cell.read:bad", "1:cell.read:1", False),
        ("bad:cell.read:0", "1:cell.read:1", False),
    ],
)
def test_conservative_independence(left: str, right: str, expected: bool) -> None:
    assert independent(PendingAction(0, left), PendingAction(1, right)) is expected
    assert not independent(PendingAction(0, left), PendingAction(0, right))


def test_reduction_limits_are_incomplete() -> None:
    limits = ploom.Config(partial_order_reduction=True, max_executions=1)
    result = explore(InlineRunner(independent_scenario, limits), limits)
    assert result.execution.outcome == Incomplete(ExplorationLimit.SCHEDULES)


def test_execution_limits_and_divergence_are_not_discarded() -> None:
    limits = ploom.Config(partial_order_reduction=True, max_steps=1)
    result = explore(InlineRunner(independent_scenario, limits), limits)
    assert result.execution.outcome == Incomplete(ExplorationLimit.STEPS)
    runner = InlineRunner(independent_scenario, config(reduced=True))
    execution = runner.run((999,), 60)
    match execution.outcome:
        case Diverged():
            pass
        case outcome:
            pytest.fail(f"expected divergence, observed {outcome}")
