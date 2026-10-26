import json
import os
import pickle
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from itertools import combinations
from math import comb
from pathlib import Path
from typing import NoReturn, SupportsIndex, override

import cloudpickle
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import ploom
from ploom.execution import (
    Diverged,
    Execution,
    ExplorationLimit,
    Failed,
    Failure,
    FailureCategory,
    Incomplete,
    Outcome,
    Passed,
    ReductionStatus,
    Statistics,
    Step,
    count_preemptions,
)
from ploom.exploration import Reduction, explore, minimize
from ploom.scenarios import ImportableScenario, SerializedScenario, load_scenario, prepare_scenario
from ploom.subprocess_runner import subprocess_runner
from ploom.thread_scheduler import TaskState, ThreadScheduler
from ploom.trace import Provenance, ScenarioTransport
from tests.helpers import failure_of


class InlineRunner:
    def __init__(self, scenario: Callable[[], object], config: ploom.Config) -> None:
        self.scenario = scenario
        self.config = config
        self.executions: list[Execution] = []

    def run(
        self,
        required_prefix: tuple[int, ...],
        timeout_seconds: float,
        expected_execution: Execution | None = None,
    ) -> Execution:
        if timeout_seconds <= 0:
            return Execution((), Incomplete(ExplorationLimit.WALL_TIME))
        execution = ThreadScheduler(self.config).execute(
            self.scenario, required_prefix, expected_execution
        )
        self.executions.append(execution)
        return execution


class ScriptedRunner:
    def __init__(self, outcome: Outcome) -> None:
        self.outcome = outcome

    def run(
        self,
        _required_prefix: tuple[int, ...],
        _timeout_seconds: float,
        _expected_execution: Execution | None = None,
    ) -> Execution:
        return Execution((), self.outcome)


def yielding_scenario(root_yields: int, child_yields: int) -> Callable[[], None]:
    def scenario() -> None:
        def child() -> None:
            for _ in range(child_yields):
                ploom.yield_now()

        task = ploom.Thread(target=child)
        task.start()
        for _ in range(root_yields):
            ploom.yield_now()
        task.join()

    return scenario


def test_lifecycle_and_unsupported_operations() -> None:
    def scenario() -> None:
        with pytest.raises(NotImplementedError, match="groups"):
            ploom.Thread(group=object())
        with pytest.raises(NotImplementedError, match="daemon"):
            ploom.Thread(daemon=True)
        unstarted = ploom.Thread()
        with pytest.raises(RuntimeError, match="before"):
            unstarted.join()
        with pytest.raises(NotImplementedError, match="timed"):
            unstarted.join(timeout=1)
        lock = ploom.Lock()
        with pytest.raises(NotImplementedError, match="timed"):
            lock.acquire(timeout=1)
        with pytest.raises(RuntimeError, match="unlocked"):
            lock.release()
        unstarted.start()
        unstarted.join()
        with pytest.raises(RuntimeError, match="once"):
            unstarted.start()

        def self_join() -> None:
            with pytest.raises(RuntimeError, match="current"):
                child.join()

        child = ploom.Thread(target=self_join)
        child.start()
        child.join()
        ploom.Thread()  # Registered but unstarted tasks own no native resource.

    scheduler = ThreadScheduler(ploom.Config(max_steps=100, max_tasks=16))
    assert scheduler.execute(scenario, (), None).outcome == Passed()
    with pytest.raises(RuntimeError, match="once"):
        scheduler.execute(scenario, (), None)


def test_cross_execution_resources_fail() -> None:
    cells: list[ploom.SharedCell[int]] = []

    def allocate() -> None:
        cells.append(ploom.SharedCell(1))

    assert (
        ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16))
        .execute(allocate, (), None)
        .outcome
        == Passed()
    )

    def use_foreign_resource() -> None:
        cells[0].read()

    execution = ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16)).execute(
        use_foreign_resource, (), None
    )
    assert "different execution" in failure_of(execution).message


def test_partial_thread_acquisition_cleanup() -> None:
    def unavailable(_thread: threading.Thread) -> NoReturn:
        raise OSError("thread resource unavailable")

    scheduler = ThreadScheduler(
        ploom.Config(max_steps=10, max_tasks=16), start_native_thread=unavailable
    )
    task = scheduler.create_task()
    with pytest.raises(OSError, match="unavailable"):
        scheduler.start_task(task, lambda: None)
    assert scheduler.task_state(task) is TaskState.UNSTARTED
    scheduler.close()


def test_cancellation_preserves_original_and_cleanup_failures() -> None:
    def scenario() -> NoReturn:
        def child() -> None:
            try:
                ploom.yield_now()
            finally:
                raise ValueError("cleanup failure")

        ploom.Thread(target=child).start()
        ploom.yield_now()
        raise AssertionError("original failure")

    execution = ThreadScheduler(ploom.Config(max_steps=100, max_tasks=16)).execute(
        scenario, (0, 0, 1, 0), None
    )
    assert failure_of(execution).message == "original failure"
    assert execution.cleanup_failures[0].message == "cleanup failure"


def test_context_manager_cancellation_completes_cleanup() -> None:
    def scenario() -> None:
        lock = ploom.Lock()
        with lock:
            lock.acquire()

    failure = failure_of(
        ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16)).execute(scenario, (), None)
    )
    assert failure.category == "deadlock"
    assert failure.message == "no modeled task can proceed: task 0 at lock.acquire:0"
    assert failure.location == "deadlock: task 0 at lock.acquire:0"


def test_replay_checks_choices_length_and_terminal_outcome() -> None:
    def scenario() -> None:
        ploom.yield_now()

    reference = ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16)).execute(
        scenario, (), None
    )
    missing = replace(reference, steps=reference.steps[:1])
    changed = replace(reference, outcome=Incomplete(ExplorationLimit.TASKS))
    longer = replace(reference, steps=(*reference.steps, reference.steps[-1]))
    for prefix, expected in (((99,), None), ((0,), missing), ((), changed), ((), longer)):
        outcome = (
            ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16))
            .execute(scenario, prefix, expected)
            .outcome
        )
        assert outcome.kind == "diverged", (prefix, expected)
    assert (
        ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16)).execute(scenario, (), reference)
        == reference
    )


def test_minimizer_reduces_preemptions() -> None:
    def scenario() -> NoReturn:
        task = ploom.Thread(target=ploom.yield_now)
        task.start()
        ploom.yield_now()
        task.join()
        raise AssertionError("same failure")

    config = ploom.Config()
    runner = InlineRunner(scenario, config)
    original = runner.run((0, 0, 1, 0, 1), 1)
    assert count_preemptions(original.steps) > 0
    reduction = minimize(runner, original, config)
    assert reduction.status == "completed"
    assert count_preemptions(reduction.execution.steps) == 0
    assert failure_of(reduction.execution).message == "same failure"
    exact_budget = minimize(runner, original, ploom.Config(minimize_executions=1))
    assert exact_budget.status == "completed"
    assert (
        runner.run(tuple(step.task for step in reduction.execution.steps), 1, reduction.execution)
        == reduction.execution
    )
    with pytest.raises(ValueError, match="failing"):
        minimize(runner, Execution((), Passed()), config)


def test_minimization_budget_and_original_persistence(tmp_path: Path) -> None:
    def scenario() -> None:
        counter = ploom.SharedCell(0)

        def increment() -> None:
            counter.write(counter.read() + 1)

        task = ploom.Thread(target=increment)
        task.start()
        increment()
        task.join()
        expected = 2
        assert counter.read() == expected

    config = ploom.Config(minimize_executions=1)
    runner = InlineRunner(scenario, config)
    original = explore(runner, config).execution
    assert failure_of(original).exception_type == "builtins.AssertionError"
    reduction = minimize(runner, original, config)
    assert reduction.status == "budget_exhausted"
    assert reduction.execution == original
    completed = minimize(runner, original, ploom.Config(minimize_timeout=10.0))
    assert completed.status == "completed"

    observed: list[Path] = []

    def persisted(trace: ploom.Trace, path: Path) -> None:
        assert ploom.read_trace(path) == trace
        assert not list(tmp_path.glob("*-minimized.json"))
        observed.append(path)

    with pytest.raises(ploom.ModelCheckError) as caught:
        ploom.check(scenario, trace_dir=tmp_path, on_failure=persisted, config=config)
    assert observed == [caught.value.trace_path]


def test_process_transport_and_worker_crash(tmp_path: Path) -> None:
    def crash() -> NoReturn:
        os._exit(9)

    with subprocess_runner(crash, ploom.Config()) as runner:
        working_directory = runner.working_directory
        assert runner.run((), 0).outcome == Incomplete(ExplorationLimit.WALL_TIME)
        with pytest.raises(RuntimeError, match="worker exited"):
            runner.run((), 5)
    assert not working_directory.exists()

    class Unserializable:
        def __call__(self) -> None:
            pass

        @override
        def __reduce_ex__(self, _protocol: SupportsIndex, /) -> NoReturn:
            raise TypeError("unserializable scenario")

    with (
        pytest.raises(TypeError, match="unserializable"),
        subprocess_runner(Unserializable(), ploom.Config()),
    ):
        pytest.fail("unserializable scenarios must fail before execution")

    payload = tmp_path / "invalid.pickle"
    payload.write_bytes(bytes(cloudpickle.dumps(42)))
    with pytest.raises(TypeError, match="callable"):
        load_scenario(SerializedScenario(path=str(payload)))
    source = ImportableScenario(module="sys", name="getrecursionlimit")
    source()
    assert prepare_scenario(source, tmp_path).provenance.transport == "local"
    with pytest.raises(TypeError, match="callable"):
        ImportableScenario(module="sys", name="version")()
    with pytest.raises(ModuleNotFoundError, match="ploom_absent_module"):
        load_scenario(ImportableScenario(module="ploom_absent_module", name="scenario"))


def test_invalid_failure_artifacts(tmp_path: Path) -> None:
    trace = ploom.Trace(
        scenario=Provenance(
            reference="local",
            digest="digest",
            transport=ScenarioTransport.LOCAL,
            python=sys.version,
        ),
        config=ploom.Config(),
        steps=(),
        outcome=Passed(),
    )
    reduction = Reduction(trace.execution(), ReductionStatus.DISABLED, 0)
    with pytest.raises(ValueError, match="failed trace"):
        ploom.ModelCheckError(
            trace,
            tmp_path,
            reduction=reduction,
            minimized_path=tmp_path,
            statistics=ploom.Statistics(0, 0, 0.0),
        )
    with pytest.raises(ValueError, match="replayable"):
        trace.with_execution(Execution((), Incomplete(ExplorationLimit.STEPS)))


def test_check_attaches_cleanup_failures_as_notes(tmp_path: Path) -> None:
    def scenario() -> NoReturn:
        gate = ploom.Lock()
        held = ploom.Lock()
        gate.acquire()
        held.acquire()

        def child() -> None:
            try:
                gate.release()
                held.acquire()
            finally:
                raise ValueError("cleanup failure")

        ploom.Thread(target=child).start()
        gate.acquire()
        raise AssertionError("original failure")

    with pytest.raises(ploom.ModelCheckError, match="original failure") as caught:
        ploom.check(scenario, trace_dir=tmp_path, config=ploom.Config(minimize_executions=0))
    assert caught.value.__notes__ == ["Cleanup failure: builtins.ValueError: cleanup failure"]


def test_external_nondeterminism_is_rejected_when_prefix_changes(tmp_path: Path) -> None:
    marker = tmp_path / "external-state"

    def scenario() -> None:
        if not marker.exists():
            marker.write_text("changed", encoding="utf-8")
            child = ploom.Thread(target=ploom.yield_now)
            child.start()
            ploom.yield_now()
            child.join()
        else:
            ploom.yield_now()

    with pytest.raises(ploom.ReplayDivergenceError):
        ploom.check(scenario)


def test_replay_timeout_is_incomplete(tmp_path: Path) -> None:
    def scenario() -> None:
        raise AssertionError("failure")

    with pytest.raises(ploom.ModelCheckError) as caught:
        ploom.check(scenario, trace_dir=tmp_path)
    evidence = caught.value.trace.model_copy(update={"config": ploom.Config(timeout=0.001)})
    with pytest.raises(ploom.IncompleteExplorationError) as incomplete:
        ploom.replay(scenario, evidence)
    assert incomplete.value.limit == "wall_time"


@given(root_yields=st.integers(0, 2), child_yields=st.integers(0, 2))
@settings(max_examples=9, deadline=None)
def test_exhaustive_search_matches_independent_interleaving_count(
    root_yields: int, child_yields: int
) -> None:
    runner = InlineRunner(yielding_scenario(root_yields, child_yields), ploom.Config())
    result = explore(runner, ploom.Config())
    middle_length = root_yields + child_yields + 1
    expected = {
        (0, 0, *(0 if index in root_positions else 1 for index in range(middle_length)), 0)
        for root_positions in combinations(range(middle_length), root_yields)
    }
    observed = {tuple(step.task for step in run.steps) for run in runner.executions}
    assert observed == expected
    assert result.statistics.executions == comb(middle_length, root_yields)
    assert result.execution.outcome == Passed()


@given(root_yields=st.integers(0, 2), child_yields=st.integers(0, 2), bound=st.integers(0, 3))
@settings(max_examples=20, deadline=None)
def test_preemption_bound_explores_exactly_the_schedules_within_it(
    root_yields: int, child_yields: int, bound: int
) -> None:
    scenario = yielding_scenario(root_yields, child_yields)
    unbounded = InlineRunner(scenario, ploom.Config())
    explore(unbounded, ploom.Config())
    bounded_config = ploom.Config(max_preemptions=bound)
    bounded = InlineRunner(scenario, bounded_config)
    explore(bounded, bounded_config)
    within = [run.steps for run in unbounded.executions if count_preemptions(run.steps) <= bound]
    assert Counter(run.steps for run in bounded.executions) == Counter(within)


def test_reporter_errors_propagate_with_the_modeled_failure(tmp_path: Path) -> None:
    def scenario() -> NoReturn:
        raise ValueError("original failure")

    def broken_reporter(trace: ploom.Trace, path: Path) -> NoReturn:
        assert ploom.read_trace(path) == trace
        raise OSError("reporter unavailable")

    with pytest.raises(OSError, match="reporter unavailable") as caught:
        ploom.check(scenario, trace_dir=tmp_path, on_failure=broken_reporter)
    assert caught.value.__notes__[0].startswith(
        "Modeled failure: builtins.ValueError: original failure"
    )
    (original,) = tmp_path.glob("*-original.json")
    assert ploom.replay(scenario, original) == ploom.read_trace(original)


def test_worker_errors_during_minimization_keep_the_original_trace(tmp_path: Path) -> None:
    marker = tmp_path / "first-counterexample"

    def scenario() -> None:
        counter = ploom.SharedCell(0)

        def increment() -> None:
            counter.write(counter.read() + 1)

        task = ploom.Thread(target=increment)
        task.start()
        increment()
        task.join()
        expected = 2
        actual = counter.read()
        if actual != expected:
            if marker.exists():
                os._exit(9)
            marker.write_text("original saved", encoding="utf-8")
        assert actual == expected

    with pytest.raises(RuntimeError, match="worker exited") as caught:
        ploom.check(scenario, trace_dir=tmp_path, config=ploom.Config(minimize_timeout=5.0))
    assert caught.value.__notes__[0].startswith("Modeled failure: builtins.AssertionError")
    (original,) = tmp_path.glob("*-original.json")
    # Restore the deliberately external test input to reproduce the original.
    marker.unlink()
    assert ploom.replay(scenario, original) == ploom.read_trace(original)


def test_slow_cancellation_cleanup_keeps_the_original_failure() -> None:
    def scenario() -> NoReturn:
        def child() -> None:
            try:
                ploom.yield_now()
            finally:
                time.sleep(0.2)  # Finite cleanup that outlasts any fixed short join bound.

        ploom.Thread(target=child).start()
        ploom.yield_now()
        raise AssertionError("original")

    execution = ThreadScheduler(ploom.Config(max_steps=100, max_tasks=16)).execute(
        scenario, (0, 0, 1, 0), None
    )
    assert failure_of(execution).message == "original"


def test_stalled_cleanup_is_reported_with_the_original_outcome() -> None:
    release = threading.Event()
    other_task_cleaned = threading.Event()

    def scenario() -> None:
        lock = ploom.Lock()
        lock.acquire()

        def blocked() -> None:
            try:
                lock.acquire()
            finally:
                other_task_cleaned.set()

        def failing() -> NoReturn:
            raise AssertionError("original")

        ploom.Thread(target=blocked).start()
        failer = ploom.Thread(target=failing)
        failer.start()
        try:
            failer.join()
        finally:
            release.wait(timeout=5)

    scheduler = ThreadScheduler(ploom.Config(max_steps=100, max_tasks=16), join_timeout=0.1)
    with ExitStack() as stack:
        stack.callback(release.set)
        caught = stack.enter_context(pytest.raises(RuntimeError, match=r"did not stop: \[0\]"))
        scheduler.execute(scenario, (), None)
    assert other_task_cleaned.is_set()
    assert "AssertionError" in "\n".join(caught.value.__notes__)
    root = scheduler.native_threads[0]
    root.join(timeout=5)
    assert not root.is_alive()


def test_model_check_errors_survive_pickling(tmp_path: Path) -> None:
    def scenario() -> None:
        lock = ploom.Lock()
        lock.acquire()
        lock.acquire()

    with pytest.raises(ploom.DeadlockError) as caught:
        ploom.check(scenario, trace_dir=tmp_path)
    caught.value.add_note("caller context")
    restored = pickle.loads(pickle.dumps(caught.value))
    assert type(restored) is ploom.DeadlockError
    assert str(restored) == str(caught.value)
    assert restored.__notes__ == caught.value.__notes__
    assert restored.minimized_trace == caught.value.minimized_trace

    cleanup = Failure(FailureCategory.EXCEPTION, "builtins.ValueError", "cleanup", "stack", "f:1")
    incomplete = ploom.IncompleteExplorationError(
        ExplorationLimit.STEPS, Statistics(1, 1, 0.0), (cleanup,)
    )
    restored_incomplete = pickle.loads(pickle.dumps(incomplete))
    assert str(restored_incomplete) == str(incomplete)
    assert restored_incomplete.__notes__ == ["stack"]
    assert restored_incomplete.cleanup_failures == (cleanup,)


def test_failure_location_names_the_scenario_frame_not_library_internals() -> None:
    def first() -> None:
        json.loads("{")

    def second() -> None:
        json.loads("{")

    locations = [
        failure_of(
            ThreadScheduler(ploom.Config(max_steps=10, max_tasks=16)).execute(scenario, (), None)
        ).location
        for scenario in (first, second)
    ]
    assert locations[0].endswith("<locals>.first")
    assert locations[1].endswith("<locals>.second")
    assert locations[0].startswith(f"{__name__}:")


PREEMPTED_FAILURE = Execution(
    (Step(0, "0:begin", (0,)), Step(0, "1:yield", (0, 1)), Step(1, "0:begin", (0, 1))),
    Failed(Failure(FailureCategory.EXCEPTION, "builtins.AssertionError", "", "", "f:1")),
)


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (Incomplete(ExplorationLimit.WALL_TIME), ReductionStatus.BUDGET_EXHAUSTED),
        (Incomplete(ExplorationLimit.STEPS), ReductionStatus.COMPLETED),
    ],
)
def test_minimization_status_names_why_it_stopped(
    outcome: Outcome, status: ReductionStatus
) -> None:
    reduction = minimize(ScriptedRunner(outcome), PREEMPTED_FAILURE, ploom.Config())
    assert reduction.status is status
    assert reduction.execution == PREEMPTED_FAILURE


def test_minimization_rejects_nondeterministic_scenarios() -> None:
    runner = ScriptedRunner(Diverged("changed"))
    with pytest.raises(ploom.ReplayDivergenceError, match="changed"):
        minimize(runner, PREEMPTED_FAILURE, ploom.Config())


def test_local_provenance_ignores_captured_values(tmp_path: Path) -> None:
    def make(value: object) -> Callable[[], None]:
        def scenario() -> None:
            assert value

        return scenario

    by_path = prepare_scenario(make(tmp_path / "per-run"), tmp_path).provenance
    by_set = prepare_scenario(make(frozenset({"alpha", "beta"})), tmp_path).provenance
    assert by_path == by_set
    assert by_path.reference == f"{__name__}:{make.__qualname__}.<locals>.scenario"


def test_failed_executions_keep_scenario_output(tmp_path: Path) -> None:
    def scenario() -> NoReturn:
        print("debug output from the scenario")
        raise AssertionError("failure")

    with pytest.raises(ploom.ModelCheckError) as caught:
        ploom.check(scenario, trace_dir=tmp_path, config=ploom.Config(minimize_executions=0))
    assert "Worker output:\ndebug output from the scenario" in str(caught.value)
