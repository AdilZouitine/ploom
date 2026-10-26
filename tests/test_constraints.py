from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

import ploom
from ploom.execution import Execution, FailureCategory, Passed, Rejected
from ploom.exploration import explore, minimize
from ploom.thread_scheduler import ThreadScheduler
from tests.helpers import failure_of


def rejection_of(execution: Execution) -> str:
    match execution.outcome:
        case Rejected(message):
            return message
        case outcome:
            pytest.fail(f"expected a rejection, observed {outcome}")


def point(thread: str, event: str, occurrence: int = 1) -> ploom.Point:
    return ploom.Point(thread=thread, event=event, occurrence=occurrence)


def before(first: ploom.Point, second: ploom.Point) -> ploom.Before:
    return ploom.Before(before=first, after=second)


class OrderedRunner:
    def __init__(self, scenario: Callable[[], object], config: ploom.Config) -> None:
        self.scenario = scenario
        self.config = config

    def run(
        self,
        required_prefix: tuple[int, ...],
        _timeout_seconds: float,
        expected_execution: Execution | None = None,
    ) -> Execution:
        return ThreadScheduler(self.config).execute(
            self.scenario, required_prefix, expected_execution
        )


def start_workers(*workers: tuple[str, Callable[[], object]]) -> None:
    threads = [ploom.Thread(name=name, target=target) for name, target in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


@pytest.mark.parametrize(
    "edges",
    [
        (before(point("A", "x"), point("A", "x")),),
        (before(point("A", "x"), point("B", "y")), before(point("B", "y"), point("A", "x"))),
        (before(point("A", "@finish"), point("A", "@start")),),
        (before(point("A", "x", 2), point("A", "x", 1)),),
        (before(point("A", "@unknown"), point("B", "x")),),
        (before(point("A", "@finish", 2), point("B", "x")),),
    ],
)
def test_invalid_orders(edges: tuple[ploom.Before, ...]) -> None:
    with pytest.raises(ValidationError):
        ploom.Config(ordering=edges)


def test_lifecycle_order_and_cleanup() -> None:
    ordering = (
        before(point("Y", "@finish"), point("A", "@finish")),
        before(point("A", "@finish"), point("X", "@start")),
    )

    def scenario() -> None:
        cleaned = ploom.SharedCell(False)

        def a() -> None:
            try:
                ploom.yield_now()
            finally:
                cleaned.write(True)

        def x() -> None:
            assert cleaned.read()
            assert not a_thread.is_alive()

        x_thread = ploom.Thread(name="X", target=x)
        a_thread = ploom.Thread(name="A", target=a)
        y_thread = ploom.Thread(name="Y")
        for thread in (x_thread, a_thread, y_thread):
            thread.start()
        for thread in (x_thread, a_thread, y_thread):
            thread.join()

    config = ploom.Config(ordering=ordering, max_executions=100_000)
    runner = OrderedRunner(scenario, config)
    execution = runner.run((), 10)
    assert execution.outcome == Passed()
    events = {
        (step.task, step.operation.split(":", 1)[1]): index
        for index, step in enumerate(execution.steps)
    }
    assert events[3, "checkpoint:@finish:1"] < events[2, "checkpoint:@finish:1"]
    assert events[2, "checkpoint:@finish:1"] < events[1, "checkpoint:@start:1"]
    # Replay verifies task choices, event identities, and constrained runnable sets.
    assert runner.run(tuple(step.task for step in execution.steps), 10, execution) == execution


def lost_update() -> Callable[[], None]:
    def scenario() -> None:
        cell = ploom.SharedCell(0)

        def increment() -> None:
            value = cell.read()
            ploom.checkpoint("read")
            ploom.checkpoint("write")
            cell.write(value + 1)

        threads = [ploom.Thread(name=name, target=increment) for name in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        expected = len(threads)
        assert cell.read() == expected

    return scenario


def race_order() -> tuple[ploom.Before, ...]:
    return (
        before(point("A", "read"), point("B", "read")),
        before(point("B", "read"), point("A", "write")),
        before(point("A", "@finish"), point("B", "write")),
    )


def test_constrained_race_artifacts_and_replay(tmp_path: Path) -> None:
    scenario = lost_update()
    config = ploom.Config(ordering=race_order(), minimize_timeout=10.0)
    with pytest.raises(ploom.ModelCheckError) as raised:
        ploom.check(scenario, config=config, trace_dir=tmp_path)
    failure = raised.value
    original = ploom.read_trace(failure.trace_path)
    reduced = ploom.read_trace(failure.minimized_path)
    assert original.config.ordering == config.ordering == reduced.config.ordering
    assert ploom.replay(scenario, failure.trace_path) == original
    assert ploom.replay(scenario, failure.minimized_path) == reduced


def test_explores_all_remaining_choices_and_skips_rejected_branches() -> None:
    def scenario() -> None:
        def a() -> None:
            ploom.checkpoint("first")
            ploom.checkpoint("second")

        def b() -> None:
            ploom.checkpoint("first")
            ploom.checkpoint("second")

        start_workers(("A", a), ("B", b))

    config = ploom.Config(
        ordering=(before(point("B", "second"), point("A", "first")),),
        max_executions=100_000,
    )
    result = explore(OrderedRunner(scenario, config), config)
    assert result.execution.outcome == Passed()
    assert result.statistics.executions > 1


def test_missing_events_and_impossible_program_order_never_pass() -> None:
    def scenario() -> None:
        def a() -> None:
            ploom.checkpoint("first")
            ploom.checkpoint("second")

        start_workers(("A", a))

    for ordering in (
        (before(point("missing", "@finish"), point("A", "first")),),
        (before(point("A", "absent"), point("A", "first")),),
        (before(point("A", "second"), point("A", "first")),),
    ):
        config = ploom.Config(ordering=ordering, max_executions=100_000)
        execution = explore(OrderedRunner(scenario, config), config).execution
        assert "missing events" in rejection_of(execution)


def test_finished_execution_missing_an_event_is_rejected() -> None:
    def scenario() -> None:
        start_workers(("A", lambda: ploom.checkpoint("x")), ("B", lambda: None))

    config = ploom.Config(ordering=(before(point("A", "x"), point("B", "y")),))
    assert "missing events" in rejection_of(OrderedRunner(scenario, config).run((), 10))


def test_check_raises_when_no_schedule_satisfies_the_order() -> None:
    def scenario() -> None:
        thread = ploom.Thread(name="A", target=lambda: ploom.checkpoint("first"))
        thread.start()
        thread.join()

    config = ploom.Config(ordering=(before(point("A", "absent"), point("A", "first")),))
    with pytest.raises(ploom.ScheduleConstraintError, match="missing events"):
        ploom.check(scenario, config=config)


def test_repeated_checkpoints_are_distinct() -> None:
    def scenario() -> None:
        def worker() -> None:
            for occurrence in (1, 2):
                ploom.checkpoint("loop", occurrence=occurrence)

        start_workers(("A", worker), ("B", worker))

    config = ploom.Config(
        ordering=(before(point("A", "loop", 2), point("B", "loop", 1)),),
        max_executions=100_000,
    )
    result = explore(OrderedRunner(scenario, config), config)
    assert result.execution.outcome == Passed()


def checkpoint_same() -> None:
    ploom.checkpoint("same")


def checkpoint_same_twice() -> None:
    ploom.checkpoint("same")
    ploom.checkpoint("same")


def duplicate_thread_names() -> None:
    start_workers(("A", checkpoint_same), ("A", checkpoint_same))


def duplicate_points() -> None:
    start_workers(("A", checkpoint_same_twice))


def unnamed_checkpoint() -> None:
    ploom.checkpoint("unnamed")


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        (duplicate_thread_names, "ambiguous thread name"),
        (duplicate_points, "duplicate checkpoint"),
        (unnamed_checkpoint, "require a named Thread"),
    ],
)
def test_ambiguous_names_rejected(scenario: Callable[[], None], message: str) -> None:
    config = ploom.Config()
    assert message in rejection_of(explore(OrderedRunner(scenario, config), config).execution)


@pytest.mark.parametrize("name", ["", "@start"])
def test_reserved_checkpoint_names_fail(name: str) -> None:
    def scenario() -> None:
        ploom.checkpoint(name)

    result = OrderedRunner(scenario, ploom.Config()).run((), 10)
    assert failure_of(result).exception_type == "builtins.ValueError"


def test_real_deadlock_after_constraints_is_still_deadlock() -> None:
    def scenario() -> None:
        def worker() -> None:
            lock = ploom.Lock()
            ploom.checkpoint("one")
            ploom.checkpoint("two")
            lock.acquire()
            lock.acquire()

        start_workers(("A", worker))

    config = ploom.Config(ordering=(before(point("A", "one"), point("A", "two")),))
    result = OrderedRunner(scenario, config).run((), 10)
    assert failure_of(result).category is FailureCategory.DEADLOCK


def test_limits_are_not_unsatisfiable_and_rejected_traces_are_invalid() -> None:
    config = ploom.Config(max_steps=1)
    with pytest.raises(ploom.IncompleteExplorationError):
        ploom.check(lost_update(), config=config)
    execution = Execution((), Rejected("unsatisfiable"))
    with pytest.raises(ValueError, match="only passed or failed"):
        ploom.Trace.from_execution(
            ploom.Provenance(
                reference="test", digest="x", transport=ploom.ScenarioTransport.LOCAL, python="3.13"
            ),
            config,
            execution,
        )
    with pytest.raises(ValueError, match="only failing"):
        minimize(OrderedRunner(lost_update(), config), execution, config)


def test_rejected_conditional_prefix_does_not_hide_a_compatible_schedule() -> None:
    def scenario() -> None:
        ready = ploom.SharedCell(False)

        def a() -> None:
            if ready.read():
                ploom.checkpoint("conditional")

        def b() -> None:
            ready.write(True)

        start_workers(("A", a), ("B", b))

    config = ploom.Config(
        ordering=(before(point("A", "conditional"), point("A", "@finish")),),
        max_executions=100_000,
    )
    runner = OrderedRunner(scenario, config)
    rejection_of(runner.run((), 10))
    assert explore(runner, config).execution.outcome == Passed()


def test_fixed_scenario_passes_every_compatible_schedule() -> None:
    def scenario() -> None:
        counter = ploom.SharedCell(0)
        lock = ploom.Lock()

        def increment() -> None:
            with lock:
                value = counter.read()
                ploom.checkpoint("read")
                counter.write(value + 1)

        start_workers(("A", increment), ("B", increment))
        expected = 2
        assert counter.read() == expected

    config = ploom.Config(
        ordering=(before(point("A", "read"), point("B", "read")),),
        max_executions=100_000,
    )
    assert explore(OrderedRunner(scenario, config), config).execution.outcome == Passed()


def test_real_deadlock_before_pending_finish_constraint() -> None:
    def scenario() -> None:
        def worker() -> None:
            lock = ploom.Lock()
            lock.acquire()
            lock.acquire()

        start_workers(("A", worker))

    config = ploom.Config(ordering=(before(point("A", "@start"), point("A", "@finish")),))
    execution = OrderedRunner(scenario, config).run((), 10)
    assert failure_of(execution).category is FailureCategory.DEADLOCK


def test_duplicate_display_names_without_named_events_remain_supported() -> None:
    def scenario() -> None:
        start_workers(("same", lambda: None), ("same", lambda: None))

    assert OrderedRunner(scenario, ploom.Config()).run((), 10).outcome == Passed()


@pytest.mark.parametrize("constrained", [False, True])
def test_names_cannot_become_ambiguous_after_identity_use(constrained: bool) -> None:
    def scenario() -> None:
        def worker() -> None:
            ploom.checkpoint("used")
            ploom.Thread(name="A")

        start_workers(("A", worker))

    config = ploom.Config(
        ordering=(before(point("A", "@start"), point("A", "@finish")),) if constrained else (),
    )
    execution = explore(OrderedRunner(scenario, config), config).execution
    assert "duplicate thread name" in rejection_of(execution)


def test_failure_on_allowed_prefix_is_not_hidden_by_future_event() -> None:
    def scenario() -> None:
        def worker() -> None:
            raise AssertionError("failure before finish")

        start_workers(("A", worker))

    config = ploom.Config(ordering=(before(point("A", "@start"), point("A", "@finish")),))
    result = explore(OrderedRunner(scenario, config), config)
    assert failure_of(result.execution).message == "failure before finish"
