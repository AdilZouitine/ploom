"""Explore schedules, preserve first failures, and reduce them independently."""

import logging
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol, assert_never, override
from uuid import uuid4

from ploom.execution import (
    Diverged,
    Execution,
    ExplorationLimit,
    Failed,
    Failure,
    FailureCategory,
    Incomplete,
    Passed,
    ReductionStatus,
    Statistics,
    alternative_prefixes,
    count_preemptions,
)
from ploom.partial_order import SleepPrefix, expand_sleep_prefix
from ploom.subprocess_runner import subprocess_runner
from ploom.trace import Config, Trace, read_trace, write_trace

logger = logging.getLogger(__name__)


class ExecutionRunner(Protocol):
    """Supply fresh executions without coupling search to its process backend."""

    def run(
        self,
        required_prefix: tuple[int, ...],
        timeout_seconds: float,
        expected_execution: Execution | None = None,
        /,
    ) -> Execution:
        """Execute a prefix or validate an exact replay within a deadline."""
        ...


type FailureObserver = Callable[[Trace, Path], None]


@dataclass(frozen=True)
class Reduction:
    """Keep reduction work independent from the original exploration result.

    Attributes:
        execution: Best matching counterexample found within the reduction budget.
        status: Whether reduction completed, ran out of budget, or was disabled.
        executions: Additional executions attempted during reduction.
    """

    execution: Execution
    status: ReductionStatus
    executions: int


class ModelCheckError(AssertionError):
    """Expose reproducible task failure evidence to ordinary pytest assertions.

    The message holds the primary failure. Cleanup failures are exception notes.

    Attributes:
        trace: Original counterexample, preserved before reduction.
        trace_path: Published original JSON artifact.
        failure: The original trace's failure.
        minimized_path: Published reduction artifact.
        reduction: Reduction status, execution count, and best execution.
        statistics: Exploration work, excluding reduction.
    """

    def __init__(
        self,
        trace: Trace,
        trace_path: Path,
        reduction: Reduction,
        minimized_path: Path,
        statistics: Statistics,
    ) -> None:
        match trace.outcome:
            case Failed(failure):
                self.failure = failure
            case Passed():
                raise ValueError("failure exception requires a failed trace")
            case _ as unreachable:
                assert_never(unreachable)
        super().__init__(trace, trace_path, reduction, minimized_path, statistics)
        self.trace = trace
        self.trace_path = trace_path
        self.reduction = reduction
        self.minimized_path = minimized_path
        self.statistics = statistics
        for cleanup_failure in trace.cleanup_failures:
            self.add_note(
                f"Cleanup failure: {cleanup_failure.exception_type}: {cleanup_failure.message}"
            )

    @override
    def __str__(self) -> str:
        """Summarize the failure, its artifacts, and its original traceback."""
        return (
            f"{self.failure.exception_type}: {self.failure.message}\n"
            f"Original trace: {self.trace_path}\nMinimized trace: {self.minimized_path}\n"
            f"Minimization: {self.reduction.status}\n{self.failure.traceback}"
        ).rstrip()

    @property
    def minimized_trace(self) -> Trace:
        """Derive minimized evidence from its sole owning execution result."""
        return self.trace.with_execution(self.reduction.execution)


class DeadlockError(ModelCheckError):
    """Distinguish modeled deadlock from task assertions and resource limits."""


def model_check_error_type(category: FailureCategory) -> type[ModelCheckError]:
    """Select the exception class that names a failure category."""
    match category:
        case FailureCategory.DEADLOCK:
            return DeadlockError
        case FailureCategory.EXCEPTION:
            return ModelCheckError
        case _ as unreachable:
            assert_never(unreachable)


class IncompleteExplorationError(RuntimeError):
    """Report a limit without implying exhaustive success.

    Attributes:
        limit: The ExplorationLimit enum member that ended the search.
        statistics: Work observed before that limit was reached.
        cleanup_failures: Secondary exceptions raised while stopping the execution,
            also attached as exception notes.
    """

    def __init__(
        self,
        limit: ExplorationLimit,
        statistics: Statistics,
        cleanup_failures: tuple[Failure, ...] = (),
    ) -> None:
        super().__init__(limit, statistics, cleanup_failures)
        self.limit = limit
        self.statistics = statistics
        self.cleanup_failures = cleanup_failures
        for cleanup_failure in cleanup_failures:
            self.add_note(cleanup_failure.traceback)

    @override
    def __str__(self) -> str:
        """Name the limit and the work done before it was reached."""
        return f"exploration incomplete: {self.limit} limit reached ({self.statistics})"


class ReplayDivergenceError(RuntimeError):
    """Reject evidence that no longer describes the current execution."""


class ScheduleSearch:
    """Run each schedule of a deterministic scenario once, in depth-first order.

    Exploration and reduction walk the same tree with different stopping rules,
    so both consume this iterator. Branches of an execution are computed after
    the consumer has seen it, so ``preemption_bound`` may tighten in response.
    A negative bound admits no schedule and ends the search. When
    ``max_executions`` executions have run and schedules remain, the search
    yields an unexecuted ``Incomplete(SCHEDULES)`` execution and ends.

    Attributes:
        executions_run: Executions run so far.
        steps_observed: Modeled steps observed so far.
    """

    def __init__(
        self,
        runner: ExecutionRunner,
        deadline: float,
        max_executions: int,
        preemption_bound: Callable[[], int | None],
        *,
        partial_order_reduction: bool = False,
    ) -> None:
        self.runner = runner
        self.deadline = deadline
        self.max_executions = max_executions
        self.preemption_bound = preemption_bound
        self.partial_order_reduction = partial_order_reduction
        self.executions_run = 0
        self.steps_observed = 0

    def __iter__(self) -> Iterator[Execution]:
        """Yield executions until the tree, the bound, or the budget is exhausted."""
        # prefixes queued under a looser bound still run after the bound
        # tightens. Store each prefix's preemption count to skip them at pop time.
        if self.partial_order_reduction and self.preemption_bound() is None:
            yield from self.sleep_set_executions()
            return
        unexplored_prefixes: list[tuple[int, ...]] = [()]
        while unexplored_prefixes:
            preemption_bound = self.preemption_bound()
            if preemption_bound is not None and preemption_bound < 0:
                return
            if self.budget_exhausted:
                yield Execution((), Incomplete(ExplorationLimit.SCHEDULES))
                return
            required_prefix = unexplored_prefixes.pop()
            execution = self.run_schedule(required_prefix)
            yield execution
            branches = alternative_prefixes(
                execution.steps, len(required_prefix), self.preemption_bound()
            )
            unexplored_prefixes.extend(reversed(branches))

    def sleep_set_executions(self) -> Iterator[Execution]:
        """Traverse sleep-set DFS, preserving incomplete and divergent outcomes.

        Workers still run complete schedules. A suffix entering a sleeping
        transition is discarded only after its awake alternatives are queued.
        This avoids coupling the worker protocol to the search algorithm.
        """
        unexplored_sleep_prefixes = [SleepPrefix()]
        while unexplored_sleep_prefixes:
            if self.budget_exhausted:
                yield Execution((), Incomplete(ExplorationLimit.SCHEDULES))
                return
            sleep_prefix = unexplored_sleep_prefixes.pop()
            execution = self.run_schedule(sleep_prefix.tasks)
            match execution.outcome:
                case Incomplete() | Diverged():
                    yield execution
                    return
                case Passed() | Failed():
                    expansion = expand_sleep_prefix(execution.steps, sleep_prefix)
                    unexplored_sleep_prefixes.extend(expansion.branches)
                    if expansion.is_representative:
                        yield execution
                case _ as unreachable:
                    assert_never(unreachable)

    @property
    def budget_exhausted(self) -> bool:
        """Whether ``max_executions`` executions have already run."""
        return self.executions_run == self.max_executions

    def run_schedule(self, required_prefix: tuple[int, ...]) -> Execution:
        """Run one schedule within the remaining time and count it."""
        execution = self.runner.run(required_prefix, self.deadline - monotonic())
        self.executions_run += 1
        self.steps_observed += len(execution.steps)
        return execution


@dataclass(frozen=True)
class SearchResult:
    """Keep terminal evidence and aggregate statistics together."""

    execution: Execution
    statistics: Statistics


def explore(runner: ExecutionRunner, config: Config) -> SearchResult:
    """Return the first execution that does not pass, or a pass after every schedule.

    With ``config.max_preemptions`` set, only schedules within that bound run.
    A limit never counts as exhaustive success.

    Args:
        runner: Backend that provides fresh executions for scheduling prefixes.
        config: Validated exploration limits.

    Returns:
        The first failing, incomplete, or divergent execution, or a pass.
    """
    started_at = monotonic()
    search = ScheduleSearch(
        runner,
        started_at + config.timeout,
        config.max_executions,
        lambda: config.max_preemptions,
        partial_order_reduction=config.partial_order_reduction,
    )
    first_unpassed_execution = next(
        (execution for execution in search if execution.outcome != Passed()),
        Execution((), Passed()),
    )
    statistics = Statistics(search.executions_run, search.steps_observed, monotonic() - started_at)
    return SearchResult(first_unpassed_execution, statistics)


def minimize(runner: ExecutionRunner, original: Execution, config: Config) -> Reduction:
    """Search for the same failure with fewer preemptions under a separate budget.

    The search only visits schedules with fewer preemptions than the best one so
    far. ``COMPLETED`` therefore means that no schedule with fewer preemptions
    reproduces the failure.

    Args:
        runner: Backend that provides fresh executions for scheduling prefixes.
        original: The counterexample already preserved by the caller.
        config: Validated limits for additional reduction work.

    Returns:
        The best counterexample found and whether reduction finished or stopped.

    Raises:
        ValueError: If the original execution did not fail.
        ReplayDivergenceError: If the scenario is nondeterministic.
    """
    match original.outcome:
        case Failed(failure):
            original_signature = failure.signature
        case Passed() | Incomplete() | Diverged():
            raise ValueError("only failing executions can be minimized")
        case _ as unreachable:
            assert_never(unreachable)
    if config.minimize_executions == 0 or config.minimize_timeout == 0:
        return Reduction(original, ReductionStatus.DISABLED, 0)
    best_execution = original
    search = ScheduleSearch(
        runner,
        monotonic() + config.minimize_timeout,
        config.minimize_executions,
        lambda: count_preemptions(best_execution.steps) - 1,
    )
    for execution in search:
        match execution.outcome:
            case Diverged(message):
                raise ReplayDivergenceError(message)
            case Incomplete(ExplorationLimit.SCHEDULES | ExplorationLimit.WALL_TIME):
                return Reduction(
                    best_execution, ReductionStatus.BUDGET_EXHAUSTED, search.executions_run
                )
            case Failed(failure) if failure.signature == original_signature and count_preemptions(
                execution.steps
            ) < count_preemptions(best_execution.steps):
                best_execution = execution
            case Passed() | Failed() | Incomplete():
                pass
            case _ as unreachable:
                assert_never(unreachable)
    return Reduction(best_execution, ReductionStatus.COMPLETED, search.executions_run)


def log_failure(trace: Trace, trace_path: Path) -> None:
    """Name the saved original before minimization when no observer is supplied."""
    logger.warning(
        "Failure trace saved before minimization: %s (%s)", trace_path, trace.scenario.reference
    )


@contextmanager
def note_on_error(note: str) -> Generator[None]:
    """Attach context to any exception that leaves the block, then re-raise it."""
    try:
        yield
    except BaseException as error:
        error.add_note(note)
        raise


def check(
    scenario: Callable[[], object],
    *,
    config: Config = Config(),
    trace_dir: Path = Path(".ploom"),
    on_failure: FailureObserver = log_failure,
) -> Statistics:
    """Exhaustively check modeled interleavings or raise with explicit evidence.

    Scenarios are ordinary zero-argument functions. Use closures, functools.partial,
    or pytest parametrization for inputs, including optional Pydantic input models.

    The original trace is published and reported before minimization starts. Any
    later error, such as a worker crash, propagates with the modeled failure in
    its notes, and the original trace stays on disk.

    Args:
        scenario: Code that constructs primitives, runs tasks, and asserts invariants.
        config: Validated resource limits.
        trace_dir: Directory for original and reduced JSON counterexamples.
        on_failure: Called with the original trace and path before reduction starts.

    Returns:
        Exploration statistics only after every reachable modeled schedule passes,
        or every schedule within ``config.max_preemptions`` when it is set.

    Raises:
        ModelCheckError: If an execution raises an exception.
        DeadlockError: If unfinished tasks remain but none can proceed.
        IncompleteExplorationError: If a resource bound prevents completion.
        ReplayDivergenceError: If observed executions contradict a required prefix.

    Examples:
        >>> def scenario():
        ...     pass
        >>> check(scenario).executions
        1
    """
    with subprocess_runner(scenario, config) as runner:
        search_result = explore(runner, config)
        execution = search_result.execution
        match execution.outcome:
            case Passed():
                return search_result.statistics
            case Incomplete(limit):
                raise IncompleteExplorationError(
                    limit, search_result.statistics, execution.cleanup_failures
                )
            case Diverged(message):
                raise ReplayDivergenceError(message)
            case Failed(failure):
                trace = Trace.from_execution(runner.prepared_scenario.provenance, config, execution)
                trace_name = uuid4().hex
                original_path = trace_dir / f"{trace_name}-original.json"
                minimized_path = trace_dir / f"{trace_name}-minimized.json"
                with note_on_error(
                    f"Modeled failure: {failure.exception_type}: {failure.message}\n"
                    f"{failure.traceback}"
                ):
                    write_trace(original_path, trace)
                    on_failure(trace, original_path)
                    reduction = minimize(runner, execution, config)
                    write_trace(minimized_path, trace.with_execution(reduction.execution))
                raise model_check_error_type(failure.category)(
                    trace, original_path, reduction, minimized_path, search_result.statistics
                )
            case _ as unreachable:
                assert_never(unreachable)


def replay(scenario: Callable[[], object], trace: Trace | Path) -> Trace:
    """Execute exactly one recorded schedule and reject inconsistent evidence.

    Reproduced failures are returned as evidence rather than raised again.

    Args:
        scenario: The original scenario with unchanged code and inputs.
        trace: A validated trace or the path to its JSON artifact.

    Returns:
        The original trace after its operations and terminal outcome reproduce.

    Raises:
        ReplayDivergenceError: If provenance, operations, or terminal evidence differ.
        IncompleteExplorationError: If the replay exceeds its resource limits.
    """
    match trace:
        case Path():
            recorded_trace = read_trace(trace)
        case Trace():
            recorded_trace = trace
        case _ as unreachable:
            assert_never(unreachable)
    started_at = monotonic()
    with subprocess_runner(scenario, recorded_trace.config) as runner:
        if runner.prepared_scenario.provenance != recorded_trace.scenario:
            raise ReplayDivergenceError("scenario source, reference, or interpreter changed")
        execution = runner.run(
            tuple(step.task for step in recorded_trace.steps),
            recorded_trace.config.timeout,
            recorded_trace.execution(),
        )
    match execution.outcome:
        case Diverged(message):
            raise ReplayDivergenceError(message)
        case Incomplete(limit):
            raise IncompleteExplorationError(
                limit,
                Statistics(1, len(execution.steps), monotonic() - started_at),
                execution.cleanup_failures,
            )
        case Passed() | Failed():
            return recorded_trace
        case _ as unreachable:
            assert_never(unreachable)
