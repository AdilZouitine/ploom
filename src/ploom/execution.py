"""Immutable execution evidence shared by the engine and boundary adapters."""

import math
from dataclasses import dataclass, replace
from enum import StrEnum, auto
from itertools import pairwise
from typing import Literal, assert_never


class TaskCheckpointKind(StrEnum):
    """Scheduling points without a primitive target."""

    BEGIN = auto()
    YIELD = auto()


class PrimitiveOperationKind(StrEnum):
    """Atomic modeled operations whose targets have execution-local identities."""

    THREAD_START = "thread.start"
    THREAD_JOIN = "thread.join"
    THREAD_IS_ALIVE = "thread.is_alive"
    CELL_READ = "cell.read"
    CELL_WRITE = "cell.write"
    LOCK_ACQUIRE = "lock.acquire"
    LOCK_TRY_ACQUIRE = "lock.try_acquire"
    LOCK_RELEASE = "lock.release"
    LOCK_LOCKED = "lock.locked"
    EVENT_IS_SET = "event.is_set"
    EVENT_SET = "event.set"
    EVENT_CLEAR = "event.clear"
    EVENT_WAIT = "event.wait"
    EVENT_WAIT_READY = "event.wait_ready"
    RLOCK_ACQUIRE = "rlock.acquire"
    RLOCK_TRY_ACQUIRE = "rlock.try_acquire"
    RLOCK_RELEASE = "rlock.release"
    RLOCK_LOCKED = "rlock.locked"
    SEMAPHORE_ACQUIRE = "semaphore.acquire"
    SEMAPHORE_TRY_ACQUIRE = "semaphore.try_acquire"
    SEMAPHORE_RELEASE = "semaphore.release"
    CONDITION_WAIT = "condition.wait"
    CONDITION_WAKE = "condition.wake"
    CONDITION_NOTIFY = "condition.notify"
    QUEUE_PUT = "queue.put"
    QUEUE_TRY_PUT = "queue.try_put"
    QUEUE_GET = "queue.get"
    QUEUE_TRY_GET = "queue.try_get"
    QUEUE_QSIZE = "queue.qsize"
    QUEUE_EMPTY = "queue.empty"
    QUEUE_FULL = "queue.full"
    QUEUE_TASK_DONE = "queue.task_done"
    QUEUE_JOIN = "queue.join"


class FailureCategory(StrEnum):
    """Distinguish task exceptions from modeled deadlocks."""

    EXCEPTION = auto()
    DEADLOCK = auto()


class ExecutionStatus(StrEnum):
    """Wire-compatible tags for the closed execution outcome union."""

    PASSED = auto()
    FAILED = auto()
    INCOMPLETE = auto()
    DIVERGED = auto()
    REJECTED = auto()


class ExplorationLimit(StrEnum):
    """The precise resource bound that prevented exhaustive exploration."""

    SCHEDULES = auto()
    STEPS = auto()
    TASKS = auto()
    WALL_TIME = auto()


class ReductionStatus(StrEnum):
    """Report bounded reduction independently from the original failure."""

    COMPLETED = auto()
    BUDGET_EXHAUSTED = auto()
    DISABLED = auto()


@dataclass(frozen=True)
class TaskCheckpoint:
    """Identify a task checkpoint without a resource target."""

    kind: TaskCheckpointKind

    @property
    def identity(self) -> str:
        """Keep trace identities stable across process boundaries."""
        return self.kind


@dataclass(frozen=True)
class PrimitiveOperation:
    """Identify a modeled operation on an execution-local task or resource."""

    kind: PrimitiveOperationKind
    target: int

    @property
    def identity(self) -> str:
        """Distinguish operations without serializing mutable program state."""
        return f"{self.kind}:{self.target}"


@dataclass(frozen=True)
class NamedCheckpoint:
    """Identify a selected event in a named task."""

    event: str
    occurrence: int = 1

    @property
    def identity(self) -> str:
        """Preserve label and occurrence in portable replay evidence."""
        return f"checkpoint:{self.event}:{self.occurrence}"


type Operation = TaskCheckpoint | PrimitiveOperation | NamedCheckpoint


@dataclass(frozen=True)
class Step:
    """Identify an executed checkpoint and the alternatives available there.

    Attributes:
        task: Execution-local identity of the selected task.
        operation: Stable logical operation identity, without program state.
        runnable: Sorted task identities the explorer could select at this point.
    """

    task: int
    operation: str
    runnable: tuple[int, ...]


@dataclass(frozen=True)
class Failure:
    """Preserve original exception evidence without serializing exception objects.

    Attributes:
        category: Task exception or modeled deadlock.
        exception_type: Qualified exception class name.
        message: Original diagnostic text.
        traceback: Original stack and process-specific diagnostic evidence.
        location: Portable source location used to identify the same failure.
    """

    category: FailureCategory
    exception_type: str
    message: str
    traceback: str
    location: str

    @property
    def signature(self) -> tuple[str, str, str]:
        """Identify a failure during reduction without depending on its values."""
        return self.category, self.exception_type, self.location


# Singleton enum annotations retain union narrowing and JSON discrimination.
@dataclass(frozen=True)
class Passed:
    """All modeled tasks completed in this execution."""

    kind: Literal[ExecutionStatus.PASSED] = ExecutionStatus.PASSED


@dataclass(frozen=True)
class Failed:
    """An assertion, task exception, or modeled deadlock was discovered."""

    failure: Failure
    kind: Literal[ExecutionStatus.FAILED] = ExecutionStatus.FAILED


@dataclass(frozen=True)
class Incomplete:
    """A resource limit prevented an execution or search from completing."""

    limit: ExplorationLimit
    kind: Literal[ExecutionStatus.INCOMPLETE] = ExecutionStatus.INCOMPLETE


@dataclass(frozen=True)
class Diverged:
    """An execution no longer matches its recorded replay evidence."""

    message: str
    kind: Literal[ExecutionStatus.DIVERGED] = ExecutionStatus.DIVERGED


@dataclass(frozen=True)
class Rejected:
    """This prefix cannot complete the requested partial execution order."""

    message: str
    kind: Literal[ExecutionStatus.REJECTED] = ExecutionStatus.REJECTED


type Outcome = Passed | Failed | Incomplete | Diverged | Rejected


@dataclass(frozen=True)
class Execution:
    """Keep logical operation order separate from the terminal outcome.

    Attributes:
        steps: Modeled operations in execution order.
        outcome: The closed passed, failed, incomplete, or diverged alternative.
        cleanup_failures: Additional failures while cancellation unwinds tasks.
    """

    steps: tuple[Step, ...]
    outcome: Outcome
    cleanup_failures: tuple[Failure, ...] = ()


@dataclass(frozen=True)
class Statistics:
    """Describe exploration work without mixing in the reduction budget.

    Attributes:
        executions: Number of isolated executions attempted.
        steps: Total observed modeled steps across those executions.
        elapsed: Exploration duration in monotonic seconds.
    """

    executions: int
    steps: int
    elapsed: float


def alternative_prefixes(
    steps: tuple[Step, ...], first_new_step: int = 0, max_preemptions: int | None = None
) -> tuple[tuple[int, ...], ...]:
    """Produce the unexplored scheduling prefixes that branch off ``steps``.

    A depth-first search passes the length of the forced prefix as
    ``first_new_step``. Ancestors of that prefix already produced every branch
    before it, so each schedule is produced exactly once without remembering
    visited prefixes.

    Extending a prefix never removes preemptions, and the default policy after a
    prefix keeps running the previous task, so it adds none. A preemption bound
    therefore prunes exactly the schedules that exceed it.

    Args:
        steps: The observed schedule whose branches should be explored.
        first_new_step: First step whose alternatives are new to the search.
        max_preemptions: Largest preemption count a produced prefix may have.

    Returns:
        Prefixes that select another runnable task at an observed checkpoint.

    Examples:
        >>> steps = (Step(0, "0:yield", (0, 1)), Step(0, "1:yield", (0, 1)))
        >>> alternative_prefixes(steps)
        ((1,), (0, 1))
        >>> alternative_prefixes(steps, first_new_step=1)
        ((0, 1),)
        >>> alternative_prefixes(steps, max_preemptions=0)
        ((1,),)
    """
    preemption_budget = math.inf if max_preemptions is None else max_preemptions
    branch_prefixes: list[tuple[int, ...]] = []
    chosen_tasks: list[int] = []
    used_preemptions = 0
    previous_task: int | None = None
    for step_index, step in enumerate(steps):
        previous_task_could_continue = previous_task in step.runnable
        if step_index >= first_new_step:
            branch_prefixes.extend(
                (*chosen_tasks, alternative_task)
                for alternative_task in step.runnable
                if alternative_task != step.task
                and used_preemptions
                + (previous_task_could_continue and alternative_task != previous_task)
                <= preemption_budget
            )
        used_preemptions += previous_task_could_continue and step.task != previous_task
        previous_task = step.task
        chosen_tasks.append(step.task)
    return tuple(branch_prefixes)


def count_preemptions(steps: tuple[Step, ...]) -> int:
    """Count switches away from a task that could have continued running.

    Args:
        steps: Observed scheduling choices in execution order.

    Returns:
        The number of task switches where the previous task remained runnable.

    Examples:
        >>> count_preemptions((Step(0, "0:begin", (0,)), Step(1, "0:begin", (0, 1))))
        1
        >>> count_preemptions(())
        0
    """
    return sum(
        current_step.task != previous_step.task and previous_step.task in current_step.runnable
        for previous_step, current_step in pairwise(steps)
    )


def validate_replay(
    execution: Execution, required_prefix: tuple[int, ...], expected_execution: Execution | None
) -> Execution:
    """Compare logical evidence while allowing process-specific diagnostic text.

    Args:
        execution: Evidence collected by the current execution.
        required_prefix: Required task choices, all of which must have been consumed.
        expected_execution: Original evidence, or None during exploration.

    Returns:
        Current evidence, or a divergence outcome explaining inconsistent replay.
    """
    match execution.outcome:
        case Diverged():
            return execution
        case _ if len(required_prefix) > len(execution.steps):
            return replace(execution, outcome=Diverged("execution ended before requested prefix"))
        case _ if expected_execution is not None and replay_identity(execution) != replay_identity(
            expected_execution
        ):
            return replace(execution, outcome=Diverged("terminal outcome changed"))
        case Passed() | Failed() | Incomplete() | Rejected():
            return execution
        case _ as unreachable:
            assert_never(unreachable)


def replay_identity(execution: Execution) -> tuple[object, ...]:
    """Keep the facts a replay must reproduce and drop process-specific text."""
    match execution.outcome:
        case Failed(failure):
            outcome_identity: object = failure.signature
        case Passed() | Incomplete() | Diverged() | Rejected() as outcome_identity:
            pass
        case _ as unreachable:
            assert_never(unreachable)
    cleanup_identity = tuple(failure.signature for failure in execution.cleanup_failures)
    return len(execution.steps), outcome_identity, cleanup_identity
