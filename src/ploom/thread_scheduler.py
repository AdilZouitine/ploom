"""Coordinate instrumented threads without depending on simultaneous execution.

The condition owns task state, checkpoint publication, and the control phase.
Only the task that holds control can execute user code. Every handoff passes
through this condition, providing synchronization on standard and free-threaded
CPython.
"""

from __future__ import annotations

import sysconfig
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum, auto
from itertools import count, filterfalse
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from ploom.execution import (
    Diverged,
    Execution,
    ExplorationLimit,
    Failed,
    Failure,
    FailureCategory,
    Incomplete,
    Operation,
    Outcome,
    Passed,
    Step,
    TaskCheckpoint,
    TaskCheckpointKind,
    validate_replay,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import FrameType, TracebackType

LIBRARY_DIRECTORIES = (
    Path(__file__).parent,
    *{Path(sysconfig.get_path(name)) for name in ("stdlib", "platstdlib", "purelib", "platlib")},
)
ROOT_TASK = 0


class Cancelled(BaseException):
    """Unwind modeled tasks when their owning execution has stopped."""


class TaskLimitError(BaseException):
    """Signal a task bound that scenario ``except Exception`` blocks cannot hide."""


class TaskState(StrEnum):
    """Lifecycle of one modeled task."""

    UNSTARTED = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class Choosing:
    """The coordinator holds control and selects the next operation."""


@dataclass(frozen=True)
class Granted:
    """One task holds control until its next checkpoint or its exit."""

    task: int


@dataclass(frozen=True)
class Cancelling:
    """The execution has stopped; only ``task``, if any, may unwind."""

    task: int | None = None


type Phase = Choosing | Granted | Cancelling


@dataclass(frozen=True)
class PendingOperation:
    """An operation that a suspended task published and waits to perform.

    Attributes:
        operation: The modeled operation.
        task_operation_index: How many operations the task published before it.
        can_proceed: Whether the operation can run now, such as a free lock.
    """

    operation: Operation
    task_operation_index: int
    can_proceed: Callable[[], bool]

    @property
    def step_operation(self) -> str:
        """Distinguish repeated operations of one task in recorded steps."""
        return f"{self.task_operation_index}:{self.operation.identity}"


@dataclass(frozen=True)
class TaskContext:
    """Bind a task to its thread for the task's whole lifetime.

    The scope ends at the thread boundary, so it absorbs every exception and
    hands it to the scheduler as evidence.
    """

    scheduler: ThreadScheduler
    task: int

    def __enter__(self) -> None:
        """Make primitives on this thread act for this task."""
        _task_local.context = self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        """Unbind the task, then publish its failure, if any, and its exit."""
        _task_local.context = None
        self.scheduler.finish_task(self.task, error)
        return True


class TaskContextStorage(threading.local):
    """Scope the threading-style facade to its current modeled task.

    This is the public facade's sole ambient dependency. Internal engine objects
    receive their runtime explicitly; task context is installed and cleared at
    the worker entry boundary.
    """

    context: TaskContext | None = None


_task_local = TaskContextStorage()


def current_task_context() -> TaskContext:
    """Require primitives to belong to an active modeled execution."""
    context = _task_local.context
    if context is None:
        raise RuntimeError("ploom primitives must be used inside ploom.check or ploom.replay")
    return context


def always_ready() -> bool:
    """Mark operations that cannot block on a modeled resource."""
    return True


def is_library_frame(frame_and_line: tuple[FrameType, int]) -> bool:
    """Whether a traceback entry belongs to ploom, the standard library, or a dependency."""
    source_file = Path(frame_and_line[0].f_code.co_filename)
    return any(source_file.is_relative_to(directory) for directory in LIBRARY_DIRECTORIES)


def failure_location(error: BaseException) -> str:
    """Name the innermost scenario frame, skipping ploom and library internals.

    Minimization accepts a schedule only if its failure has the same location.
    The innermost frame of an error raised inside ``json`` or ploom is shared by
    unrelated call sites, so it would merge distinct failures.

    Args:
        error: An exception raised by a modeled task.

    Returns:
        A portable ``module:line:qualname`` location.
    """
    frames = tuple(traceback.walk_tb(error.__traceback__))
    frame, line_number = next(filterfalse(is_library_frame, reversed(frames)), frames[-1])
    module = frame.f_globals.get("__name__", Path(frame.f_code.co_filename).name)
    return f"{module}:{line_number}:{frame.f_code.co_qualname}"


def failure_from_exception(error: BaseException) -> Failure:
    """Preserve a task exception as portable evidence without the live object."""
    return Failure(
        FailureCategory.EXCEPTION,
        f"{type(error).__module__}.{type(error).__qualname__}",
        str(error),
        "".join(traceback.format_exception(error)),
        failure_location(error),
    )


class ThreadScheduler:
    """Own one execution's threads and grant control to exactly one of them.

    Args:
        max_steps: Maximum modeled operations before the execution is incomplete.
        max_tasks: Maximum concurrently running tasks, including the root task.
        join_timeout: Seconds to wait for each cancelled task, or None to wait
            until a supervising process enforces its deadline.
        partial_order_reduction: Require resources and tasks to be allocated before
            any child starts, preserving identities across commuting transitions.
        start_native_thread: Starts a native thread; injectable to test failure.
    """

    def __init__(
        self,
        max_steps: int,
        max_tasks: int,
        *,
        join_timeout: float | None = None,
        partial_order_reduction: bool = False,
        start_native_thread: Callable[[threading.Thread], None] = threading.Thread.start,
    ) -> None:
        self.coordination = threading.Condition()
        self.task_states: dict[int, TaskState] = {}
        self.native_threads: dict[int, threading.Thread] = {}
        self.pending_operations: dict[int, PendingOperation] = {}
        self.published_operation_counts: Counter[int] = Counter()
        self.recorded_steps: list[Step] = []
        self.phase: Phase = Choosing()
        self.terminal_outcome: Outcome | None = None
        self.cleanup_failures: list[Failure] = []
        self.start_native_thread = start_native_thread
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        self.join_timeout = join_timeout
        self.resource_identity_counter = count()
        self.partial_order_reduction = partial_order_reduction

    def allocate_resource(self) -> int:
        """Assign a stable execution-local identity while one task owns control."""
        self.require_initial_allocation()
        return next(self.resource_identity_counter)

    def require_initial_allocation(self) -> None:
        """Reject allocation after a child starts when reduction is enabled."""
        if self.partial_order_reduction and any(task != ROOT_TASK for task in self.native_threads):
            raise NotImplementedError(
                "partial-order reduction requires all resources and threads "
                "to be constructed before starting any child thread"
            )

    @property
    def cancelling(self) -> bool:
        """Whether the execution has stopped and every task must unwind."""
        match self.phase:
            case Cancelling():
                return True
            case Choosing() | Granted():
                return False
            case _ as unreachable:
                assert_never(unreachable)

    def create_task(self) -> int:
        """Register an unstarted task without acquiring an OS thread."""
        with self.coordination:
            self.require_initial_allocation()
            task = len(self.task_states)
            self.task_states[task] = TaskState.UNSTARTED
            return task

    def start_task(self, task: int, target: Callable[[], object]) -> None:
        """Start a task and wait until its first checkpoint is published."""
        with self.coordination:
            if self.task_states[task] is not TaskState.UNSTARTED:
                raise RuntimeError("threads can only be started once")
            running_tasks = sum(state is TaskState.RUNNING for state in self.task_states.values())
            if running_tasks >= self.max_tasks:
                if self.terminal_outcome is None:
                    self.terminal_outcome = Incomplete(ExplorationLimit.TASKS)
                raise TaskLimitError
            native_thread = threading.Thread(
                target=self.run_task, args=(task, target), name=f"ploom-{task}"
            )
            # The new thread blocks on the held condition before it reads task state.
            self.start_native_thread(native_thread)
            self.task_states[task] = TaskState.RUNNING
            self.native_threads[task] = native_thread
            self.coordination.wait_for(lambda: task in self.pending_operations or self.cancelling)

    def task_state(self, task: int) -> TaskState:
        """Inspect lifecycle while the coordinator or active task owns control."""
        return self.task_states[task]

    def checkpoint(
        self, operation: Operation, can_proceed: Callable[[], bool] = always_ready
    ) -> None:
        """Publish an operation and suspend until the coordinator selects it."""
        context = current_task_context()
        if context.scheduler is not self:
            raise RuntimeError("primitive belongs to a different execution")
        task = context.task
        with self.coordination:
            if self.cancelling:
                raise Cancelled
            task_operation_index = self.published_operation_counts[task]
            self.published_operation_counts[task] += 1
            self.pending_operations[task] = PendingOperation(
                operation, task_operation_index, can_proceed
            )
            if self.phase == Granted(task):
                self.phase = Choosing()
            self.coordination.notify_all()
            self.coordination.wait_for(lambda: self.phase in (Granted(task), Cancelling(task)))
            if self.phase == Cancelling(task):
                raise Cancelled

    def run_task(self, task: int, target: Callable[[], object]) -> None:
        """Run one task's code inside the scope that publishes its outcome."""
        with TaskContext(self, task):
            self.checkpoint(TaskCheckpoint(TaskCheckpointKind.BEGIN))
            target()

    def finish_task(self, task: int, error: BaseException | None) -> None:
        """Keep the first task failure as the outcome, then release control."""
        with self.coordination:
            match error:
                case None | Cancelled() | TaskLimitError():
                    pass
                case BaseException() if self.terminal_outcome is None:
                    self.terminal_outcome = Failed(failure_from_exception(error))
                case BaseException():
                    self.cleanup_failures.append(failure_from_exception(error))
                case _ as unreachable:
                    assert_never(unreachable)
            self.task_states[task] = TaskState.FINISHED
            self.pending_operations.pop(task, None)
            if self.phase == Granted(task):
                self.phase = Choosing()
            self.coordination.notify_all()

    def __enter__(self) -> ThreadScheduler:
        """Acquire a single-use execution scope before starting any workers."""
        if self.task_states or self.cancelling:
            raise RuntimeError("a scheduler can execute only once")
        return self

    def __exit__(self, *_exc: object) -> None:
        """Cancel waiters and join workers even when execution exits early."""
        self.close()

    def deadlock_outcome(self) -> Failed:
        """Describe which task waits on which operation when none can proceed."""
        blocked_tasks = ", ".join(
            f"task {task} at {pending.operation.identity}"
            for task, pending in sorted(self.pending_operations.items())
        )
        return Failed(
            Failure(
                FailureCategory.DEADLOCK,
                "ploom.DeadlockError",
                f"no modeled task can proceed: {blocked_tasks}",
                "",
                f"deadlock: {blocked_tasks}",
            )
        )

    def choose_next_step(
        self, required_prefix: tuple[int, ...], expected_execution: Execution | None
    ) -> Step | Outcome:
        """Return the next step, or the outcome that ends the execution."""
        if self.terminal_outcome is not None:
            return self.terminal_outcome
        if not self.pending_operations:
            return Passed()
        runnable_tasks = tuple(
            task
            for task, pending in sorted(self.pending_operations.items())
            if pending.can_proceed()
        )
        if not runnable_tasks:
            return self.deadlock_outcome()
        if len(self.recorded_steps) >= self.max_steps:
            return Incomplete(ExplorationLimit.STEPS)
        return self.select_step(runnable_tasks, required_prefix, expected_execution)

    def select_step(
        self,
        runnable_tasks: tuple[int, ...],
        required_prefix: tuple[int, ...],
        expected_execution: Execution | None,
    ) -> Step | Diverged:
        """Pick the required task, or else keep the current task while it can proceed.

        The default keeps control with the current task, as in Loom, so the
        default schedule has no preemptions. Replay must also see the recorded step.
        """
        step_index = len(self.recorded_steps)
        previous_task = self.recorded_steps[-1].task if self.recorded_steps else ROOT_TASK
        default_task = previous_task if previous_task in runnable_tasks else runnable_tasks[0]
        chosen_task = (
            required_prefix[step_index] if step_index < len(required_prefix) else default_task
        )
        if chosen_task not in runnable_tasks:
            return Diverged(f"task {chosen_task} is not runnable at step {step_index}")
        step = Step(
            chosen_task, self.pending_operations[chosen_task].step_operation, runnable_tasks
        )
        if expected_execution is not None and (
            step_index >= len(expected_execution.steps)
            or expected_execution.steps[step_index] != step
        ):
            return Diverged(f"operation or runnable tasks changed at step {step_index}")
        return step

    def schedule(
        self, required_prefix: tuple[int, ...], expected_execution: Execution | None
    ) -> Outcome:
        """Grant control one step at a time until the execution has an outcome."""
        with self.coordination:
            while True:
                self.coordination.wait_for(lambda: self.phase == Choosing())
                match self.choose_next_step(required_prefix, expected_execution):
                    case Step(task=chosen_task) as step:
                        self.recorded_steps.append(step)
                        del self.pending_operations[chosen_task]
                        self.phase = Granted(chosen_task)
                        self.coordination.notify_all()
                    case Passed() | Failed() | Incomplete() | Diverged() as outcome:
                        self.terminal_outcome = outcome
                        return outcome
                    case _ as unreachable:
                        assert_never(unreachable)

    def execute(
        self,
        scenario: Callable[[], object],
        required_prefix: tuple[int, ...],
        expected_execution: Execution | None,
    ) -> Execution:
        """Run a scenario as the root task and collect its execution evidence."""
        with self:
            self.start_task(self.create_task(), scenario)
            outcome = self.schedule(required_prefix, expected_execution)
        execution = Execution(tuple(self.recorded_steps), outcome, tuple(self.cleanup_failures))
        return validate_replay(execution, required_prefix, expected_execution)

    def close(self) -> None:
        """Cancel tasks one at a time and join their threads before returning.

        Only the execution owner calls this, after scheduling has stopped.

        Raises:
            RuntimeError: If a task does not stop within ``join_timeout``. The note
                keeps the outcome recorded before cancellation.
        """
        with self.coordination:
            self.phase = Cancelling()
            started_threads = list(self.native_threads.items())
        stalled_tasks: list[int] = []
        for task, native_thread in started_threads:
            with self.coordination:
                self.phase = Cancelling(task)
                self.coordination.notify_all()
            native_thread.join(self.join_timeout)
            if native_thread.is_alive():
                stalled_tasks.append(task)
        if not stalled_tasks:
            return
        error = RuntimeError(
            f"modeled workers did not stop: {stalled_tasks}; supervisor must terminate execution"
        )
        with self.coordination:
            if self.terminal_outcome is not None:
                error.add_note(f"Outcome before cancellation: {self.terminal_outcome}")
        raise error
