"""Threading-style facade over explicitly modeled scheduling points."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import NoneType

from ploom.execution import (
    PrimitiveOperation,
    PrimitiveOperationKind,
    TaskCheckpoint,
    TaskCheckpointKind,
)
from ploom.thread_scheduler import TaskState, ThreadScheduler, always_ready, current_task_context

type Immutable = None | bool | int | float | complex | str | bytes | tuple[Immutable, ...]

IMMUTABLE_SCALAR_TYPES = frozenset({NoneType, bool, int, float, complex, str, bytes})


def is_deeply_immutable(value: object) -> bool:
    """Whether a value is an exact built-in scalar or a tuple of such values.

    Exact types are required because a subclass can add mutable attributes.

    Examples:
        >>> is_deeply_immutable((1, "ready", (None, True)))
        True
        >>> is_deeply_immutable((1, []))
        False
    """
    return type(value) in IMMUTABLE_SCALAR_TYPES or (
        type(value) is tuple and all(map(is_deeply_immutable, value))
    )


def validate_immutable(value: object) -> None:
    """Reject aliases that could mutate shared state outside modeled operations.

    Args:
        value: A value entering a shared cell at construction or write time.

    Raises:
        TypeError: If the value is not an exact built-in scalar or immutable tuple.
    """
    if not is_deeply_immutable(value):
        raise TypeError("SharedCell accepts only built-in scalars and deeply immutable tuples")


@dataclass(frozen=True)
class ModeledResource:
    """One execution-local resource and the scheduler that orders its operations.

    Attributes:
        scheduler: The scheduler of the execution that owns the resource.
        identity: Execution-local identity; for a thread, its task identity.
    """

    scheduler: ThreadScheduler
    identity: int

    def checkpoint(
        self, kind: PrimitiveOperationKind, can_proceed: Callable[[], bool] = always_ready
    ) -> None:
        """Wait until the scheduler selects this operation on this resource."""
        self.scheduler.checkpoint(PrimitiveOperation(kind, self.identity), can_proceed)


def allocate_resource() -> ModeledResource:
    """Register a new resource with the current task's execution."""
    scheduler = current_task_context().scheduler
    return ModeledResource(scheduler, scheduler.allocate_resource())


class SharedCell[T: Immutable]:
    """Expose sequentially consistent reads and writes as scheduling points.

    Construct the cell inside a scenario. Its scheduler owns all accesses and its
    values must remain deeply immutable between modeled writes. The ``Immutable``
    bound states this to type checkers; construction and writes enforce it at
    runtime, including subclasses that the bound cannot exclude.

    Args:
        value: Initial built-in scalar or recursively immutable tuple.

    Raises:
        TypeError: If the initial value permits uninstrumented mutation.
        RuntimeError: If construction occurs outside a modeled execution.

    Examples:
        >>> from ploom import check
        >>> def scenario():
        ...     counter = SharedCell(0)
        ...     counter.write(counter.read() + 1)
        ...     assert counter.read() == 1
        >>> check(scenario).executions
        1
    """

    def __init__(self, value: T) -> None:
        validate_immutable(value)
        self._value = value
        self._resource = allocate_resource()

    def read(self) -> T:
        """Read atomically after allowing other modeled tasks to execute.

        Returns:
            The current immutable value without copying or revalidating it.
        """
        self._resource.checkpoint(PrimitiveOperationKind.CELL_READ)
        return self._value

    def write(self, value: T) -> None:
        """Validate a new value before publishing one atomic modeled write.

        Args:
            value: Replacement immutable value.

        Raises:
            TypeError: If the replacement permits uninstrumented mutation.
        """
        validate_immutable(value)
        self._resource.checkpoint(PrimitiveOperationKind.CELL_WRITE)
        self._value = value


class Lock:
    """Model a non-reentrant mutex without blocking on a native lock.

    Construct locks inside the scenario. Any modeled task may release a held lock,
    matching Python's basic mutex semantics.

    Examples:
        >>> from ploom import check
        >>> def scenario():
        ...     lock = Lock()
        ...     with lock:
        ...         assert lock.locked()
        ...         assert not lock.acquire(blocking=False)
        ...     assert not lock.locked()
        >>> check(scenario).executions
        1
    """

    def __init__(self) -> None:
        self._resource = allocate_resource()
        self._locked = False

    def _can_acquire(self) -> bool:
        return not self._locked

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire atomically, waiting only at a modeled scheduling point.

        Args:
            blocking: Whether to wait until the lock becomes available.
            timeout: Only the standard untimed sentinel, -1, is supported.

        Returns:
            True if acquired, or False for an unsuccessful nonblocking attempt.

        Raises:
            NotImplementedError: If a timed acquisition is requested.
        """
        if timeout != -1:
            raise NotImplementedError("timed lock acquisition is not modeled")
        if blocking:
            self._resource.checkpoint(PrimitiveOperationKind.LOCK_ACQUIRE, self._can_acquire)
            self._locked = True
            return True
        self._resource.checkpoint(PrimitiveOperationKind.LOCK_TRY_ACQUIRE)
        acquired = not self._locked
        self._locked = True
        return acquired

    def release(self) -> None:
        """Release a held lock from any modeled task.

        Raises:
            RuntimeError: If the lock is already unlocked.
        """
        self._resource.checkpoint(PrimitiveOperationKind.LOCK_RELEASE)
        if not self._locked:
            raise RuntimeError("release unlocked lock")
        self._locked = False

    def locked(self) -> bool:
        """Observe lock state at a modeled scheduling point.

        Returns:
            Whether the lock is currently held.
        """
        self._resource.checkpoint(PrimitiveOperationKind.LOCK_LOCKED)
        return self._locked

    def __enter__(self) -> bool:
        """Acquire before entering a modeled critical section."""
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        """Release on both normal completion and exceptions."""
        self.release()


def idle(*_args: object, **_kwargs: object) -> None:
    """Do nothing, like ``threading.Thread`` without a target."""


class Thread:
    """Run an ordinary callable under the deterministic thread scheduler.

    Construction registers a task. Starting acquires its native execution context;
    joining waits for modeled completion. Each thread may start only once.

    Args:
        group: Reserved for compatibility; only None is supported.
        target: Callable to execute, or None for an empty task.
        name: Optional descriptive name.
        args: Positional arguments bound to the target at construction.
        kwargs: Keyword arguments copied and bound to the target.
        daemon: Only None and False are supported.

    Raises:
        NotImplementedError: If daemon mode or a thread group is requested.
        RuntimeError: If construction occurs outside a modeled execution.

    Examples:
        >>> from ploom import check
        >>> def scenario():
        ...     worker = Thread()
        ...     worker.start()
        ...     worker.join()
        ...     assert not worker.is_alive()
        >>> check(scenario).executions
        1
    """

    def __init__(  # noqa: PLR0913 -- match the core threading.Thread constructor
        self,
        group: object | None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: tuple[object, ...] = (),
        kwargs: Mapping[str, object] | None = None,
        *,
        daemon: bool | None = None,
    ) -> None:
        if group is not None:
            raise NotImplementedError("thread groups are not modeled")
        if daemon is True:
            raise NotImplementedError("daemon threads are not modeled")
        self.name = name
        self._bound_target = partial(idle if target is None else target, *args, **(kwargs or {}))
        scheduler = current_task_context().scheduler
        self._task = ModeledResource(scheduler, scheduler.create_task())

    def _task_state(self) -> TaskState:
        return self._task.scheduler.task_state(self._task.identity)

    def _finished(self) -> bool:
        return self._task_state() is TaskState.FINISHED

    def start(self) -> None:
        """Start once, making the new task available to the explorer.

        Raises:
            RuntimeError: If the thread has already started.
        """
        self._task.checkpoint(PrimitiveOperationKind.THREAD_START)
        self._task.scheduler.start_task(self._task.identity, self._bound_target)

    def join(self, timeout: float | None = None) -> None:
        """Wait for modeled completion without suppressing child-task failures.

        Args:
            timeout: Only None is supported; waits are bounded by exploration.

        Raises:
            RuntimeError: If joining oneself or an unstarted thread.
            NotImplementedError: If a timed join is requested.
        """
        if timeout is not None:
            raise NotImplementedError("timed thread joins are not modeled")
        if current_task_context().task == self._task.identity:
            raise RuntimeError("cannot join current thread")
        if self._task_state() is TaskState.UNSTARTED:
            raise RuntimeError("cannot join thread before it is started")
        self._task.checkpoint(PrimitiveOperationKind.THREAD_JOIN, self._finished)

    def is_alive(self) -> bool:
        """Observe task lifecycle at a modeled scheduling point.

        Returns:
            Whether the task has started and has yet to finish.
        """
        self._task.checkpoint(PrimitiveOperationKind.THREAD_IS_ALIVE)
        return self._task_state() is TaskState.RUNNING


def yield_now() -> None:
    """Expose a scheduling choice without reading or mutating modeled state.

    Raises:
        RuntimeError: If called outside a modeled execution.

    Examples:
        >>> from ploom import check
        >>> check(yield_now).executions
        1
    """
    current_task_context().scheduler.checkpoint(TaskCheckpoint(TaskCheckpointKind.YIELD))
