"""Untimed synchronization with sequentially consistent modeled operations.

Only the scheduled task mutates these resources. Readiness predicates inspect
state while the coordinator owns control; they never acquire native locks.
"""

from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from queue import Empty, Full
from typing import override

from ploom.execution import PrimitiveOperationKind
from ploom.primitives import Immutable, Lock, allocate_resource, validate_immutable
from ploom.thread_scheduler import current_task_context


@dataclass(eq=False)
class Waiter:
    """Remember a notification independently of later resource state changes."""

    notified: bool = False

    def ready(self) -> bool:
        """Whether this registered wait has received a notification."""
        return self.notified


@contextmanager
def registered_waiter(waiters: deque[Waiter]) -> Generator[Waiter]:
    """Register a waiter for one wait, and unregister it when the wait ends.

    A notifier may already have removed the waiter, so the cleanup tolerates that.
    """
    waiter = Waiter()
    waiters.append(waiter)
    try:
        yield waiter
    finally:
        if waiter in waiters:
            waiters.remove(waiter)


class Event:
    """Model an untimed event whose set wakes all currently registered waiters.

    Clearing the flag does not revoke a wake already delivered to a waiter.
    Timed waits are rejected; wait returns True after observing a set flag or
    receiving a notification. Construct the event inside a scenario.
    """

    def __init__(self) -> None:
        self._resource = allocate_resource()
        self._flag = False
        self._waiters: deque[Waiter] = deque()

    def is_set(self) -> bool:
        """Observe the flag at a modeled scheduling point."""
        self._resource.checkpoint(PrimitiveOperationKind.EVENT_IS_SET)
        return self._flag

    def set(self) -> None:
        """Set the flag and latch a notification for every current waiter."""
        self._resource.checkpoint(PrimitiveOperationKind.EVENT_SET)
        self._flag = True
        for waiter in self._waiters:
            waiter.notified = True

    def clear(self) -> None:
        """Clear the flag without withdrawing notifications already delivered."""
        self._resource.checkpoint(PrimitiveOperationKind.EVENT_CLEAR)
        self._flag = False

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for a set flag or notification; only timeout=None is supported."""
        if timeout is not None:
            raise NotImplementedError("timed event waits are not modeled")
        self._resource.checkpoint(PrimitiveOperationKind.EVENT_WAIT)
        if self._flag:
            return True
        with registered_waiter(self._waiters) as waiter:
            self._resource.checkpoint(PrimitiveOperationKind.EVENT_WAIT_READY, waiter.ready)
        return True


class RLock:
    """Model a reentrant mutex owned by a single modeled task.

    Only its owner may release it. Condition waits can release every recursion
    level and restore the original depth. Timed acquisition is unsupported.
    """

    def __init__(self) -> None:
        self._resource = allocate_resource()
        self._owner: int | None = None
        self._depth = 0

    def _can_acquire(self, task: int) -> bool:
        return self._owner is None or self._owner == task

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire or recurse; a nonblocking attempt can return False."""
        if timeout != -1:
            raise NotImplementedError("timed rlock acquisition is not modeled")
        task = current_task_context().task
        self._resource.checkpoint_attempt(
            blocking,
            PrimitiveOperationKind.RLOCK_ACQUIRE,
            PrimitiveOperationKind.RLOCK_TRY_ACQUIRE,
            partial(self._can_acquire, task),
        )
        if not self._can_acquire(task):
            return False
        self._owner = task
        self._depth += 1
        return True

    def release(self) -> None:
        """Release one recursion level, rejecting a task that is not the owner."""
        self._resource.checkpoint(PrimitiveOperationKind.RLOCK_RELEASE)
        if self._owner != current_task_context().task:
            raise RuntimeError("cannot release un-acquired rlock")
        self._depth -= 1
        if not self._depth:
            self._owner = None

    def locked(self) -> bool:
        """Observe whether any task owns the lock."""
        self._resource.checkpoint(PrimitiveOperationKind.RLOCK_LOCKED)
        return self._owner is not None

    # Condition uses the same lock protocol as threading.Condition.
    def _is_owned(self) -> bool:
        return self._owner == current_task_context().task

    def _release_save(self) -> int:
        saved_depth = self._depth
        self._owner = None
        self._depth = 0
        return saved_depth

    def _acquire_restore(self, saved_depth: int) -> None:
        self.acquire()
        self._depth = saved_depth

    def __enter__(self) -> bool:
        """Acquire a recursion level for a context manager."""
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        """Release the context manager's recursion level."""
        self.release()


class Semaphore:
    """Model an untimed counting semaphore without imposing waiter fairness.

    Releases add permits atomically. Every blocked acquirer is a scheduling
    alternative whenever a permit is available. Any task may release permits.
    """

    def __init__(self, value: int = 1) -> None:
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._resource = allocate_resource()
        self._value = value

    def _can_acquire(self) -> bool:
        return self._value > 0

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Consume a permit, optionally returning False instead of blocking."""
        if timeout is not None:
            raise NotImplementedError("timed semaphore acquisition is not modeled")
        self._resource.checkpoint_attempt(
            blocking,
            PrimitiveOperationKind.SEMAPHORE_ACQUIRE,
            PrimitiveOperationKind.SEMAPHORE_TRY_ACQUIRE,
            self._can_acquire,
        )
        if not self._can_acquire():
            return False
        self._value -= 1
        return True

    def release(self, n: int = 1) -> None:
        """Publish n permits; n must be a positive integer."""
        if n < 1:
            raise ValueError("n must be a positive integer")
        self._resource.checkpoint(PrimitiveOperationKind.SEMAPHORE_RELEASE)
        self._validate_release(n)
        self._value += n

    def _validate_release(self, _n: int, /) -> None:
        """Accept an unbounded semaphore release."""

    def __enter__(self) -> bool:
        """Consume a permit for the context manager."""
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        """Return the context manager's permit."""
        self.release()


class BoundedSemaphore(Semaphore):
    """Reject releases that would exceed the initial permit count."""

    def __init__(self, value: int = 1) -> None:
        super().__init__(value)
        self._initial_value = value

    @override
    def _validate_release(self, n: int, /) -> None:
        if self._value + n > self._initial_value:
            raise ValueError("semaphore released too many times")


class Condition:
    """Model untimed condition waits with FIFO notification selection.

    The default lock is an RLock. A modeled Lock or RLock may be supplied.
    Waiting atomically registers the waiter and fully releases the lock, then
    reacquires it after notification, restoring recursion depth. notify(n)
    selects the oldest n registered waiters; other Python-permitted selection
    policies and spurious wakeups are outside this model. Wake/reacquire order
    is explored. Cancellation unregisters the waiter without reacquiring a lock
    in an execution that is already stopping.
    """

    def __init__(self, lock: Lock | RLock | None = None) -> None:
        self._resource = allocate_resource()
        self._lock = RLock() if lock is None else lock
        if self._lock._resource.scheduler is not self._resource.scheduler:
            raise RuntimeError("condition lock belongs to a different execution")
        self._waiters: deque[Waiter] = deque()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """Acquire the associated lock with its normal recursion semantics."""
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        """Release one level of the associated lock."""
        self._lock.release()

    def wait(self, timeout: float | None = None) -> bool:
        """Release the owned lock, wait for notification, then reacquire it."""
        if timeout is not None:
            raise NotImplementedError("timed condition waits are not modeled")
        self._resource.checkpoint(PrimitiveOperationKind.CONDITION_WAIT)
        if not self._lock._is_owned():
            raise RuntimeError("cannot wait on un-acquired lock")
        with registered_waiter(self._waiters) as waiter:
            saved_depth = self._lock._release_save()
            self._resource.checkpoint(PrimitiveOperationKind.CONDITION_WAKE, waiter.ready)
        self._lock._acquire_restore(saved_depth)
        return True

    def wait_for[T](self, predicate: Callable[[], T], timeout: float | None = None) -> T:
        """Recheck a predicate under the lock after each untimed notification."""
        if timeout is not None:
            raise NotImplementedError("timed condition waits are not modeled")
        result = predicate()
        while not result:
            self.wait()
            result = predicate()
        return result

    def notify(self, n: int = 1) -> None:
        """Notify up to n oldest waiters while holding the associated lock."""
        self._notify(n)

    def notify_all(self) -> None:
        """Notify all waiters registered when this operation executes."""
        self._notify(None)

    def _notify(self, n: int | None) -> None:
        self._resource.checkpoint(PrimitiveOperationKind.CONDITION_NOTIFY)
        if not self._lock._is_owned():
            raise RuntimeError("cannot notify on un-acquired lock")
        waiter_count = len(self._waiters)
        notified_count = waiter_count if n is None else min(max(n, 0), waiter_count)
        for _ in range(notified_count):
            self._waiters.popleft().notified = True

    def __enter__(self) -> bool:
        """Acquire the associated lock."""
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        """Release the associated lock."""
        self.release()


class Queue[T: Immutable]:
    """Model an untimed FIFO queue of deeply immutable values.

    maxsize <= 0 means unbounded. A successful put increments unfinished work;
    task_done acknowledges work and join waits for all acknowledgements. Empty
    and Full are the standard queue exceptions. Shutdown and timed waits are
    unsupported. Size/empty/full observations do not reserve an item or slot.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._resource = allocate_resource()
        self._maxsize = maxsize
        self._items: deque[T] = deque()
        self._unfinished_tasks = 0

    def _can_put(self) -> bool:
        return self._maxsize <= 0 or len(self._items) < self._maxsize

    def _can_get(self) -> bool:
        return bool(self._items)

    def _has_no_unfinished_tasks(self) -> bool:
        return not self._unfinished_tasks

    def put(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Enqueue an immutable value or raise Full for a nonblocking attempt."""
        if timeout is not None:
            raise NotImplementedError("timed queue operations are not modeled")
        validate_immutable(item)
        self._resource.checkpoint_attempt(
            block,
            PrimitiveOperationKind.QUEUE_PUT,
            PrimitiveOperationKind.QUEUE_TRY_PUT,
            self._can_put,
        )
        if not self._can_put():
            raise Full
        self._items.append(item)
        self._unfinished_tasks += 1

    def put_nowait(self, item: T) -> None:
        """Enqueue without waiting for capacity."""
        self.put(item, block=False)

    def get(self, block: bool = True, timeout: float | None = None) -> T:
        """Dequeue the oldest item or raise Empty for a nonblocking attempt."""
        if timeout is not None:
            raise NotImplementedError("timed queue operations are not modeled")
        self._resource.checkpoint_attempt(
            block,
            PrimitiveOperationKind.QUEUE_GET,
            PrimitiveOperationKind.QUEUE_TRY_GET,
            self._can_get,
        )
        if not self._can_get():
            raise Empty
        return self._items.popleft()

    def get_nowait(self) -> T:
        """Dequeue without waiting for an item."""
        return self.get(block=False)

    def qsize(self) -> int:
        """Observe the current item count without reserving any item."""
        self._resource.checkpoint(PrimitiveOperationKind.QUEUE_QSIZE)
        return len(self._items)

    def empty(self) -> bool:
        """Observe whether the queue currently has no items."""
        self._resource.checkpoint(PrimitiveOperationKind.QUEUE_EMPTY)
        return not self._items

    def full(self) -> bool:
        """Observe whether a bounded queue currently has no free slot."""
        self._resource.checkpoint(PrimitiveOperationKind.QUEUE_FULL)
        return not self._can_put()

    def task_done(self) -> None:
        """Acknowledge one enqueued item, rejecting excess acknowledgements."""
        self._resource.checkpoint(PrimitiveOperationKind.QUEUE_TASK_DONE)
        if not self._unfinished_tasks:
            raise ValueError("task_done() called too many times")
        self._unfinished_tasks -= 1

    def join(self) -> None:
        """Wait for every enqueued item to be acknowledged."""
        self._resource.checkpoint(PrimitiveOperationKind.QUEUE_JOIN, self._has_no_unfinished_tasks)
