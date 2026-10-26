# Supported operations and guarantees

Ploom explores a sequentially consistent model of **ploom operations**. Ordinary
Python between checkpoints runs uninterrupted. Exactly one modeled task executes
user code at a time, including ordered cancellation cleanup; simultaneous execution
and a disabled GIL are not required.

## Primitives

| Primitive | Supported behavior |
| --- | --- |
| `Thread` | Target arguments, optional name, `start`, `join`, `is_alive` |
| `Lock` | Blocking/nonblocking acquisition, release, status inspection, context manager |
| `RLock` | Owner-only release, recursion, blocking/nonblocking acquisition, status, context manager |
| `Event` | `set`, `clear`, `is_set`, untimed `wait`; notifications survive a subsequent clear |
| `Semaphore`, `BoundedSemaphore` | Untimed blocking/nonblocking acquisition, `release(n)`, context manager |
| `Condition` | Default `RLock` or supplied modeled lock, untimed `wait`/`wait_for`, FIFO `notify(n)`, `notify_all` |
| `Queue[T]` | FIFO immutable values, bounded/unbounded capacity, blocking/nonblocking put/get, observations, `task_done`/`join` |
| `SharedCell[T]` | Atomic modeled reads and writes |
| `yield_now()` | A scheduling choice without a shared-state operation |

Thread groups, daemon mode, and timed waits are rejected. Threads may start only
once. Joining an unstarted task or oneself is an error. Started tasks finish before
an execution succeeds; unstarted tasks own no native thread.

Locks are non-reentrant and any task may release a held lock, matching Python's
basic mutex. Lock poisoning is not modeled: a context manager releases its lock
on an exception, and the exception fails the execution.

Shared cells accept exact built-in `None`, `bool`, `int`, `float`, `complex`, `str`,
and `bytes` values, plus recursively immutable tuples. Mutable objects, subclasses,
and custom instances are rejected on construction and writes. Reads do not repeat
that validation.

## Exploration

The explorer traverses every runnable-task choice until exhaustion, the first
failure, or a resource limit. Exhaustive success applies to the specified finite
scenario and supported modeled operations. With `max_preemptions` set, success
applies only to schedules within that bound. There is no fairness guarantee or
partial-order reduction in this version. A spinning task can exhaust its step or
time budget instead of giving another task a turn.

The search requires a deterministic scenario: the same prefix must produce the
same operations. Every worker runs with the configured `PYTHONHASHSEED`, so
`str` hashing and set order are the same in each process. Other sources of
nondeterminism, such as time, randomness, or external state, are the scenario's
responsibility. When ploom observes them, it raises `ReplayDivergenceError`.

The native condition owns task lifecycle, checkpoint publication, and the control
phase. Control is with the coordinator, with one task, or with one task that is
unwinding during cancellation. Correctness does not rely on the GIL. Cancellation
unwinds one task at a time, joins all owned workers, and preserves additional
cleanup exceptions. Cleanup can take as long as it needs. A parent process
enforces hard execution deadlines and kills stalled workers.

## Side effects

Each execution starts in a fresh process, so Python globals and module state do
not leak between schedules or back to the caller. The worker uses the parent's
`sys.path`. A scenario that cannot be imported there is an environment error, not
a counterexample. Imports can run again in each
process. **Files, network calls, and other external effects are not rolled back.**
Keep them outside scenarios unless deliberately testing an unsupported boundary.
A Python context manager inside a forcibly terminated process is not guaranteed
to finish; cooperative cleanup and hard termination are distinct behaviors.

## Boundaries

All blocking and shared-state coordination must use ploom primitives. Native locks,
arbitrary `threading`, I/O, uninstrumented mutation, and native weak-memory behavior
fall outside the guarantee. Ploom rejects unsupported API options and detects
stalled workers or observable replay divergence, but cannot detect every
uninstrumented action. Assertions find incorrect outcomes; automatic data-race
detection is not implemented.

Worker requests, results, and saved traces share strict JSON boundary validation.
Wrong types, unknown fields, and inconsistent choices fail explicitly. The engine
uses immutable dataclasses and enums; Pydantic stays at serialization boundaries.

Native atomic operations and a parallel stress backend are not implemented.
Loom's C11 semantics are not Python's memory model.

## Coordination semantics

These primitives use the same single-task, sequentially consistent scheduler as
`SharedCell` and `Lock`; no modeled wait blocks on a native synchronization
primitive. Timed waits, timed acquisition, and queue shutdown are unsupported.
Only the untimed timeout sentinel is accepted. Semaphore initial counts must be
nonnegative. Semaphore `release(n)` requires a positive `n`, and bounded releases
must not exceed the initial count.

`Event.set()` latches a notification for every already registered waiter. A later
`clear()` resets the flag but does not undo those notifications. A wait that
starts after clear blocks until another set. Events and conditions record wait
registration separately from resumption so both orders are explored.

`Condition.wait()` requires the associated lock to be acquired. It atomically
registers a waiter and releases the lock, including all `RLock` recursion levels.
After notification it competes to reacquire the lock and restores that depth.
`wait_for(predicate)` evaluates the predicate while the caller holds the lock
and repeats untimed waits until it becomes truthy. As with Python's basic lock,
`Condition(Lock())` can check only that the lock is held, not which task holds it;
`RLock` checks ownership. Locks must belong to the same execution.

**Condition coverage uses FIFO notification selection.** `notify(n)` selects up
to n oldest registered waiters; nonpositive n wakes none. `notify_all()` selects
all currently registered waiters. The scheduler explores selected waiters'
resumption and lock acquisition order, but does not explore other waiter
selection policies or spurious condition wakeups. Python's public API does not
promise FIFO selection, so a passing condition test covers this modeled policy
only. During cancellation, waiters unregister without reacquiring a lock in the
stopping execution; owned native threads are still joined by the scheduler.

Queues accept the same deeply immutable payloads as `SharedCell`, preventing
unmodeled shared mutation through aliases. Each successful put increments
unfinished work, get removes an item without acknowledging it, `task_done()`
acknowledges one item, and `join()` waits until unfinished work is zero. Queue
observations (`qsize`, `empty`, `full`) are scheduling points and do not reserve
an item or slot. `get_nowait`/`put_nowait` raise standard `queue.Empty`/`queue.Full`.
Neither queues, mutexes, nor semaphores promise waiter fairness.
