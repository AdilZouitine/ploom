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

Event, Condition, Semaphore, Queue, native atomic operations, and a parallel stress
backend are not implemented. Loom's C11 semantics are not Python's memory model.
