"""Validate partial execution orders independently from the scheduler."""

from enum import StrEnum
from graphlib import TopologicalSorter

from pydantic import BaseModel, ConfigDict, Field

RESERVED_EVENT_PREFIX = "@"


class LifecycleEvent(StrEnum):
    """Reserved events that every constrained named thread selects."""

    START = "@start"
    FINISH = "@finish"


class Point(BaseModel):
    """Identify a named thread event; lifecycle events are @start and @finish.

    Checkpoints describe the instant the checkpoint is selected, before the
    caller continues. Explicit occurrence numbers distinguish loop iterations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    thread: str = Field(min_length=1)
    event: str = Field(min_length=1)
    occurrence: int = Field(default=1, ge=1)


class Before(BaseModel):
    """Require one event to be selected strictly before another event."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    before: Point
    after: Point


def is_valid_lifecycle_use(point: Point) -> bool:
    """Whether a reserved event name is a lifecycle event with occurrence 1."""
    if not point.event.startswith(RESERVED_EVENT_PREFIX):
        return True
    return point.event in LifecycleEvent and point.occurrence == 1


def thread_orders(earlier: Point, later: Point) -> bool:
    """Whether one thread's own execution selects ``earlier`` before ``later``."""
    if earlier.thread != later.thread or earlier == later:
        return False
    return (
        earlier.event == LifecycleEvent.START
        or later.event == LifecycleEvent.FINISH
        or (earlier.event == later.event and earlier.occurrence < later.occurrence)
    )


def validate_ordering(ordering: tuple[Before, ...]) -> tuple[Before, ...]:
    """Reject cycles, including contradictions with thread lifecycle order.

    Raises:
        ValueError: A point misuses a reserved event, or the order has a cycle.
    """
    points = {point for edge in ordering for point in (edge.before, edge.after)}
    if not all(map(is_valid_lifecycle_use, points)):
        raise ValueError("reserved lifecycle events are @start and @finish, occurrence 1")
    predecessors = {
        point: {edge.before for edge in ordering if edge.after == point}
        | {earlier for earlier in points if thread_orders(earlier, point)}
        for point in points
    }
    TopologicalSorter(predecessors).prepare()
    return ordering


class ScheduleConstraintError(ValueError):
    """No terminal execution satisfies the requested partial order."""
