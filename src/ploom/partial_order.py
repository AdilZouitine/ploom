"""Sleep-set reduction for sequentially consistent, explicitly modeled effects.

A transition includes its primitive and the following task-local Python segment.
The opt-in contract forbids other cross-task effects. Lifecycle operations and
unknown identities are dependent with everything. No weak-memory model is used.
"""

from dataclasses import dataclass
from itertools import accumulate

from ploom.execution import PrimitiveOperationKind, Step

RESOURCE_ACCESS_KINDS = frozenset(
    {
        PrimitiveOperationKind.CELL_READ,
        PrimitiveOperationKind.CELL_WRITE,
        PrimitiveOperationKind.LOCK_ACQUIRE,
        PrimitiveOperationKind.LOCK_TRY_ACQUIRE,
        PrimitiveOperationKind.LOCK_RELEASE,
        PrimitiveOperationKind.LOCK_LOCKED,
    }
)


@dataclass(frozen=True)
class PendingAction:
    """Identify one task's published next operation.

    Attributes:
        task: Task that will perform the operation.
        operation: Recorded step operation, or None when it is unknown.
    """

    task: int
    operation: str | None = None


@dataclass(frozen=True)
class ResourceAccess:
    """A cell or lock operation on one execution-local resource."""

    kind: PrimitiveOperationKind
    resource: int


def is_ascii_decimal(text: str) -> bool:
    """Whether text is a non-negative decimal integer written in ASCII digits."""
    return text.isascii() and text.isdecimal()


def resource_access(operation: str | None) -> ResourceAccess | None:
    """Parse a recorded ``index:kind:resource`` cell or lock operation.

    Examples:
        >>> resource_access("2:cell.read:5")
        ResourceAccess(kind=<PrimitiveOperationKind.CELL_READ: 'cell.read'>, resource=5)
        >>> resource_access("0:yield") is None
        True
    """
    if operation is None:
        return None
    match operation.split(":"):
        case [operation_index, kind, resource] if (
            kind in RESOURCE_ACCESS_KINDS
            and is_ascii_decimal(operation_index)
            and is_ascii_decimal(resource)
        ):
            return ResourceAccess(PrimitiveOperationKind(kind), int(resource))
        case _:
            return None


def independent(left: PendingAction, right: PendingAction) -> bool:
    """Whether swapping two enabled transitions preserves state and enabledness.

    Only known cell and lock operations participate. In particular, a yield is
    not evidence that the following Python segment has no observable effects.
    Cell and lock resource identities share one execution-local namespace.
    """
    if left.task == right.task:
        return False
    match resource_access(left.operation), resource_access(right.operation):
        case (
            ResourceAccess(kind=PrimitiveOperationKind.CELL_READ),
            ResourceAccess(kind=PrimitiveOperationKind.CELL_READ),
        ):
            return True
        case ResourceAccess(resource=left_resource), ResourceAccess(resource=right_resource):
            return left_resource != right_resource
        case _:
            return False


@dataclass(frozen=True)
class SleepPrefix:
    """Own a DFS branch and transitions already covered by earlier siblings."""

    tasks: tuple[int, ...] = ()
    sleeping: tuple[PendingAction, ...] = ()


@dataclass(frozen=True)
class SleepSetExpansion:
    """The DFS successors of one execution under a sleep prefix.

    Attributes:
        is_representative: Whether the execution is the first of its equivalence
            class, so its outcome counts.
        branches: Successor prefixes in stack order.
    """

    is_representative: bool
    branches: tuple[SleepPrefix, ...]


def with_next_action(
    next_actions: dict[int, PendingAction], step: Step
) -> dict[int, PendingAction]:
    """Record ``step`` as its task's next action, walking a schedule backwards."""
    return next_actions | {step.task: PendingAction(step.task, step.operation)}


def expand_sleep_prefix(steps: tuple[Step, ...], sleep_prefix: SleepPrefix) -> SleepSetExpansion:
    """Build DFS successors and say whether the observed suffix is representative.

    An enabled task's next selected step identifies its already-published action.
    If failure prevents that selection, its action is unknown and never commutes.
    A suffix selecting a sleeping action belongs to an earlier sibling's class;
    its outcome is ignored, but its awake siblings still need exploration.

    Branches are in stack order: deeper alternatives precede shallower ones when
    popped, so a sibling's complete subtree is visited before it enters sleep.
    """
    no_actions: dict[int, PendingAction] = {}
    # next_actions_at[index] maps each task to its first action in steps[index:].
    next_actions_at = tuple(accumulate(reversed(steps), with_next_action, initial=no_actions))[::-1]
    sleeping_actions = sleep_prefix.sleeping
    branches: list[SleepPrefix] = []
    chosen_tasks = list(sleep_prefix.tasks)
    for index in range(len(sleep_prefix.tasks), len(steps)):
        step = steps[index]
        next_actions = next_actions_at[index]
        chosen_action = next_actions[step.task]
        explored_actions = list(sleeping_actions)
        if chosen_action not in sleeping_actions:
            explored_actions.append(chosen_action)
        sibling_branches: list[SleepPrefix] = []
        for task in step.runnable:
            sibling_action = next_actions.get(task, PendingAction(task))
            if task == step.task or sibling_action in sleeping_actions:
                continue
            sibling_sleep_set = tuple(
                explored_action
                for explored_action in explored_actions
                if independent(sibling_action, explored_action)
            )
            sibling_branches.append(SleepPrefix((*chosen_tasks, task), sibling_sleep_set))
            explored_actions.append(sibling_action)
        branches.extend(reversed(sibling_branches))
        if chosen_action in sleeping_actions:
            return SleepSetExpansion(is_representative=False, branches=tuple(branches))
        sleeping_actions = tuple(
            sleeping_action
            for sleeping_action in sleeping_actions
            if independent(chosen_action, sleeping_action)
        )
        chosen_tasks.append(step.task)
    return SleepSetExpansion(is_representative=True, branches=tuple(branches))
