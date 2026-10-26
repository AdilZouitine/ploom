"""Measure depth-first schedule search and reduction over a synthetic schedule tree.

A pure-Python runner stands in for the subprocess backend, so these workloads
measure the search loop itself without process startup or native-thread noise.
"""

from collections.abc import Callable

import pytest

from ploom.execution import (
    Execution,
    Failed,
    Failure,
    FailureCategory,
    Passed,
    ReductionStatus,
    Step,
    count_preemptions,
)
from ploom.exploration import explore, minimize
from ploom.trace import Config

# Generous wall-time budgets keep instrumented runs from hitting deadlines.
UNBOUNDED_CONFIG = Config(
    max_executions=100_000, timeout=3600.0, minimize_executions=100_000, minimize_timeout=3600.0
)
LOST_UPDATE = Failure(
    FailureCategory.EXCEPTION,
    "AssertionError",
    "root task was interrupted",
    "",
    "scenario.py:1",
)

# Interleavings of two six-operation tasks: 12 choose 6.
TWO_BY_SIX_SCHEDULES = 924

type Verdict = Callable[[tuple[Step, ...]], bool]


def always_passes(_steps: tuple[Step, ...]) -> bool:
    """Accept every schedule so the search visits the whole tree."""
    return True


def root_interrupted(steps: tuple[Step, ...]) -> bool:
    """Fail every schedule where another task runs between the root's first and last step."""
    return not (steps[0].task == 0 and steps[-1].task == 0)


class SyntheticRunner:
    """Replay a fixed number of always-runnable operations per task.

    The default policy matches the real scheduler: follow the required prefix,
    then keep the previous task running while it can, else pick the lowest task.
    """

    def __init__(self, tasks: int, operations_per_task: int, verdict: Verdict) -> None:
        self.tasks = tasks
        self.operations_per_task = operations_per_task
        self.verdict = verdict

    def run(
        self,
        required_prefix: tuple[int, ...],
        _timeout_seconds: float,
        _expected_execution: Execution | None = None,
        /,
    ) -> Execution:
        """Build the execution selected by ``required_prefix`` and the default policy."""
        remaining = [self.operations_per_task] * self.tasks
        steps: list[Step] = []
        previous_task = 0
        for step_index in range(self.tasks * self.operations_per_task):
            runnable = tuple(task for task, left in enumerate(remaining) if left)
            if step_index < len(required_prefix):
                chosen_task = required_prefix[step_index]
            elif previous_task in runnable:
                chosen_task = previous_task
            else:
                chosen_task = runnable[0]
            operation_index = self.operations_per_task - remaining[chosen_task]
            steps.append(Step(chosen_task, f"{operation_index}:yield", runnable))
            remaining[chosen_task] -= 1
            previous_task = chosen_task
        schedule = tuple(steps)
        outcome = Passed() if self.verdict(schedule) else Failed(LOST_UPDATE)
        return Execution(schedule, outcome)


@pytest.mark.benchmark
def test_exhaustive_exploration() -> None:
    """Measure a full search of every interleaving of two six-operation tasks."""
    result = explore(SyntheticRunner(2, 6, always_passes), UNBOUNDED_CONFIG)
    assert result.execution.outcome == Passed()
    assert result.statistics.executions == TWO_BY_SIX_SCHEDULES


@pytest.mark.benchmark
def test_preemption_bounded_exploration() -> None:
    """Measure a preemption-bounded search over three five-operation tasks."""
    config = UNBOUNDED_CONFIG.model_copy(update={"max_preemptions": 2})
    result = explore(SyntheticRunner(3, 5, always_passes), config)
    assert result.execution.outcome == Passed()
    assert result.statistics.executions > 1


@pytest.mark.benchmark
def test_failure_minimization() -> None:
    """Measure reduction of an alternating counterexample to a single preemption."""
    runner = SyntheticRunner(2, 6, root_interrupted)
    original = runner.run((0, 1) * 5 + (1, 0), 0.0)
    assert isinstance(original.outcome, Failed)
    reduction = minimize(runner, original, UNBOUNDED_CONFIG)
    assert reduction.status is ReductionStatus.COMPLETED
    assert count_preemptions(reduction.execution.steps) == 1
