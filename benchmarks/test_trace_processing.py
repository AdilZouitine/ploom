"""Measure pure trace-processing costs without subprocess or native-thread noise."""

import pytest

from ploom.execution import (
    Execution,
    Passed,
    Step,
    alternative_prefixes,
    count_preemptions,
    validate_replay,
)


@pytest.mark.benchmark
def test_branch_enumeration() -> None:
    """Measure alternate-prefix generation for a contended 200-step trace."""
    steps = tuple(Step(index % 2, f"{index // 2}:yield", (0, 1)) for index in range(200))
    branch_prefixes = alternative_prefixes(steps)
    assert len(branch_prefixes) == len(steps)


@pytest.mark.benchmark
def test_preemption_counting() -> None:
    """Measure logical task-switch counting over a longer trace."""
    steps = tuple(Step(index % 2, f"{index // 2}:yield", (0, 1)) for index in range(1000))
    assert count_preemptions(steps) == len(steps) - 1


@pytest.mark.benchmark
def test_replay_comparison() -> None:
    """Measure consistency checks for an unchanged successful execution."""
    steps = tuple(Step(0, f"{index}:yield", (0,)) for index in range(1000))
    execution = Execution(steps, Passed())
    replayed = validate_replay(execution, tuple(step.task for step in steps), execution)
    assert replayed == execution
