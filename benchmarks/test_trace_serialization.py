"""Measure validation and JSON encoding of traces at the evidence boundary."""

import pytest

from ploom.execution import Execution, Failed, Failure, FailureCategory, Step
from ploom.trace import Config, ExecutionEvidence, Provenance, ScenarioTransport, Trace

TRACE_LENGTH = 1000
FAILURE = Failure(
    FailureCategory.EXCEPTION,
    "AssertionError",
    "the account is overdrawn",
    'Traceback (most recent call last):\n  File "scenario.py", line 14\n',
    "scenario.py:14",
)
PROVENANCE = Provenance(
    reference="tests.scenarios:withdraw",
    digest="0" * 64,
    transport=ScenarioTransport.IMPORT,
    python="3.13.0",
)


def contended_steps(length: int) -> tuple[Step, ...]:
    """Build a long two-task schedule that alternates between both tasks."""
    return tuple(Step(index % 2, f"cell.write:{index // 2}", (0, 1)) for index in range(length))


@pytest.fixture(scope="module")
def trace() -> Trace:
    """A failing 1000-step trace, validated once outside the measurement."""
    return Trace.from_execution(
        PROVENANCE, Config(), Execution(contended_steps(TRACE_LENGTH), Failed(FAILURE))
    )


@pytest.fixture(scope="module")
def trace_json(trace: Trace) -> bytes:
    """The JSON document ``write_trace`` would publish for ``trace``."""
    return trace.model_dump_json(indent=2).encode()


@pytest.mark.benchmark
def test_trace_construction() -> None:
    """Measure strict validation when a failing execution becomes a trace."""
    execution = Execution(contended_steps(TRACE_LENGTH), Failed(FAILURE))
    trace = Trace.from_execution(PROVENANCE, Config(), execution)
    assert len(trace.steps) == TRACE_LENGTH


@pytest.mark.benchmark
def test_trace_encoding(trace: Trace) -> None:
    """Measure JSON encoding of a 1000-step trace."""
    assert trace.model_dump_json(indent=2)


@pytest.mark.benchmark
def test_trace_decoding(trace_json: bytes) -> None:
    """Measure parsing and validation of a 1000-step trace artifact."""
    decoded = Trace.model_validate_json(trace_json)
    assert len(decoded.steps) == TRACE_LENGTH


@pytest.mark.benchmark
def test_worker_evidence_round_trip() -> None:
    """Measure the worker-to-parent evidence encoding and validation round trip."""
    execution = Execution(contended_steps(TRACE_LENGTH), Failed(FAILURE))
    evidence = ExecutionEvidence(
        steps=execution.steps,
        outcome=execution.outcome,
        cleanup_failures=execution.cleanup_failures,
    )
    decoded = ExecutionEvidence.model_validate_json(evidence.model_dump_json(indent=2))
    assert decoded.execution() == execution
