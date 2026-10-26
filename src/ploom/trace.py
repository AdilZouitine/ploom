"""Validate configuration and portable evidence at external boundaries."""

from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, assert_never

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ploom.execution import (
    Diverged,
    Execution,
    Failed,
    Failure,
    Incomplete,
    Outcome,
    Passed,
    Step,
)


class Config(BaseModel):
    """Bound search and reduction independently; exhaustion never means success.

    Attributes:
        max_executions: Maximum schedules explored before declaring incompleteness.
        max_steps: Maximum modeled operations in a single execution.
        max_tasks: Maximum concurrent tasks, including the root task.
        timeout: Exploration wall-time budget in seconds.
        max_preemptions: Explore only schedules with at most this many preemptions,
            or None to explore every schedule. A bounded pass covers only the
            schedules within the bound.
        partial_order_reduction: Opt into sleep-set reduction under the documented
            modeled-effects contract. Preemption bounds use exhaustive search.
        hash_seed: ``PYTHONHASHSEED`` for every worker. Each schedule runs in a fresh
            process, so ``str`` hashing and set order must not change between them.
        minimize_executions: Additional executions allowed for failure reduction.
        minimize_timeout: Separate reduction wall-time budget in seconds.

    Examples:
        >>> limits = Config(max_steps=20, timeout=5.0)
        >>> limits.max_steps
        20
        >>> limits.max_tasks
        16
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    # A fresh worker process costs about 0.1 s, so this deterministic bound
    # binds before the default wall-time bound on a typical machine.
    max_executions: int = Field(default=200, gt=0)
    max_steps: int = Field(default=1000, gt=0)
    max_tasks: int = Field(default=16, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    max_preemptions: int | None = Field(default=None, ge=0)
    partial_order_reduction: bool = False
    hash_seed: int = Field(default=0, ge=0, le=4_294_967_295)
    minimize_executions: int = Field(default=100, ge=0)
    minimize_timeout: float = Field(default=2.0, ge=0)


class ScenarioTransport(StrEnum):
    """Select portable imports or trusted temporary callable serialization."""

    IMPORT = auto()
    LOCAL = auto()


class Provenance(BaseModel):
    """Identify scenario code and input without embedding executable data."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    reference: str
    digest: str
    transport: ScenarioTransport
    python: str


def is_valid_step(step: Step) -> bool:
    """Whether a step chose one of its sorted, unique, non-negative runnable tasks.

    Examples:
        >>> is_valid_step(Step(1, "0:begin", (0, 1)))
        True
        >>> is_valid_step(Step(1, "0:begin", (1, 0)))
        False
    """
    return (
        bool(step.operation)
        and step.task in step.runnable
        and step.runnable == tuple(sorted(set(step.runnable)))
        and all(task >= 0 for task in step.runnable)
    )


def require_valid_steps(steps: tuple[Step, ...]) -> tuple[Step, ...]:
    """Reject impossible or ambiguous scheduling records at the boundary."""
    if not all(map(is_valid_step, steps)):
        raise ValueError("invalid scheduling decision")
    return steps


Schedule = Annotated[tuple[Step, ...], AfterValidator(require_valid_steps)]


class ExecutionEvidence(BaseModel):
    """Parse worker evidence with the same strict scheduling contract as traces.

    Attributes:
        steps: Ordered scheduling choices and operation identities.
        outcome: Terminal execution outcome, selected by its explicit wire tag.
        cleanup_failures: Secondary failures captured while cancelling tasks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    steps: Schedule
    outcome: Annotated[Outcome, Field(discriminator="kind")]
    cleanup_failures: tuple[Failure, ...] = ()

    def execution(self) -> Execution:
        """Return validated evidence in the engine's framework-free types."""
        return Execution(self.steps, self.outcome, self.cleanup_failures)


class Trace(BaseModel):
    """Attach portable provenance to one replayable, terminal execution.

    Only passed and failed executions are evidence. A limit or a divergence says
    nothing reproducible about the scenario, so a trace cannot hold one.

    Attributes:
        version: Supported JSON schema version.
        scenario: Code, input, and interpreter provenance.
        config: Original execution and exploration limits.
        steps: Ordered scheduling choices and operation identities.
        outcome: Original terminal execution outcome.
        cleanup_failures: Secondary failures captured while cancelling tasks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1] = 1
    scenario: Provenance
    config: Config
    steps: Schedule
    outcome: Annotated[Passed | Failed, Field(discriminator="kind")]
    cleanup_failures: tuple[Failure, ...] = ()

    def execution(self) -> Execution:
        """Return validated evidence in the engine's framework-free types."""
        return Execution(self.steps, self.outcome, self.cleanup_failures)

    @classmethod
    def from_execution(cls, scenario: Provenance, config: Config, execution: Execution) -> Trace:
        """Attach provenance and limits to a passed or failed execution.

        Args:
            scenario: Code, input, and interpreter provenance.
            config: Limits under which the execution ran.
            execution: The passed or failed execution to preserve.

        Returns:
            A trace in the current schema version.

        Raises:
            ValueError: If the execution did not pass or fail.
        """
        match execution.outcome:
            case Passed() | Failed() as outcome:
                return cls(
                    scenario=scenario,
                    config=config,
                    steps=execution.steps,
                    outcome=outcome,
                    cleanup_failures=execution.cleanup_failures,
                )
            case Incomplete() | Diverged():
                raise ValueError("only passed or failed executions are replayable evidence")
            case _ as unreachable:
                assert_never(unreachable)

    def with_execution(self, execution: Execution) -> Trace:
        """Derive new evidence while keeping its provenance and limits unchanged.

        Raises:
            ValueError: If the execution did not pass or fail.
        """
        return Trace.from_execution(self.scenario, self.config, execution)


def write_trace(trace_path: Path, trace: Trace) -> None:
    """Publish a complete JSON document through an atomic replacement.

    A reader never sees a partial trace, so an artifact that exists is evidence.

    Args:
        trace_path: Destination file; its directory is created if needed.
        trace: Validated evidence to preserve.
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ploom-", dir=trace_path.parent) as staging_directory:
        staged_trace = Path(staging_directory) / "trace.json"
        staged_trace.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        staged_trace.replace(trace_path)


def read_trace(trace_path: Path) -> Trace:
    """Parse and validate portable evidence once at the JSON boundary.

    Args:
        trace_path: JSON artifact to read.

    Returns:
        A validated immutable trace ready for replay.

    Raises:
        OSError: If the artifact cannot be read.
        pydantic.ValidationError: If its schema or scheduling choices are invalid.
    """
    return Trace.model_validate_json(trace_path.read_bytes())


def write_execution(result_path: Path, execution: Execution) -> None:
    """Encode worker evidence using the same validated wire representation."""
    evidence = ExecutionEvidence(
        steps=execution.steps,
        outcome=execution.outcome,
        cleanup_failures=execution.cleanup_failures,
    )
    result_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")


def read_execution(result_path: Path) -> Execution:
    """Validate evidence received from an isolated worker."""
    return ExecutionEvidence.model_validate_json(result_path.read_bytes()).execution()
