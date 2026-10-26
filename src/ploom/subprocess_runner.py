"""Execute schedules in fresh processes with hard, parent-owned deadlines."""

import os
import subprocess
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, assert_never

from pydantic import BaseModel, ConfigDict, Field

from ploom.execution import Diverged, Execution, ExplorationLimit, Failed, Incomplete, Passed
from ploom.scenarios import PreparedScenario, ScenarioSource, prepare_scenario
from ploom.trace import Config, read_execution

LOG_TAIL_BYTES = 65536


class WorkerRequest(BaseModel):
    """Validate external worker input before constructing the thread scheduler.

    Attributes:
        source: Scenario transport to resolve before execution.
        config: Validated execution limits.
        required_prefix: Task choices forced before deterministic scheduling.
        import_path: The parent's ``sys.path``, so the worker resolves the same
            modules, including directories that pytest inserted.
        expected_execution: Exact evidence to validate during replay, if any.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    source: Annotated[ScenarioSource, Field(discriminator="kind")]
    config: Config
    required_prefix: tuple[int, ...]
    import_path: tuple[str, ...]
    expected_execution: Execution | None = None


def read_log_tail(log_path: Path) -> str:
    """Read the end of a worker log without loading unbounded output."""
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - LOG_TAIL_BYTES))
        return log.read().decode(errors="replace")


@dataclass(frozen=True)
class SubprocessRunner:
    """Execute schedules while the surrounding context owns temporary transport.

    Attributes:
        config: Validated limits sent to every isolated worker.
        prepared_scenario: Executable source and its durable replay provenance.
        working_directory: Scoped directory for worker results and logs.
    """

    config: Config
    prepared_scenario: PreparedScenario
    working_directory: Path

    def run(
        self,
        required_prefix: tuple[int, ...],
        timeout_seconds: float,
        expected_execution: Execution | None = None,
    ) -> Execution:
        """Kill and reap unresponsive workers instead of abandoning their threads.

        The request travels on the worker's stdin. Scenario output goes to a
        per-execution log. A failed execution keeps that output in its failure
        traceback, so ``print`` debugging stays visible.

        Args:
            required_prefix: Task choices forced before deterministic scheduling.
            timeout_seconds: Remaining wall-time budget.
            expected_execution: Exact evidence to validate during replay, if any.

        Returns:
            Ordered execution evidence or an explicit wall-time limit outcome.

        Raises:
            RuntimeError: If a worker exits without valid execution evidence.
        """
        if timeout_seconds <= 0:
            return Execution((), Incomplete(ExplorationLimit.WALL_TIME))
        result_path = self.working_directory / "result.json"
        result_path.unlink(missing_ok=True)
        request = WorkerRequest(
            source=self.prepared_scenario.source,
            config=self.config,
            required_prefix=required_prefix,
            import_path=tuple(sys.path),
            expected_execution=expected_execution,
        )
        log_path = self.working_directory / "worker.log"
        with log_path.open("wb") as log:
            try:
                worker = subprocess.run(
                    [sys.executable, "-m", "ploom.execution_worker", str(result_path)],
                    input=request.model_dump_json().encode(),
                    stdout=log,
                    stderr=log,
                    env={**os.environ, "PYTHONHASHSEED": str(self.config.hash_seed)},
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return Execution((), Incomplete(ExplorationLimit.WALL_TIME))
        if worker.returncode != 0 or not result_path.exists():
            raise RuntimeError(
                f"ploom execution worker exited unexpectedly: {read_log_tail(log_path)}"
            )
        execution = read_execution(result_path)
        scenario_output = read_log_tail(log_path)
        match execution.outcome:
            case Failed(failure) if scenario_output:
                traceback = f"{failure.traceback}\nWorker output:\n{scenario_output}"
                return replace(execution, outcome=Failed(replace(failure, traceback=traceback)))
            case Passed() | Failed() | Incomplete() | Diverged():
                return execution
            case _ as unreachable:
                assert_never(unreachable)


@contextmanager
def subprocess_runner(
    scenario: Callable[[], object], config: Config
) -> Generator[SubprocessRunner]:
    """Own scenario transport from preparation through exploration and reduction.

    Args:
        scenario: Ordinary scenario code whose inputs must be isolated.
        config: Validated execution limits.

    Yields:
        A runner with fresh processes and scoped temporary files.
    """
    with TemporaryDirectory(prefix="ploom-") as temporary_directory:
        working_directory = Path(temporary_directory)
        yield SubprocessRunner(
            config, prepare_scenario(scenario, working_directory), working_directory
        )
