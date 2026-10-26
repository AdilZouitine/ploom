import json
from contextlib import suppress
from pathlib import Path

import pytest
from pydantic import ValidationError

import ploom
from ploom.execution import (
    Diverged,
    Execution,
    ExplorationLimit,
    Failed,
    Failure,
    FailureCategory,
    Incomplete,
    Passed,
    Step,
    validate_replay,
)
from ploom.subprocess_runner import WorkerRequest
from ploom.thread_scheduler import ThreadScheduler
from ploom.trace import (
    Provenance,
    ScenarioTransport,
    read_execution,
    write_execution,
    write_trace,
)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "steps": [{"task": "0", "operation": "0:begin", "runnable": [0]}],
            "outcome": {"kind": "passed"},
        },
        {"steps": [], "outcome": {"kind": "passed"}, "unexpected": True},
        {"steps": [], "outcome": {"kind": "passed", "unexpected": True}},
        {
            "steps": [{"task": -1, "operation": "0:begin", "runnable": [-1]}],
            "outcome": {"kind": "passed"},
        },
    ],
)
def test_worker_evidence_rejects_invalid_wire_data(
    payload: dict[str, object], tmp_path: Path
) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        read_execution(path)


def test_worker_request_rejects_coerced_task_choices() -> None:
    payload = {
        "source": {"kind": "local", "path": "scenario.pickle"},
        "config": {},
        "required_prefix": [0],
        "import_path": [],
    }
    WorkerRequest.model_validate_json(json.dumps(payload))
    payload["required_prefix"] = ["0"]
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate_json(json.dumps(payload))


def test_worker_evidence_round_trip_preserves_cleanup_failures(tmp_path: Path) -> None:
    failure = Failure(
        FailureCategory.EXCEPTION, "builtins.ValueError", "cleanup", "stack", "file:1"
    )
    execution = Execution((Step(0, "0:begin", (0,)),), Passed(), (failure,))
    path = tmp_path / "result.json"
    write_execution(path, execution)
    assert read_execution(path) == execution


def test_failed_trace_publication_removes_staging_files(tmp_path: Path) -> None:
    trace = ploom.Trace(
        scenario=Provenance(
            reference="example:scenario",
            digest="digest",
            transport=ScenarioTransport.IMPORT,
            python="3.13",
        ),
        config=ploom.Config(),
        steps=(),
        outcome=Passed(),
    )
    destination = tmp_path / "existing-directory"
    destination.mkdir()
    with pytest.raises(OSError, match="existing-directory"):
        write_trace(destination, trace)
    assert set(tmp_path.iterdir()) == {destination}


@pytest.mark.parametrize("deadlock", [True, False])
def test_cancellation_preserves_cleanup_failures_after_scheduler_outcomes(deadlock: bool) -> None:
    def scenario() -> None:
        lock = ploom.Lock()
        try:
            if deadlock:
                lock.acquire()
                lock.acquire()
            else:
                ploom.yield_now()
        finally:
            raise ValueError("cleanup failed")

    scheduler = ThreadScheduler(ploom.Config(max_steps=100 if deadlock else 1, max_tasks=16))
    execution = scheduler.execute(scenario, (), None)
    match execution.outcome:
        case Failed(failure):
            assert failure.category is FailureCategory.DEADLOCK
        case Incomplete(limit):
            assert limit is ExplorationLimit.STEPS
        case unexpected:
            pytest.fail(f"unexpected execution outcome: {unexpected}")
    assert [failure.message for failure in execution.cleanup_failures] == ["cleanup failed"]


def test_replay_divergence_preserves_secondary_evidence() -> None:
    failure = Failure(
        FailureCategory.EXCEPTION, "builtins.ValueError", "cleanup", "stack", "file:1"
    )
    actual = Execution((), Passed(), (failure,))
    replayed = validate_replay(actual, (), Execution((), Passed()))
    assert replayed.outcome == Diverged("terminal outcome changed")
    assert replayed.cleanup_failures == actual.cleanup_failures


def test_minimized_publication_failure_preserves_the_original_counterexample(
    tmp_path: Path,
) -> None:
    def scenario() -> None:
        raise AssertionError("original failure")

    def block_minimized_publication(_trace: ploom.Trace, original_path: Path) -> None:
        original_path.with_name(original_path.name.replace("-original", "-minimized")).mkdir()

    with pytest.raises(OSError, match="minimized") as caught:
        ploom.check(scenario, trace_dir=tmp_path, on_failure=block_minimized_publication)
    assert caught.value.__notes__[0].startswith(
        "Modeled failure: builtins.AssertionError: original failure"
    )
    (original,) = tmp_path.glob("*-original.json")
    assert ploom.replay(scenario, original) == ploom.read_trace(original)


def test_task_limit_survives_scenario_exception_handlers() -> None:
    def scenario() -> None:
        with suppress(BaseException):
            ploom.Thread().start()
        ploom.yield_now()

    execution = ThreadScheduler(ploom.Config(max_steps=100, max_tasks=1)).execute(
        scenario, (), None
    )
    assert execution.outcome == Incomplete(ExplorationLimit.TASKS)


def test_incomplete_python_api_retains_cleanup_diagnostics() -> None:
    def scenario() -> None:
        try:
            ploom.yield_now()
        finally:
            raise ValueError("cleanup after step limit")

    with pytest.raises(ploom.IncompleteExplorationError, match="step limit") as caught:
        ploom.check(scenario, config=ploom.Config(max_steps=1))
    assert caught.value.limit is ploom.ExplorationLimit.STEPS
    assert caught.value.cleanup_failures[0].message == "cleanup after step limit"


def test_original_publication_error_keeps_the_modeled_failure_in_exception_notes(
    tmp_path: Path,
) -> None:
    def scenario() -> None:
        raise ValueError("original counterexample")

    blocked_directory = tmp_path / "blocked-directory"
    blocked_directory.write_text("not a directory", encoding="utf-8")
    with pytest.raises(FileExistsError, match="blocked-directory") as caught:
        ploom.check(scenario, trace_dir=blocked_directory)
    assert "original counterexample" in "\n".join(caught.value.__notes__)
    assert list(tmp_path.iterdir()) == [blocked_directory]
