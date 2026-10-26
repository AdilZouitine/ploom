"""Process entry point for one isolated modeled execution."""

import sys
from pathlib import Path

from ploom.scenarios import load_scenario
from ploom.subprocess_runner import WorkerRequest
from ploom.thread_scheduler import ThreadScheduler
from ploom.trace import write_execution


def main() -> None:
    """Read the request from stdin and write the execution to the given path.

    Loading the scenario happens before any modeled task exists, so an import
    error crashes the worker instead of becoming a counterexample. Deadlines and
    diagnostics stay with the parent.
    """
    request = WorkerRequest.model_validate_json(sys.stdin.buffer.read())
    sys.path[:] = request.import_path
    scenario = load_scenario(request.source)
    scheduler = ThreadScheduler(request.config.max_steps, request.config.max_tasks)
    execution = scheduler.execute(scenario, request.required_prefix, request.expected_execution)
    write_execution(Path(sys.argv[1]), execution)


if __name__ == "__main__":
    main()
