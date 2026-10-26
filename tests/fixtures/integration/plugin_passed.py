"""Save successful execution evidence and replay it through the pytest fixture."""

from pathlib import Path

import ploom
from ploom.pytest_plugin import ScenarioCheck
from ploom.subprocess_runner import subprocess_runner
from ploom.trace import write_trace


def test_pass(ploom_check: ScenarioCheck) -> None:
    def scenario() -> None:
        ploom.yield_now()

    with subprocess_runner(scenario, ploom.Config()) as runner:
        execution = runner.run((), 5)
        assert execution.outcome == ploom.Passed()
        trace = ploom.Trace.from_execution(
            runner.prepared_scenario.provenance, ploom.Config(), execution
        )
    write_trace(Path.cwd() / "passed.json", trace)
    assert ploom_check(scenario).executions == 1
