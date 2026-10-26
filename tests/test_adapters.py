import hashlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

import ploom
from ploom.__main__ import import_scenario, main
from ploom.execution import Passed, Step
from ploom.trace import Provenance, ScenarioTransport, write_trace
from tests.helpers import run_ploom

INVALID_EXIT = 2


def test_cli_check_replay_and_changed_code(integration_workspace: Path) -> None:
    module = integration_workspace / "scenarios.py"
    assert run_ploom(integration_workspace, "check", "scenarios:passing").returncode == 0
    failed = run_ploom(integration_workspace, "check", "scenarios:failing")
    assert failed.returncode == 1
    assert "counterexample" in failed.stderr
    trace = next((integration_workspace / ".ploom").glob("*-original.json"))
    replayed = run_ploom(integration_workspace, "replay", str(trace))
    assert replayed.returncode == 1
    assert "reproduced failed" in replayed.stdout
    module.write_text(module.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    divergent = run_ploom(integration_workspace, "replay", str(trace))
    assert divergent.returncode == INVALID_EXIT
    assert "changed" in divergent.stderr
    assert (
        run_ploom(integration_workspace, "check", "scenarios:not_callable").returncode
        == INVALID_EXIT
    )
    configuration = integration_workspace / "step_limit.json"
    limited = run_ploom(
        integration_workspace, "check", "scenarios:passing", "--config", str(configuration)
    )
    assert limited.returncode == INVALID_EXIT
    assert "steps limit" in limited.stderr


def test_console_script_resolves_scenarios_from_the_working_directory(
    integration_workspace: Path,
) -> None:
    script = shutil.which("ploom", path=Path(sys.executable).parent)
    assert script is not None
    result = subprocess.run(
        [script, "check", "scenarios:passing"],
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_errors_and_passed_replay(integration_workspace: Path) -> None:
    for reference in ("invalid", ":function", "module:", "module:object.method"):
        with pytest.raises(ValueError, match="module:function"):
            import_scenario(reference)
    assert main(["replay", str(integration_workspace / "absent")]) == INVALID_EXIT

    def failing() -> None:
        raise ValueError("failure")

    with pytest.raises(ploom.ModelCheckError) as caught:
        ploom.check(
            failing, trace_dir=integration_workspace, config=ploom.Config(minimize_executions=0)
        )
    assert main(["replay", str(caught.value.trace_path)]) == INVALID_EXIT
    assert caught.value.reduction.status == "disabled"

    module = integration_workspace / "passing.py"
    # An empty importable scenario's only operation is the root begin checkpoint.
    trace = ploom.Trace(
        scenario=Provenance(
            reference="passing:scenario",
            digest=hashlib.sha256(module.read_bytes()).hexdigest(),
            transport=ScenarioTransport.IMPORT,
            python=f"{sys.version_info.major}.{sys.version_info.minor}",
        ),
        config=ploom.Config(),
        steps=(Step(0, "0:begin", (0,)),),
        outcome=Passed(),
    )
    path = integration_workspace / "passed.json"
    write_trace(path, trace)
    passed = run_ploom(integration_workspace, "replay", str(path))
    assert passed.returncode == 0, passed.stderr


def test_pytest_plugin_failure_and_replay(integration_workspace: Path) -> None:
    test = integration_workspace / "plugin_failure.py"
    command = [sys.executable, "-m", "pytest", "-q", str(test)]
    initial = subprocess.run(
        command,
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert initial.returncode == 1, initial.stdout + initial.stderr
    assert "ploom first failure" in initial.stdout
    assert "Minimized trace:" in initial.stdout
    assert "--ploom-replay=" in initial.stdout
    assert "plugin_failure.py::test_failure --ploom-replay=" in initial.stdout
    trace = next((integration_workspace / ".ploom").glob("*-original.json"))
    reproduced = subprocess.run(
        [*command, f"--ploom-replay={trace}"],
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert reproduced.returncode == 1
    assert "ploom reproduced builtins.AssertionError" in reproduced.stdout
    without_terminal = subprocess.run(
        [*(part for part in command if part != "-q"), "-p", "no:terminal"],
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert without_terminal.returncode == 1


def test_pytest_plugin_passed_and_direct_api(integration_workspace: Path) -> None:
    test = integration_workspace / "plugin_mixed.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test)],
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    assert "1 failed, 1 passed" in result.stdout
    assert "ploom.replay(scenario" in result.stdout
    invalid_replay = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test), "--ploom-replay=absent.json"],
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert "requires exactly one test" in invalid_replay.stderr


def test_pytest_passed_trace_replay(integration_workspace: Path) -> None:
    test = integration_workspace / "plugin_passed.py"
    command = [sys.executable, "-m", "pytest", "-q", str(test)]
    for extra in ([], [f"--ploom-replay={integration_workspace / 'passed.json'}"]):
        result = subprocess.run(
            [*command, *extra],
            cwd=integration_workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_module_level_scenario_in_a_directory_that_pytest_inserts(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_module_level.py").write_text(
        textwrap.dedent(
            """
            import ploom


            def scenario():
                ploom.yield_now()


            def test_module_level(ploom_check):
                assert ploom_check(scenario).executions == 1
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(suite)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_reports_unloadable_scenarios_as_invalid_input() -> None:
    with pytest.raises(ValueError, match="cannot load scenario") as caught:
        import_scenario("ploom_absent_module:scenario")
    assert type(caught.value.__cause__) is ModuleNotFoundError
    assert main(["check", "ploom_absent_module:scenario"]) == INVALID_EXIT


def test_trace_boundary_validation(integration_workspace: Path) -> None:
    invalid = integration_workspace / "trace.json"
    invalid.write_text('{"version": 2}', encoding="utf-8")
    with pytest.raises(ValidationError):
        ploom.read_trace(invalid)
    with pytest.raises(ValidationError, match="scheduling decision"):
        ploom.Trace(
            scenario=Provenance(
                reference="example:f",
                digest="digest",
                transport=ScenarioTransport.IMPORT,
                python="3",
            ),
            config=ploom.Config(),
            steps=(Step(1, "read", (0,)),),
            outcome=Passed(),
        )
