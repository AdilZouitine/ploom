"""Thin pytest fixture and artifact reporting over the framework-independent API."""

import os
import shlex
import subprocess
from collections.abc import Callable, Generator
from pathlib import Path
from time import monotonic
from typing import Protocol, assert_never

import pytest

from ploom import (
    Config,
    Failed,
    ModelCheckError,
    Passed,
    Statistics,
    Trace,
    check,
    read_trace,
    replay,
)

# Windows shells do not read POSIX quotes, so each platform quotes in its own syntax.
join_command_line = subprocess.list2cmdline if os.name == "nt" else shlex.join


class ScenarioCheck(Protocol):
    """Allow ordinary pytest parametrization without a custom scenario DSL."""

    def __call__(self, scenario: Callable[[], object], *, config: Config = ...) -> Statistics:
        """Check a scenario or replay it when requested on the command line."""
        ...


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose artifact placement and explicit replay through standard pytest flags."""
    option_group = parser.getgroup("ploom")
    option_group.addoption(
        "--ploom-trace-dir", default=".ploom", help="ploom failure artifact directory"
    )
    option_group.addoption("--ploom-replay", default=None, help="replay a trace using ploom_check")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reject replay invocations that would silently run an unrelated test."""
    if config.getoption("--ploom-replay") is None:
        return
    match items:
        case [pytest.Function(fixturenames=fixture_names)] if "ploom_check" in fixture_names:
            return
        case _:
            raise pytest.UsageError("--ploom-replay requires exactly one test using ploom_check")


@pytest.fixture
def ploom_check(request: pytest.FixtureRequest) -> ScenarioCheck:
    """Bind test identity and reporting without changing the core check function.

    With ``--ploom-replay``, every call replays the same trace, so a replayed test
    should call the fixture once.
    """
    trace_directory = Path(str(request.config.getoption("--ploom-trace-dir"))).resolve()
    replay_trace_option = request.config.getoption("--ploom-replay")

    def report_original(_trace: Trace, trace_path: Path) -> None:
        match request.config.pluginmanager.get_plugin("terminalreporter"):
            case pytest.TerminalReporter() as terminal_reporter:
                command = join_command_line(
                    ["pytest", request.node.nodeid, f"--ploom-replay={trace_path}"]
                )
                terminal_reporter.write_line(
                    f"ploom first failure: {trace_path}\nReplay: {command}"
                )

    def check_or_replay(scenario: Callable[[], object], *, config: Config = Config()) -> Statistics:
        if replay_trace_option is None:
            return check(
                scenario, config=config, trace_dir=trace_directory, on_failure=report_original
            )
        recorded_trace = read_trace(Path(str(replay_trace_option)))
        started_at = monotonic()
        match replay(scenario, recorded_trace).outcome:
            case Failed(failure):
                pytest.fail(
                    f"ploom reproduced {failure.exception_type}: {failure.message}\n"
                    f"Trace: {replay_trace_option}",
                    pytrace=False,
                )
            case Passed():
                return Statistics(1, len(recorded_trace.steps), monotonic() - started_at)
            case _ as unreachable:
                assert_never(unreachable)

    return check_or_replay


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Attach original and reduced evidence to failed test reports."""
    report = yield
    match call.excinfo:
        case pytest.ExceptionInfo(value=ModelCheckError() as model_check_error):
            trace_path = model_check_error.trace_path
            match item:
                case pytest.Function(fixturenames=fixture_names) if "ploom_check" in fixture_names:
                    command = join_command_line(
                        ["pytest", item.nodeid, f"--ploom-replay={trace_path}"]
                    )
                case _:
                    command = f"ploom.replay(scenario, Path({str(trace_path)!r}))"
            report.sections.append(("ploom", f"{model_check_error}\nReplay: {command}"))
    return report
