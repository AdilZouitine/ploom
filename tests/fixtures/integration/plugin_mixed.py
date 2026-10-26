"""Exercise the fixture and direct Python API in the same pytest session."""

import ploom
from ploom.pytest_plugin import ScenarioCheck


def test_pass(ploom_check: ScenarioCheck) -> None:
    def scenario() -> None:
        ploom.yield_now()

    assert ploom_check(scenario).executions == 1


def test_direct() -> None:
    def scenario() -> None:
        raise AssertionError("direct API failure")

    ploom.check(scenario)
