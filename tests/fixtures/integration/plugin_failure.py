"""An intentional failure used to verify pytest reporting and replay."""

from ploom.pytest_plugin import ScenarioCheck


def test_failure(ploom_check: ScenarioCheck) -> None:
    def scenario() -> None:
        raise AssertionError("original failure")

    ploom_check(scenario)
