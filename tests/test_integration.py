import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import run_ploom


@pytest.mark.parametrize(
    "scenario", ["cross_thread_join", "mutex_publication", "release_after_exception"]
)
def test_supported_concurrency_scenarios_pass_exhaustively(
    integration_workspace: Path, scenario: str
) -> None:
    result = run_ploom(integration_workspace, "check", f"concurrency:{scenario}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exhaustive exploration passed" in result.stdout


@pytest.mark.parametrize("parent_seed", ["1", "2", "3", "4"])
def test_workers_share_one_hash_seed_whatever_the_parent_uses(
    integration_workspace: Path, parent_seed: str
) -> None:
    result = run_ploom(
        integration_workspace,
        "check",
        "scenarios:set_iteration_order",
        environment={"PYTHONHASHSEED": parent_seed},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "executions=3," in result.stdout


def test_opposite_lock_order_deadlock_replays(integration_workspace: Path) -> None:
    failure = run_ploom(integration_workspace, "check", "concurrency:opposite_lock_order")
    assert failure.returncode == 1, failure.stdout + failure.stderr
    assert "ploom.DeadlockError" in failure.stderr
    original = next((integration_workspace / ".ploom").glob("*-original.json"))
    reproduced = run_ploom(integration_workspace, "replay", str(original))
    assert reproduced.returncode == 1, reproduced.stdout + reproduced.stderr
    assert "reproduced failed" in reproduced.stdout


def test_python_state_is_fresh_but_external_effects_persist(integration_workspace: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(integration_workspace / "side_effects.py")],
        cwd=integration_workspace,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["executions"] > 1
    assert report["external_writes"] == report["executions"]
    assert report["parent_starts"] == 0
