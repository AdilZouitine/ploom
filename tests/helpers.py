import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from ploom.execution import Execution, Failed, Failure


def run_ploom(
    directory: Path, *arguments: str, environment: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ploom", *arguments],
        cwd=directory,
        env={**os.environ, **(environment or {})},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def failure_of(execution: Execution) -> Failure:
    match execution.outcome:
        case Failed(failure):
            return failure
        case outcome:
            pytest.fail(f"expected a failure, observed {outcome}")
