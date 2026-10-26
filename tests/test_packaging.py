import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    destination = tmp_path_factory.mktemp("distributions")
    subprocess.run(["uv", "build", "--out-dir", str(destination)], check=True, timeout=120)
    return sorted(destination.iterdir())


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_installed_distribution(distributions: list[Path], suffix: str, tmp_path: Path) -> None:
    artifact = next(path for path in distributions if path.name.endswith(suffix))
    environment = tmp_path / "environment"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)], check=True, timeout=30
    )
    interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(interpreter), str(artifact)],
        check=True,
        timeout=120,
    )
    probe = Path(__file__).parent / "fixtures" / "installed_package_probe.py"
    result = subprocess.run(
        [str(interpreter), "-I", str(probe)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    installed = json.loads(result.stdout)
    assert installed["version"] == "0.1.0"
    assert environment.resolve() in Path(installed["file"]).resolve().parents
