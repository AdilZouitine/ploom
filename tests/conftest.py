import shutil
from pathlib import Path

import pytest

collect_ignore = ["fixtures"]


@pytest.fixture
def integration_workspace(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "integration"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    return tmp_path
