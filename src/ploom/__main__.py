"""Typed Tyro commands over the framework-independent checking and replay API."""

import sys
import traceback
from pathlib import Path
from typing import assert_never

from tyro.extras import subcommand_cli_from_dict

from ploom import (
    Config,
    Failed,
    IncompleteExplorationError,
    ModelCheckError,
    Passed,
    ReplayDivergenceError,
    ScenarioTransport,
    check,
    read_trace,
    replay,
)
from ploom.scenarios import ImportableScenario


def import_scenario(reference: str) -> ImportableScenario:
    """Validate an explicitly requested module-level scenario reference.

    Args:
        reference: Scenario location in module:function syntax.

    Returns:
        A callable that preserves the import reference for portable replay.

    Raises:
        ValueError: If the reference has invalid syntax or cannot be loaded.
    """
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name or "." in function_name:
        raise ValueError("scenario must use module:function syntax")
    scenario = ImportableScenario(module=module_name, name=function_name)
    try:
        scenario.resolve()
    except (ImportError, AttributeError, TypeError) as error:
        raise ValueError(f"cannot load scenario {reference}: {error}") from error
    return scenario


def check_command(
    scenario: str, /, config: Path | None = None, trace_dir: Path = Path(".ploom")
) -> int:
    """Explore every modeled schedule until success, failure, or a limit.

    Args:
        scenario: Importable scenario in module:function syntax.
        config: Optional JSON configuration file.
        trace_dir: Directory for original and minimized failure traces.

    Returns:
        Zero after exhaustive success. Failure exceptions reach the CLI boundary.
    """
    limits = Config() if config is None else Config.model_validate_json(config.read_bytes())
    statistics = check(import_scenario(scenario), config=limits, trace_dir=trace_dir)
    print(f"Exhaustive exploration passed: {statistics}")
    return 0


def replay_command(trace: Path, /) -> int:
    """Execute one portable recorded schedule and report its reproduced outcome.

    Args:
        trace: Versioned JSON trace from an importable scenario.

    Returns:
        One for reproduced failure, or zero for reproduced success.

    Raises:
        ValueError: If a local scenario requires its originating test instead.
    """
    recorded_trace = read_trace(trace)
    if recorded_trace.scenario.transport is not ScenarioTransport.IMPORT:
        raise ValueError("local scenarios must replay through their pytest test or Python API")
    reproduced_trace = replay(import_scenario(recorded_trace.scenario.reference), recorded_trace)
    print(f"Replay reproduced {reproduced_trace.outcome.kind}: {trace}")
    match reproduced_trace.outcome:
        case Failed():
            return 1
        case Passed():
            return 0
        case _ as unreachable:
            assert_never(unreachable)


def main(argv: list[str] | None = None) -> int:
    """Generate CLI parsing from typed command functions and Google docstrings.

    Scenario references resolve against the working directory, as with
    ``python -m``. Console scripts do not add it to ``sys.path`` themselves.

    Args:
        argv: Explicit command arguments, or None to use the process arguments.

    Returns:
        Zero for success, one for failure, or two for invalid/incomplete execution.
    """
    working_directory = str(Path.cwd())
    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)
    try:
        return subcommand_cli_from_dict(
            {"check": check_command, "replay": replay_command}, prog="ploom", args=argv
        )
    except ModelCheckError as error:
        print("".join(traceback.format_exception_only(error)), file=sys.stderr, end="")
        print(f"Replay: python -m ploom replay {error.trace_path}", file=sys.stderr)
        return 1
    except (IncompleteExplorationError, ReplayDivergenceError, ValueError, OSError) as error:
        print("".join(traceback.format_exception_only(error)), file=sys.stderr, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
