"""Prepare trusted scenario code and preserve its portable replay provenance."""

import hashlib
import importlib
import pkgutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Literal, Protocol, assert_never, runtime_checkable

import cloudpickle
from pydantic import BaseModel, ConfigDict

from ploom.trace import Provenance, ScenarioTransport

PYTHON_VERSION = "{}.{}".format(*sys.version_info[:2])


@runtime_checkable
class ScenarioFunction(Protocol):
    """A scenario is an ordinary callable executed without arguments.

    A class pattern checks only that ``__call__`` exists, like ``callable()``, but
    narrows to a zero-argument signature that ``callable()`` cannot express.
    """

    def __call__(self) -> object:
        """Execute assertions and modeled operations within an execution."""
        ...


def require_callable(candidate: object) -> ScenarioFunction:
    """Reject scenario sources that do not produce executable code.

    Args:
        candidate: An imported symbol or deserialized object.

    Returns:
        The candidate, narrowed to a callable scenario.

    Raises:
        TypeError: If the candidate is not callable.
    """
    match candidate:
        case ScenarioFunction():
            return candidate
        case _:
            raise TypeError("scenario must be callable")


class ImportableScenario(BaseModel):
    """Keep a module reference callable without losing its import provenance.

    Attributes:
        module: Importable module containing the scenario.
        name: The scenario's module-level symbol.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal[ScenarioTransport.IMPORT] = ScenarioTransport.IMPORT
    module: str
    name: str

    def resolve(self) -> ScenarioFunction:
        """Import the module symbol and require it to be callable.

        Returns:
            The scenario function named by this reference.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the module has no such symbol.
            TypeError: If the imported symbol is not callable.
        """
        return require_callable(pkgutil.resolve_name(f"{self.module}:{self.name}"))

    def __call__(self) -> object:
        """Resolve and invoke the module symbol in the isolated process.

        Returns:
            The scenario's return value, which exploration ignores.
        """
        return self.resolve()()


class SerializedScenario(BaseModel):
    """Locate temporary executable input; replay JSON never embeds this data."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal[ScenarioTransport.LOCAL] = ScenarioTransport.LOCAL
    path: str


type ScenarioSource = ImportableScenario | SerializedScenario


@dataclass(frozen=True)
class PreparedScenario:
    """Separate durable replay identity from disposable executable transport."""

    provenance: Provenance
    source: ScenarioSource


def importable_reference(scenario: Callable[[], object]) -> ImportableScenario | None:
    """Name a scenario that a fresh process can import by module and symbol."""
    match scenario:
        case ImportableScenario():
            return scenario
        case FunctionType() if scenario.__qualname__ == scenario.__name__:
            module = sys.modules.get(scenario.__module__)
            if module is not None and vars(module).get(scenario.__name__) is scenario:
                return ImportableScenario(module=scenario.__module__, name=scenario.__name__)
    return None


def prepare_scenario(scenario: Callable[[], object], working_directory: Path) -> PreparedScenario:
    """Prefer module imports and serialize local callables only when needed.

    A local function's digest covers its defining source file, not its pickle.
    Pickles embed captured values such as ``tmp_path`` and hash-ordered sets that
    change on every run. Replay compares every operation, so changed inputs that
    alter the schedule still diverge.

    Args:
        scenario: Scenario code supplied by the Python API or CLI.
        working_directory: Owned temporary directory for trusted serialized code.

    Returns:
        The scenario's replay provenance and worker transport.
    """
    reference = importable_reference(scenario)
    if reference is not None:
        module = importlib.import_module(reference.module)
        match vars(module).get("__file__"):
            case str() as module_file if module.__name__ != "__main__":
                return PreparedScenario(
                    Provenance(
                        reference=f"{reference.module}:{reference.name}",
                        digest=hashlib.sha256(Path(module_file).read_bytes()).hexdigest(),
                        transport=ScenarioTransport.IMPORT,
                        python=PYTHON_VERSION,
                    ),
                    reference,
                )
    pickled_scenario = bytes(cloudpickle.dumps(scenario))
    pickle_path = working_directory / "scenario.pickle"
    pickle_path.write_bytes(pickled_scenario)
    match scenario:
        case FunctionType() if Path(scenario.__code__.co_filename).is_file():
            local_reference = f"{scenario.__module__}:{scenario.__qualname__}"
            identity_source = Path(scenario.__code__.co_filename).read_bytes()
        case _:
            local_reference = "local scenario"
            identity_source = pickled_scenario
    return PreparedScenario(
        Provenance(
            reference=local_reference,
            digest=hashlib.sha256(identity_source).hexdigest(),
            transport=ScenarioTransport.LOCAL,
            python=PYTHON_VERSION,
        ),
        SerializedScenario(path=str(pickle_path)),
    )


def load_scenario(source: ScenarioSource) -> ScenarioFunction:
    """Resolve executable transport before the scheduler receives a callable.

    A scenario that cannot load is an environment error, not a modeled failure,
    so this must run outside any modeled task.

    Args:
        source: A validated import reference or trusted serialized-code location.

    Returns:
        The callable that the worker will execute.

    Raises:
        ImportError: If the scenario module cannot be imported.
        AttributeError: If the module has no such symbol.
        TypeError: If the source does not contain a callable.
    """
    match source:
        case ImportableScenario():
            return source.resolve()
        case SerializedScenario(path=pickle_path):
            return require_callable(cloudpickle.loads(Path(pickle_path).read_bytes()))
        case _ as unreachable:
            assert_never(unreachable)
