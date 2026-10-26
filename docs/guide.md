# Checking and replay

## Python API

`ploom.check(scenario)` returns `Statistics` only after every reachable modeled
schedule passes. Scenarios are ordinary zero-argument functions. Use closures,
`functools.partial`, or pytest parametrization for inputs.

| Result | Python behavior |
| --- | --- |
| Every schedule passes | Return exploration statistics |
| Assertion or task exception | Raise `ModelCheckError` |
| Unfinished tasks with none runnable | Raise `DeadlockError` |
| Search or execution limit | Raise `IncompleteExplorationError` |
| Inconsistent replay or scheduling prefix | Raise `ReplayDivergenceError` |

Failures carry the original trace, artifact path, best reduced trace, reduction
status, and statistics. The message holds only the primary failure. Cleanup
failures are exception notes, also when a limit ends execution. All ploom errors
can be pickled.

After a failure is found, ploom publishes the original trace, reports it, then
minimizes it. An error in these steps propagates unchanged: for example a
reporter `OSError`, a worker crash, or a nondeterministic scenario. Its first
exception note holds the modeled failure and its traceback, and the original
trace stays on disk.

The failure traceback includes the scenario's output from that execution, so
`print` debugging works. Ploom shows output from passing executions only when a
worker crashes.

The public types in `ploom.Trace` are exported from `ploom`. These include
`Passed`, `Failed`, `Failure`, `Step`, and `Provenance`. You do not need private
modules to inspect a trace.

## Pytest

The installed plugin supplies a `ploom_check` fixture. It reports the original
failure before minimization and prints a command that reruns the same test:

```python
from ploom.pytest_plugin import ScenarioCheck


def test_concurrency(ploom_check: ScenarioCheck) -> None:
    def scenario() -> None:
        # Build primitives, run tasks, and assert the result here.
        pass

    ploom_check(scenario)
```

```sh
uv run pytest tests/test_example.py::test_concurrency --ploom-replay=.ploom/ID-original.json
```

Replay requires exactly one selected test using the fixture. During replay, each
fixture call replays the same trace, so call `ploom_check` once in that test.
Module-level scenarios in test files work in pytest's default import mode,
because each worker gets the parent's `sys.path`. Set
`--ploom-trace-dir=PATH` to choose the artifact directory. The direct Python API
also accepts `trace_dir` and an `on_failure(trace, path)` reporter. The default
reporter logs a warning with the path.

## CLI

Importable scenarios use `module:function` references:

```sh
uv run ploom check example:scenario
uv run ploom check example:scenario --config limits.json
uv run ploom replay .ploom/ID-original.json
```

`python -m ploom` provides the same commands. Exit codes are 0 for success, 1 for
a discovered or reproduced failure, and 2 for invalid input, divergence, or an
incomplete search. A scenario that cannot be imported is invalid input. Other
unexpected errors keep their full traceback. Run `ploom check --help` for
arguments.

## Limits

All limits are configurable through `ploom.Config` or the CLI's JSON config file.

| Limit | Default |
| --- | --- |
| Executions | 200 |
| Modeled steps per execution | 1,000 |
| Concurrent tasks, including the root | 16 |
| Exploration wall time, including worker startup | 30 seconds |
| Preemptions per schedule (`max_preemptions`) | Unbounded |
| Worker `PYTHONHASHSEED` (`hash_seed`) | 0 |
| Additional minimization executions | 100 |
| Additional minimization wall time | 2 seconds |

```python
limits = ploom.Config(max_executions=5000, timeout=120.0)
ploom.check(scenario, config=limits)
```

Reaching a limit raises an error. A fresh process per execution favors isolation
over throughput. Each worker spends about 50 ms importing ploom and pydantic,
and a small program can still have many schedules.

Most concurrency bugs need only a few preemptions. Set `max_preemptions` to
explore every schedule with at most that many preemptions. The number of these
schedules grows polynomially, not exponentially. A pass with this bound covers
only the schedules within the bound:

```python
ploom.check(scenario, config=ploom.Config(max_preemptions=2))
```

Every worker uses the same `hash_seed`. Without one seed, `str` hashing and set
order would change between processes. A scenario that iterates over a set could
then change its operations between schedules and make the search skip branches.
The seed is stored in the trace configuration, and replay uses it.

## Replay and artifacts

`ploom.replay(scenario, trace_path)` runs exactly one recorded schedule. When the
schedule reproduces, it returns the trace, including its failure, and does not
raise that failure again. These must match: the scenario reference and source
digest, the Python minor version, the runnable choices, the operation identities,
and the terminal failure identity. A trace holds only a passed or failed outcome.
A limit or a divergence is not replayable evidence.

JSON traces contain decisions, operation identities, and failure evidence—not
full program state or executable pickle data. Use the same source and dependency
environment for reproduction. Importable scenarios are portable. A local closure
is identified by its qualified name and the digest of its defining source file.
Captured values such as `tmp_path` do not affect this identity. If changed inputs
alter the schedule, replay detects it when it compares each operation.
Cloudpickle transport is temporary and intended for trusted local code.

Minimization looks for the same failure with fewer preemptions. A preemption is a
switch away from a task that could continue. Two failures are the same if they
have the same category, exception type, and location. The location is the
innermost scenario frame, not a frame in ploom or a library. For a deadlock, the
location is the set of blocked tasks and their operations. Minimization has its
own budget, and it visits only schedules with fewer preemptions than the best so
far. Status `completed` means that no schedule with fewer preemptions reproduces
the failure. `budget_exhausted` means that the budget stopped the search first.
The other status is `disabled`. A nondeterministic scenario raises
`ReplayDivergenceError` during minimization.
