# Development

The README owns the basic setup and project scope. Use uv with a standard CPython
that satisfies `requires-python` in `pyproject.toml`. Commit `uv.lock` when
dependencies change.

```sh
uv sync --locked
make hooks
make check
uv build
```

`make hooks` installs the [prek](https://prek.j178.dev) Git hook from `prek.toml`.
Each commit then checks file hygiene, formatting, lint, types, and the lockfile.
Run every hook on the whole repository with `uv run --locked prek run --all-files`.

`make check` runs formatting, lint, types, behavior/integration tests, and Google-style
doctests with a 95% statement/branch coverage floor. Tests use real implementations,
not mocks or monkeypatching. Subprocess programs live under `tests/fixtures/`.

For the same suite on free-threaded CPython:

```sh
UV_PROJECT_ENVIRONMENT=.venv-freethreaded uv run --python 3.14t --locked pytest
```

Without Make, run the underlying commands:

```sh
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked ty check
uv run --locked pytest --cov=ploom --cov-report=term-missing
```

## Documentation

Documentation tools are isolated in the `docs` dependency group.

```sh
make docs
uv run --locked --group docs mkdocs serve
```

The strict build produces `site/`. CI builds it and uploads the rendered site as
an artifact. API pages use mkdocstrings' Python handler and the Material theme.

## Performance

```sh
make benchmark
```

The `benchmarks` dependency group contains pytest-codspeed. Benchmarks cover:

- trace processing: branch enumeration, preemption counting, replay comparison;
- schedule search: exhaustive and preemption-bounded `explore`, and `minimize`,
  driven by a pure-Python runner over a synthetic schedule tree;
- trace serialization: trace construction, JSON encoding and decoding, and the
  worker evidence round trip;
- the in-process `ThreadScheduler`: one execution with many hand-offs, and an
  exhaustive exploration of the README example guarded by a lock.

None of them start worker processes. The scheduler benchmarks hand control
between native threads, so their simulated time excludes the time spent in system
calls. CodSpeed runs these workloads in simulation mode on standard Python in
Linux CI.

## CI reporting

The existing Linux/macOS/Windows matrix runs standard CPython 3.13 and 3.14 and
free-threaded CPython 3.14. Hypothesis has no 3.13t wheel, so 3.13t is not tested.
Coverage XML is uploaded to Codecov; failed-run traces are retained
as workflow artifacts. Separate jobs build the docs and run CodSpeed benchmarks.

Enable this repository in Codecov and CodSpeed (including their GitHub apps for
PR reporting). Both workflows use GitHub OIDC instead of stored upload tokens.
Their hosted dashboards cannot be verified by a local test run.

## What the Loom comparison covers

The integration suite takes behavioral scenarios from the neighboring Loom
checkout and adapts them to ploom's existing operations:

| Loom scenario | Ploom coverage |
| --- | --- |
| Mutex exclusion and publication | Protected updates and visibility after lock handoff |
| Two mutexes acquired in opposite order | Reproducible deadlock detection |
| Several joins and joining from another task | Join lifecycle and cross-thread joins |
| Cleanup after a panic or branch limit | Exceptions in `finally` preserve the primary outcome |
| Thread-local/execution isolation | Repeated runs start with fresh Python state |
| Yielding and exploration bounds | Reachable schedules and explicit incompleteness |

Rust-specific Arc ownership, poisoned locks, native atomics, fences, and C11 litmus
tests are outside the supported Python model. They do not justify adding primitives
without first specifying their Python semantics.
