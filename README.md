# ploom

[![CI](https://github.com/AdilZouitine/ploom/actions/workflows/ci.yml/badge.svg)](https://github.com/AdilZouitine/ploom/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/AdilZouitine/ploom/graph/badge.svg)](https://codecov.io/gh/AdilZouitine/ploom)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/AdilZouitine/ploom)
[![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14%20%7C%20free--threaded-blue)](https://docs.python.org/3/howto/free-threading-python.html)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Find thread bugs in Python. Replay the schedule that failed.**

A thread bug depends on which thread runs next. A test can pass a thousand times
and still miss the bad order. Ploom runs your test under a controlled scheduler.
It tries every order of ploom operations, stops at the first failure, and saves
that order as a JSON trace. Replay the trace to get the same failure again.

[Documentation](https://adilzouitine.github.io/ploom/) · [Guide](https://adilzouitine.github.io/ploom/guide/) · [Guarantees](https://adilzouitine.github.io/ploom/semantics/) · [Python API](https://adilzouitine.github.io/ploom/api/) · [Development](https://adilzouitine.github.io/ploom/development/)

## Install

Ploom is not on PyPI yet. Install it from GitHub:

```sh
uv add --dev "ploom @ git+https://github.com/AdilZouitine/ploom.git"
```

## Example: two withdrawals from one account

Mathieu and Lucas share an account that holds 100. They both withdraw 100 at the
same time. Only one withdrawal must succeed.

```python
import ploom


def scenario():
    balance = ploom.SharedCell(100)

    def withdraw():
        if balance.read() >= 100:
            balance.write(balance.read() - 100)

    mathieu = ploom.Thread(target=withdraw)
    lucas = ploom.Thread(target=withdraw)
    mathieu.start()
    lucas.start()
    mathieu.join()
    lucas.join()
    assert balance.read() >= 0, "the account is overdrawn"


ploom.check(scenario)
```

`SharedCell` holds a value that the threads share. `ploom.Thread` is a thread
that ploom schedules. With `threading.Thread`, this code passes almost every
time. Ploom tries the possible orders and finds the bad one in a few seconds:

```text
ploom.exploration.ModelCheckError: builtins.AssertionError: the account is overdrawn
Original trace: .ploom/f981…-original.json
Minimized trace: .ploom/f981…-minimized.json
```

The minimized trace shows the bad order:

1. Lucas reads the balance: 100. He can withdraw.
2. Mathieu reads the balance: 100. He can withdraw.
3. Mathieu withdraws. The balance is 0.
4. Lucas withdraws. The balance is -100.

To fix the bug, hold a lock during the check and the withdrawal:

```python
lock = ploom.Lock()

def withdraw():
    with lock:
        if balance.read() >= 100:
            balance.write(balance.read() - 100)
```

Now all 127 orders pass, and `ploom.check` returns the exploration statistics.
[`examples/bank_account.py`](examples/bank_account.py) contains the version
without the lock.

## Use it in pytest

The `ploom_check` fixture explores a scenario inside a test:

```python
def test_withdraw(ploom_check):
    ploom_check(scenario)
```

When the test fails, ploom prints the command that replays the failure:

```text
ploom first failure: .ploom/ID-original.json
Replay: pytest tests/test_bank.py::test_withdraw --ploom-replay=.ploom/ID-original.json
```

## Replay a failure

Replay runs one saved schedule. It does not search. Run the printed command:

```sh
uv run pytest tests/test_bank.py::test_withdraw --ploom-replay=.ploom/ID-original.json
```

A module-level scenario can also use the command line:

```sh
uv run python -m ploom check examples.bank_account:scenario
uv run python -m ploom replay .ploom/ID-minimized.json
```

Replay stops with `ReplayDivergenceError` if the scenario's source file, the
Python minor version, or the recorded operations changed.

## How it works

1. Each ploom operation is a scheduling point. Only one task runs at a time, so
   the result does not depend on the GIL.
2. A depth-first search tries each runnable task at each scheduling point.
3. Each schedule runs in a fresh Python process. Python state from one schedule
   does not leak into the next. All processes use the same hash seed, so set
   order does not change between schedules.
4. At the first failure, ploom saves the trace. Then it looks for the same failure
   with fewer preemptions and saves a minimized trace. A preemption is a switch
   away from a task that could continue.
5. A search limit raises `IncompleteExplorationError`. A limit never counts as a
   pass.

A fresh process costs about 0.1 seconds per schedule. Keep scenarios small. For
larger scenarios, set `Config(max_preemptions=2)` to explore every schedule with
at most two preemptions.

## Scope

| Area | Supported |
| --- | --- |
| Threads | `Thread(target, args, kwargs)`, `start`, `join`, `is_alive` |
| Locks | `Lock`: blocking and nonblocking `acquire`, `release`, `locked`, `with` |
| Shared state | `SharedCell` reads and writes of immutable values, `yield_now()` |
| Entry points | Python API, pytest fixture, `python -m ploom` |
| Interpreters | Standard CPython 3.13 and 3.14, free-threaded CPython 3.14 |

Ploom sees only ploom operations, and it models them as sequentially consistent.
It does not see `threading` locks, `Event`, `Queue`, I/O, or weak-memory effects.
It does not detect data races. Your assertions find the bugs.

Ploom does not roll back files, network calls, or other external effects between
schedules.

## Develop

```sh
uv sync --locked
make check
```

Run `make docs` to build the documentation. The [development guide](https://adilzouitine.github.io/ploom/development/)
covers benchmarks and CI.

## Acknowledgment

Ploom is inspired by [Loom](https://github.com/tokio-rs/loom), the concurrency
testing library for Rust. Loom showed how to explore schedules and reproduce
failures. Ploom defines its own Python operations. It does not implement Loom's
C11 memory model or its partial-order reduction.

## License

Ploom is licensed under [Apache-2.0](LICENSE).
