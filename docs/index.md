# Find the ordering your test missed

Ploom runs ordinary Python functions under a controlled thread scheduler. Instead
of hoping that a stress test hits a bad ordering, you ask ploom to try the choices
at each modeled operation.

## Install

```sh
uv add --dev ploom
# or
pip install ploom
```

Scenarios can be module-level functions, local functions, closures, or partials.
Pytest is needed only for test integration, not for the Python API or CLI.

## Protect the whole operation

A shared cell makes each read and write atomic. It does **not** make a read followed
by a write atomic. Use a lock around the entire increment:

```python
import ploom


def scenario() -> None:
    counter = ploom.SharedCell(0)
    lock = ploom.Lock()

    def increment() -> None:
        with lock:
            counter.write(counter.read() + 1)

    worker = ploom.Thread(target=increment)
    worker.start()
    increment()
    worker.join()
    assert counter.read() == 2


if __name__ == "__main__":
    print(ploom.check(scenario))
```

Save this as `example.py`, then run:

```sh
uv run python example.py
# Or use the same scenario through the CLI:
uv run ploom check example:scenario
```

Remove the lock to expose the lost update. The repository's
`examples/bank_account.py` fails on purpose with the same kind of bug.

Assertions belong **inside** the scenario. Each execution runs in a fresh process;
its Python objects do not return to the calling test.

Continue with [checking and replay](guide.md), the [guarantees](semantics.md), or
the generated [Python API](api.md).
