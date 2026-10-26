"""Python state is isolated between executions; filesystem effects persist."""

import json
from pathlib import Path

import ploom

starts: list[int] = []


def scenario() -> None:
    assert not starts
    starts.append(1)
    with Path("executions.log").open("a", encoding="utf-8") as output:
        output.write("execution\n")
    worker = ploom.Thread(target=ploom.yield_now)
    worker.start()
    ploom.yield_now()
    worker.join()


def main() -> None:
    first = ploom.check(scenario)
    second = ploom.check(scenario)
    assert not starts
    executions = first.executions + second.executions
    lines = len(Path("executions.log").read_text(encoding="utf-8").splitlines())
    assert lines == executions
    print(
        json.dumps(
            {"executions": executions, "external_writes": lines, "parent_starts": len(starts)}
        )
    )


if __name__ == "__main__":
    main()
