"""Importable scenarios exercised through the installed command-line interface."""

import ploom


def passing() -> None:
    ploom.yield_now()


def failing() -> None:
    raise AssertionError("counterexample")


HASH_ORDERED_NAMES = frozenset({"alpha", "beta", "gamma", "delta"})


def set_iteration_order() -> None:
    """Change operation order with PYTHONHASHSEED unless every worker shares one seed."""
    for name in HASH_ORDERED_NAMES:
        if name == "alpha":
            worker = ploom.Thread(target=ploom.yield_now)
            worker.start()
            ploom.yield_now()
            worker.join()
        else:
            ploom.yield_now()


not_callable = 3
