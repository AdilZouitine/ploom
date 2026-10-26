from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from queue import Empty, Full

import pytest

import ploom
from ploom.execution import Execution, FailureCategory, Passed
from ploom.exploration import explore
from ploom.thread_scheduler import ThreadScheduler
from tests.helpers import failure_of


class PrimitiveRunner:
    def __init__(self, scenario: Callable[[], object]) -> None:
        self.scenario = scenario

    def run(
        self,
        required_prefix: tuple[int, ...],
        _timeout_seconds: float,
        expected_execution: Execution | None = None,
        /,
    ) -> Execution:
        return ThreadScheduler(300, 8).execute(self.scenario, required_prefix, expected_execution)


def exhaustive(scenario: Callable[[], object]) -> None:
    result = explore(
        PrimitiveRunner(scenario),
        ploom.Config(max_executions=100_000, timeout=120.0),
    )
    assert result.execution.outcome == Passed(), result.execution


def test_event_single_task_contract_and_timeouts() -> None:
    def scenario() -> None:
        event = ploom.Event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
        assert event.wait()
        event.clear()
        assert not event.is_set()
        with pytest.raises(NotImplementedError):
            event.wait(0)

    exhaustive(scenario)


def test_event_wait_notification_survives_clear_exact_schedule() -> None:
    def scenario() -> None:
        event = ploom.Event()

        def signal() -> None:
            event.set()
            event.clear()

        worker = ploom.Thread(target=signal)
        worker.start()
        assert event.wait()
        worker.join()

    runner = PrimitiveRunner(scenario)
    # Root begins/starts worker/registers wait; worker begins/sets/clears and
    # finishes before root resumes its latched notification.
    execution = runner.run((0, 0, 0, 1, 1, 1), 1)
    assert execution.outcome == Passed()
    assert runner.run(tuple(step.task for step in execution.steps), 1, execution) == execution


def test_rlock_recursion_and_nonowner_contract() -> None:
    def scenario() -> None:
        lock = ploom.RLock()
        assert not lock.locked()
        with pytest.raises(RuntimeError):
            lock.release()
        with lock:
            assert lock.acquire(False)
            assert lock.locked()

            def contender() -> None:
                assert not lock.acquire(False)
                with pytest.raises(RuntimeError):
                    lock.release()

            worker = ploom.Thread(target=contender)
            worker.start()
            worker.join()
            lock.release()
        assert not lock.locked()
        with pytest.raises(NotImplementedError):
            lock.acquire(timeout=0)

    exhaustive(scenario)


@pytest.mark.parametrize("bounded", [False, True])
def test_semaphore_permits_and_nonblocking(bounded: bool) -> None:
    def scenario() -> None:
        semaphore = ploom.BoundedSemaphore(2) if bounded else ploom.Semaphore(2)
        assert semaphore.acquire(False)
        assert semaphore.acquire(False)
        assert not semaphore.acquire(False)
        semaphore.release(2)
        with semaphore:
            assert semaphore.acquire(False)
            semaphore.release()
        with pytest.raises(NotImplementedError):
            semaphore.acquire(timeout=0)
        for invalid in (0, -1):
            with pytest.raises(ValueError, match="positive integer"):
                semaphore.release(invalid)
        if bounded:
            with pytest.raises(ValueError, match="too many"):
                semaphore.release()

    exhaustive(scenario)
    with pytest.raises(ValueError, match="initial value"):
        ploom.Semaphore(-1)


def test_semaphore_blocked_acquisition() -> None:
    def scenario() -> None:
        semaphore = ploom.Semaphore(0)
        worker = ploom.Thread(target=semaphore.release)
        worker.start()
        assert semaphore.acquire()
        worker.join()
        assert not semaphore.acquire(False)

    exhaustive(scenario)


@pytest.mark.parametrize("basic_lock", [False, True])
def test_condition_wait_for_releases_and_restores_lock(basic_lock: bool) -> None:
    def scenario() -> None:
        lock = ploom.Lock() if basic_lock else ploom.RLock()
        condition = ploom.Condition(lock)
        ready = ploom.SharedCell(False)

        def produce() -> None:
            with condition:
                ready.write(True)
                condition.notify_all()

        with condition:
            if not basic_lock:
                condition.acquire()
            worker = ploom.Thread(target=produce)
            worker.start()
            assert condition.wait_for(ready.read)
            if not basic_lock:
                condition.release()
        worker.join()
        assert not lock.locked()

    exhaustive(scenario)


def test_condition_contract_and_notify_without_waiters() -> None:
    def scenario() -> None:
        condition = ploom.Condition()
        with pytest.raises(RuntimeError):
            condition.wait()
        with pytest.raises(RuntimeError):
            condition.notify()
        with pytest.raises(NotImplementedError):
            condition.wait(timeout=0)
        with pytest.raises(NotImplementedError):
            condition.wait_for(lambda: True, timeout=0)
        with condition:
            condition.notify()
            condition.notify(-1)
            condition.notify_all()
            assert condition.wait_for(lambda: "ready") == "ready"

    exhaustive(scenario)


def test_queue_fifo_capacity_and_task_tracking() -> None:
    def scenario() -> None:
        capacity = 2
        queue = ploom.Queue[int](capacity)
        assert queue.empty()
        assert not queue.full()
        assert queue.qsize() == 0
        with pytest.raises(Empty):
            queue.get_nowait()
        queue.put_nowait(1)
        second = 2
        queue.put(second)
        assert queue.full()
        assert not queue.empty()
        assert queue.qsize() == capacity
        with pytest.raises(Full):
            queue.put_nowait(3)
        assert queue.get_nowait() == 1
        assert queue.get() == second
        queue.task_done()
        queue.task_done()
        queue.join()
        with pytest.raises(ValueError, match="too many"):
            queue.task_done()
        with pytest.raises(NotImplementedError):
            queue.put(0, timeout=0)
        with pytest.raises(NotImplementedError):
            queue.get(timeout=0)

    exhaustive(scenario)


def test_queue_producer_consumer_and_join() -> None:
    def scenario() -> None:
        queue = ploom.Queue[int](1)

        second = 2

        def consume() -> None:
            assert queue.get() == 1
            queue.task_done()
            assert queue.get() == second
            queue.task_done()

        worker = ploom.Thread(target=consume)
        worker.start()
        queue.put(1)
        second = 2
        queue.put(second)
        queue.join()
        worker.join()

    exhaustive(scenario)


def test_queue_unbounded_and_immutable_payloads() -> None:
    def scenario() -> None:
        queue = ploom.Queue[tuple[int, str]]()
        queue.put((1, "first"))
        queue.put((2, "second"))
        assert not queue.full()
        assert queue.get() == (1, "first")

    exhaustive(scenario)


@pytest.mark.parametrize(
    "scenario",
    [
        lambda: ploom.Event().wait(),
        lambda: ploom.Semaphore(0).acquire(),
        lambda: ploom.Queue[int]().get(),
    ],
)
def test_blocked_primitives_report_deadlock_and_replay(
    scenario: Callable[[], object], tmp_path: Path
) -> None:
    with pytest.raises(ploom.DeadlockError) as caught:
        ploom.check(scenario, trace_dir=tmp_path)
    assert ploom.replay(scenario, caught.value.trace_path) == caught.value.trace


def test_condition_wait_cancellation_unregisters_waiter() -> None:
    def scenario() -> None:
        condition = ploom.Condition()
        with condition:
            condition.wait()

    execution = PrimitiveRunner(scenario).run((), 1)
    assert failure_of(execution).category is FailureCategory.DEADLOCK


def test_condition_fifo_notify_then_notify_all() -> None:
    def scenario() -> None:
        condition = ploom.Condition()
        registered = ploom.Semaphore(0)
        awakened = ploom.Semaphore(0)
        next_ticket = ploom.SharedCell(0)
        first_awake = ploom.SharedCell(-1)

        def wait() -> None:
            with condition:
                ticket = next_ticket.read()
                next_ticket.write(ticket + 1)
                registered.release()
                condition.wait()
                if first_awake.read() == -1:
                    first_awake.write(ticket)
                awakened.release()

        first = ploom.Thread(target=wait)
        second = ploom.Thread(target=wait)
        first.start()
        second.start()
        registered.acquire()
        registered.acquire()
        with condition:
            condition.notify(1)
        awakened.acquire()
        assert first_awake.read() == 0
        assert not awakened.acquire(False)
        with condition:
            condition.notify_all()
        awakened.acquire()
        first.join()
        second.join()

    result = explore(
        PrimitiveRunner(scenario),
        ploom.Config(max_executions=100_000, timeout=120.0, max_preemptions=1),
    )
    assert result.execution.outcome == Passed(), result.execution


def test_rlock_blocking_and_exception_cleanup() -> None:
    def scenario() -> None:
        lock = ploom.RLock()
        lock.acquire()

        def acquire() -> None:
            with lock:
                assert lock.locked()
                with pytest.raises(ArithmeticError), lock:
                    raise ArithmeticError("abort inner critical section")
                assert lock.locked()

        worker = ploom.Thread(target=acquire)
        worker.start()
        lock.release()
        worker.join()
        assert not lock.locked()

    exhaustive(scenario)


def test_queue_join_waits_for_acknowledgement_after_get() -> None:
    def scenario() -> None:
        queue = ploom.Queue[int]()
        dequeued = ploom.Event()
        acknowledged = ploom.SharedCell(False)
        queue.put(1)

        def consume() -> None:
            assert queue.get() == 1
            dequeued.set()
            ploom.yield_now()
            acknowledged.write(True)
            queue.task_done()

        worker = ploom.Thread(target=consume)
        worker.start()
        dequeued.wait()
        assert queue.empty()
        queue.join()
        assert acknowledged.read()
        worker.join()

    exhaustive(scenario)


def test_queue_nonblocking_consumer_race() -> None:
    def scenario() -> None:
        queue = ploom.Queue[int](1)
        consumed = ploom.SharedCell(False)

        def consume() -> None:
            try:
                assert queue.get_nowait() == 1
            except Empty:
                pass
            else:
                consumed.write(True)
                queue.task_done()

        worker = ploom.Thread(target=consume)
        worker.start()
        queue.put_nowait(1)
        worker.join()
        if consumed.read():
            assert queue.empty()
        else:
            assert queue.get_nowait() == 1
            queue.task_done()
        queue.join()

    exhaustive(scenario)


def test_condition_rejects_cross_execution_lock() -> None:
    locks: list[ploom.Lock] = []

    def allocate() -> None:
        locks.append(ploom.Lock())

    assert PrimitiveRunner(allocate).run((), 1).outcome == Passed()

    def foreign_lock() -> None:
        with pytest.raises(RuntimeError, match="different execution"):
            ploom.Condition(locks[0])

    exhaustive(foreign_lock)


class MutableInt(int):
    values: list[int]


def test_queue_rejects_payload_with_mutable_aliases() -> None:
    value = MutableInt(1)
    value.values = []

    def scenario() -> None:
        queue = ploom.Queue[int]()
        with pytest.raises(TypeError, match="immutable"):
            queue.put(value)
        assert queue.empty()
        queue.join()

    exhaustive(scenario)


def test_queue_nonblocking_producers_race_for_capacity() -> None:
    def scenario() -> None:
        queue = ploom.Queue[int](1)
        second = 2

        def produce() -> None:
            with suppress(Full):
                queue.put_nowait(second)

        worker = ploom.Thread(target=produce)
        worker.start()
        with suppress(Full):
            queue.put_nowait(1)
        worker.join()
        assert queue.get_nowait() in (1, second)
        assert queue.empty()
        queue.task_done()
        queue.join()

    exhaustive(scenario)
