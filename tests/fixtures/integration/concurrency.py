"""Loom-inspired cases expressed with ploom's supported Python operations."""

import ploom


def cross_thread_join() -> None:
    result = ploom.SharedCell(0)
    producer = ploom.Thread(target=lambda: result.write(1))

    def wait_for_producer() -> None:
        producer.join()
        assert result.read() == 1

    observer = ploom.Thread(target=wait_for_producer)
    producer.start()
    observer.start()
    observer.join()
    producer.join()


def mutex_publication() -> None:
    lock = ploom.Lock()
    ready = ploom.SharedCell(False)
    payload = ploom.SharedCell(0)
    expected = 42

    def publish() -> None:
        with lock:
            payload.write(expected)
            ready.write(True)

    producer = ploom.Thread(target=publish)
    producer.start()
    with lock:
        if ready.read():
            assert payload.read() == expected
    producer.join()
    assert payload.read() == expected


def release_after_exception() -> None:
    lock = ploom.Lock()
    completed = ploom.SharedCell(False)

    def recover() -> None:
        try:
            with lock:
                raise ValueError("recoverable error inside the critical section")
        except ValueError:
            completed.write(True)

    worker = ploom.Thread(target=recover)
    worker.start()
    with lock:
        assert lock.locked()
    worker.join()
    assert completed.read()
    assert not lock.locked()


def opposite_lock_order() -> None:
    first = ploom.Lock()
    second = ploom.Lock()

    def lock_in_order() -> None:
        with first, second:
            pass

    worker = ploom.Thread(target=lock_in_order)
    worker.start()
    with second, first:
        pass
    worker.join()
