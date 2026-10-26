"""Two threads withdraw from one account. Without a lock, the account goes negative."""

import ploom

WITHDRAWAL_AMOUNT = 100


def scenario() -> None:
    """Withdraw the full balance twice from one account."""
    balance = ploom.SharedCell(WITHDRAWAL_AMOUNT)

    def withdraw() -> None:
        if balance.read() >= WITHDRAWAL_AMOUNT:
            balance.write(balance.read() - WITHDRAWAL_AMOUNT)

    mathieu = ploom.Thread(target=withdraw)
    lucas = ploom.Thread(target=withdraw)
    mathieu.start()
    lucas.start()
    mathieu.join()
    lucas.join()
    assert balance.read() >= 0, "the account is overdrawn"


if __name__ == "__main__":
    ploom.check(scenario)
