# -*- coding: utf-8 -*-

"""
HomeWork Task 1
"""

import argparse
import collections
import time
from typing import Optional, Any, Callable


def timed_call(
        _func_: Callable,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[Any, Any]] = None,
        executions_number: int = 5,) -> tuple[Any, float]:
    """
    Execution of the specified function with measurement of execution time.

    :param _func_: specified function (Callable, mandatory)
    :param args: positional arguments (Tuple of Any, optional)
    :param kwargs: keyword arguments (Dictionary of Any, optional)
    :param executions_number: number of runs (Integer, optional)
    :return: The result returned by the specified function and the average execution time based on the number of runs
    """
    if not callable(_func_):
        raise ValueError("The function must be an callable object")
    start_time: float = time.perf_counter()
    func_result: Any = None
    for _ in range(executions_number or 1):
        func_result = _func_(*(args or ()), **(kwargs or {}))
    duration_time: float = round((time.perf_counter() - start_time) / (executions_number or 1), 8)
    return func_result, duration_time


def find_coins_greedy(coins: list[int], amount: int) -> dict[int, int]:
    """
    Determining the optimal way to give change to the customer (Greedy algorithm).

    :param coins: available coins (List of Integer, mandatory)
    :param amount: amount of change (Integer, mandatory)
    :return: Change distributed by coins (Dictionary of coins (Integer) and count (Integer))
    """

    change_coins: dict[int, int] = {}

    # Sort the coins in reverse order (from the largest denomination to the smallest).
    coins_sorted: list[int] = sorted(coins, reverse=True)

    # Check whether the coin can be used for the change
    for coin in coins_sorted:
        count = amount // coin
        if count > 0:
            change_coins[coin] = count
            amount -= coin * count
        if amount < coins_sorted[-1]:
            break

    if amount > 0:
        raise ValueError("It is not possible to give change with the given coins")

    # Return the change distributed by coins
    return change_coins


def find_min_coins(coins: list[int], amount: int) -> dict[int, int]:
    """
    Determining the optimal way to give change to the customer (Dynamic programming method).

    :param coins: available coins (List of Integer, mandatory)
    :param amount: amount of change (Integer, mandatory)
    :return: Change distributed by coins (Dictionary of coins (Integer) and count (Integer))
    """

    # Sort the coins in reverse order (from the largest denomination to the smallest).
    coins_sorted: list[int] = sorted(coins, reverse=True)

    # The minimum number of coins to make up the amount, for each amount
    min_coins_required: list[int|float] = [0] + [float('inf')] * amount
    # The largest coin used to make up the amount, for each amount
    max_coin_used: list[int] = [0] * (amount + 1)

    # Calculate minimum number of coins and the largest coin used to make up the amount, for each amount
    for amt in range(1, amount + 1):
        for coin in coins_sorted:
            if amt >= coin and min_coins_required[amt - coin] + 1 < min_coins_required[amt]:
                min_coins_required[amt] = min_coins_required[amt - coin] + 1
                max_coin_used[amt] = coin

    if min_coins_required[amount] == float("inf"):
        raise ValueError("It is not possible to give change with the given coins")

    # Construct the change distributed by coins
    change_coins: dict[int, int] = collections.defaultdict(int)
    while amount > 0:
        coin = max_coin_used[amount]
        change_coins[coin] += 1
        amount -= coin

    # Return the change distributed by coins
    return dict(change_coins)

def give_change_to_customer(amount: int) -> None:
    coins: list[int] = [50, 25, 10, 5, 2, 1]

    print(f"Calculating the optimal way to give the customer change of {amount}")
    print(f"Coins available: {coins}")

    print("Change by Greedy algorithm: {0} with average time {1:.8f} seconds".format(
        *timed_call(find_coins_greedy, args=(coins, amount), executions_number=10))
    )

    print("Change by Dynamic programming method: {0} with average time {1:.8f} seconds".format(
        *timed_call(find_min_coins, args=(coins, amount), executions_number=10))
    )

def cli() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="Determine the optimal way to give change to the customer",
            epilog="Good bye!")
        parser.add_argument("-c", "--change", type=int, default=113, help="Amount of change (Default 113)")

        args = parser.parse_args()

        give_change_to_customer(args.change)
    except Exception as e:
        print(e)

    exit(0)
