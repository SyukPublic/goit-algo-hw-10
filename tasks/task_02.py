# -*- coding: utf-8 -*-

"""
HomeWork Task 2
"""

from typing import Callable
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi


def plot_integrate_function(func: Callable, a: float, b: float) -> None:
    """
    Plotting the given function for integral calculation.

    :param func: The given function for integral calculation (Callable, mandatory)
    :param a: Lower limit of integration (Float, mandatory)
    :param b: Upper limit of integration (Float, mandatory)
    """

    if not callable(func):
        raise ValueError("The function must be an callable object")

    # Creating a range of values for x
    x = np.linspace(-0.5, 2.5, 400)
    y = func(x)

    # Creating a plot
    fig, ax = plt.subplots()

    # Plotting the function
    ax.plot(x, y, 'r', linewidth=2)

    # Filling the area under the curve
    ix = np.linspace(a, b)
    iy = func(ix)
    ax.fill_between(ix, iy, color='gray', alpha=0.3)

    # Setting up the plot
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(0, max(y) + 0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    # Adding the integration limits and the plot title
    ax.axvline(x=a, color='gray', linestyle='--')
    ax.axvline(x=b, color='gray', linestyle='--')
    ax.set_title('Графік інтегрування f(x) від ' + str(a) + ' до ' + str(b))
    plt.grid()

    plt.show()


def integrate_by_monte_carlo(func: Callable, a: float, b: float, points_number: int = 10_000) -> tuple[float, float]:
    """
    Calculation of the integral of a function using the Monte Carlo method.

    :param func: The given function for integral calculation (Callable, mandatory)
    :param a: Lower limit of integration (Float, mandatory)
    :param b: Upper limit of integration (Float, mandatory)
    :param points_number: Number of random points for integral calculation (Int, optional)
    :return: Value of the integral and standard error value (Tuple of Float)
    """

    if not callable(func):
        raise ValueError("The function must be an callable object")

    x = np.random.uniform(a, b, points_number)
    y_min, y_max = np.min(func(x)), np.max(func(x))
    y = np.random.uniform(y_min, y_max, points_number)
    under_curve_points_number = np.sum(y < func(x))

    # Estimated integral value
    value = float((b - a) * (y_max - y_min) * (under_curve_points_number / points_number))
    # Standard error of the mean estimate * interval width
    se = float((b - a) * func(x).std(ddof=1) / np.sqrt(points_number))
    return value, se


def f(x: float) -> float:
    """
    Integration function
    """
    return x ** 2


def test_integrate() -> None:
    a = 0.0
    b = 2.0

    plot_integrate_function(f, a, b)

    # Integration using scipy.integrate.quad
    print("#", "=" * 100, "#")
    print("Etalon integral value calculated using scipy.integrate.quad")
    etalon_integral_value, etalon_se_value = spi.quad(f, a, b)
    print(
        f"Result of integration using scipy.integrate.quad: {etalon_integral_value} "
        f"(estimate of the absolute error: {etalon_se_value:.15f})"
    )
    print("#", "=" * 100, "#")

    # Integration using Monte Carlo method
    experiments_number = 100
    for points_number in [10, 100, 1_000, 10_000, 100_000]:
        integral_value, se_value = integrate_by_monte_carlo(f, a, b, points_number=points_number)
        print(
            f"Result of integration using Monte Carlo method: {integral_value} "
            f"(estimate of the absolute error: {se_value:.15f})"
            f" - number of random points {points_number}, number of experiments 1"
        )

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(integrate_by_monte_carlo, f, a, b, points_number=points_number)
                for _ in range(experiments_number)
            ]
            integral_values: list[float] = [future.result()[0] for future in futures]
            se_values: list[float] = [future.result()[1] for future in futures]
            integral_value = np.mean(integral_values)
            se_value = np.mean(se_values)/np.sqrt(experiments_number)

            print(
                f"Result of integration using Monte-Carlo method: {integral_value} "
                f"(estimate of the absolute error: {se_value:.15f})"
                f"(number of random points {points_number}, number of experiments {experiments_number})"
            )

    # Visualization of the dependence of the integral value calculation by the Monte Carlo method on the number of points
    integral_values: list[float] = []
    se_values: list[float] = []
    points_numbers = np.unique(np.logspace(np.log10(10), np.log10(1_000_000), num=1_000).astype(int))
    for points_number in points_numbers:
        integral_value, se_value = integrate_by_monte_carlo(f, a, b, points_number=points_number)
        integral_values.append(integral_value)
        se_values.append(se_value)

    plt.figure(figsize=(16, 6))
    plt.errorbar(points_numbers, integral_values, yerr=se_values, fmt="o-", capsize=5, label="MC integral value ± SE")
    plt.axhline(etalon_integral_value, color="red", linestyle="--", label="scipy.integrate.quad integral value")
    plt.xscale("log")
    plt.xlabel("N (Points Number)")
    plt.ylabel("Integral Value ± SE")
    plt.title("Integration using Monte-Carlo (MC) method by N")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.show()
