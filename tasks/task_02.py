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


def integrate_by_monte_carlo(func: Callable, a: float, b: float, points_number: int = 10_000) -> float:
    """
    Calculation of the integral of a function using the Monte Carlo method.

    :param func: The given function for integral calculation (Callable, mandatory)
    :param a: Lower limit of integration (Float, mandatory)
    :param b: Upper limit of integration (Float, mandatory)
    :param points_number: Number of random points for integral calculation (Int, optional)
    :return: Value of the integral (Float)
    """

    if not callable(func):
        raise ValueError("The function must be an callable object")

    x = np.random.uniform(a, b, points_number)
    y_min, y_max = np.min(func(x)), np.max(func(x))
    y = np.random.uniform(y_min, y_max, points_number)
    under_curve_points_number = np.sum(y < func(x))
    return (b - a) * (y_max - y_min) * (under_curve_points_number / points_number)


def f(x: float) -> float:
    """
    Integration function
    """
    return x ** 2


def test_integrate() -> None:
    a = 0.0
    b = 2.0

    # Integration using Monte Carlo method
    points_number = 1_000
    integral_value = integrate_by_monte_carlo(f, a, b, points_number=points_number)
    print(
        f"Result of integration using Monte Carlo method: {integral_value} "
        f"(number of random points {points_number}, number of experiments 1)"
    )
    points_number = 10_000
    integral_value = integrate_by_monte_carlo(f, a, b, points_number=points_number)
    print(
        f"Result of integration using Monte Carlo method: {integral_value} "
        f"(number of random points {points_number}, number of experiments 1)"
    )
    points_number = 100_000
    integral_value = integrate_by_monte_carlo(f, a, b, points_number=points_number)
    print(
        f"Result of integration using Monte Carlo method: {integral_value} "
        f"(number of random points {points_number}, number of experiments 1)"
    )

    experiments_number = 100
    points_number = 10_000
    integral_value = None
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(integrate_by_monte_carlo, f, a, b, points_number=points_number)
            for _ in range(experiments_number)
        ]
        integral_values: list[float] = [future.result() for future in futures]
        integral_value = np.mean(integral_values)
    print(
        f"Result of integration using Monte Carlo method: {integral_value} "
        f"(number of random points {points_number}, number of experiments {experiments_number})"
    )

    # Integration using scipy.integrate.quad
    print(
        "Result of integration using scipy.integrate.quad: {0} (estimate of the absolute error: {1})".format(
            *spi.quad(f, a, b)
        )
    )

    plot_integrate_function(f, a, b)
