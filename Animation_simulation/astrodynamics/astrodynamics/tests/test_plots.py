"""Tests for rigid-Earth plotting functions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from astrodynamics.simulation import simulate_rigid_earth
from astrodynamics.visualization.plots import (
    plot_obliquity,
    plot_polar_motion,
    plot_precession_angle,
    plot_rigid_earth_summary,
    plot_sidereal_angle,
)


def short_result():
    return simulate_rigid_earth(
        duration_days=0.02,
        output_step=300.0,
        max_step=300.0,
    )


def test_plot_polar_motion_returns_figure_and_axis() -> None:
    result = short_result()

    figure, axis = plot_polar_motion(result)

    assert figure is not None
    assert axis is not None
    assert len(axis.lines) == 1

    plt.close(figure)


def test_plot_precession_returns_line() -> None:
    result = short_result()

    figure, axis = plot_precession_angle(result)

    assert len(axis.lines) == 1
    assert axis.lines[0].get_xdata().size == result.time.size

    plt.close(figure)


def test_plot_obliquity_returns_line() -> None:
    result = short_result()

    figure, axis = plot_obliquity(result)

    assert len(axis.lines) == 1
    assert axis.lines[0].get_ydata().size == result.time.size

    plt.close(figure)


def test_plot_sidereal_angle_returns_line() -> None:
    result = short_result()

    figure, axis = plot_sidereal_angle(result)

    assert len(axis.lines) == 1

    plt.close(figure)


def test_summary_contains_four_axes() -> None:
    result = short_result()

    figure, axes = plot_rigid_earth_summary(result)

    assert axes.shape == (2, 2)
    assert len(figure.axes) == 4

    plt.close(figure)