"""Static plots for the rigid-Earth rotation simulation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from astrodynamics.constants import RAD_TO_ARCSEC, RAD_TO_DEG
from astrodynamics.simulation import SimulationResult


def plot_polar_motion(
    result: SimulationResult,
) -> tuple[Figure, Axes]:
    """Plot the transverse angular-velocity components.

    The plot shows omega_1 against omega_2 in the Earth body-fixed frame.
    Both components are normalized by the nominal initial spin rate.
    """
    angular_velocity = result.angular_velocity_body
    omega_reference = angular_velocity[0, 2]

    omega_1_normalized = angular_velocity[:, 0] / omega_reference
    omega_2_normalized = angular_velocity[:, 1] / omega_reference

    figure, axis = plt.subplots(figsize=(7, 7))

    axis.plot(
        omega_1_normalized,
        omega_2_normalized,
        linewidth=1.2,
    )

    axis.scatter(
        omega_1_normalized[0],
        omega_2_normalized[0],
        marker="o",
        label="Initial state",
        zorder=3,
    )

    axis.scatter(
        omega_1_normalized[-1],
        omega_2_normalized[-1],
        marker="x",
        label="Final state",
        zorder=3,
    )

    axis.set_xlabel(r"$\omega_1 / \omega_\oplus$")
    axis.set_ylabel(r"$\omega_2 / \omega_\oplus$")
    axis.set_title("Polar Motion in the Body-Fixed Frame")
    axis.grid(True)
    axis.axis("equal")
    axis.legend()

    figure.tight_layout()

    return figure, axis


def plot_precession_angle(
    result: SimulationResult,
) -> tuple[Figure, Axes]:
    """Plot the precession angle psi as a function of time."""
    figure, axis = plt.subplots(figsize=(9, 5))

    psi_arcseconds = result.psi * RAD_TO_ARCSEC

    axis.plot(
        result.time_days,
        psi_arcseconds,
        linewidth=1.2,
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(r"$\psi$ [arcsec]")
    axis.set_title("Precession and Nutation in Longitude")
    axis.grid(True)

    figure.tight_layout()

    return figure, axis


def plot_obliquity(
    result: SimulationResult,
    *,
    show_variation: bool = True,
) -> tuple[Figure, Axes]:
    """Plot the Earth's obliquity.

    Parameters
    ----------
    result
        Numerical simulation result.
    show_variation
        If True, plot epsilon - epsilon_0 in arcseconds. If False,
        plot the full obliquity in degrees.
    """
    figure, axis = plt.subplots(figsize=(9, 5))

    if show_variation:
        obliquity_values = (
            result.obliquity - result.obliquity[0]
        ) * RAD_TO_ARCSEC

        ylabel = r"$\varepsilon-\varepsilon_0$ [arcsec]"
        title = "Variation of the Earth's Obliquity"
    else:
        obliquity_values = result.obliquity * RAD_TO_DEG

        ylabel = r"$\varepsilon$ [deg]"
        title = "Earth Obliquity"

    axis.plot(
        result.time_days,
        obliquity_values,
        linewidth=1.2,
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)

    figure.tight_layout()

    return figure, axis


def plot_sidereal_angle(
    result: SimulationResult,
    *,
    wrap_angle: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the sidereal angle theta.

    Parameters
    ----------
    result
        Numerical simulation result.
    wrap_angle
        If True, wrap theta into the interval [0, 360) degrees.
        Otherwise plot the accumulated angle in radians.
    """
    figure, axis = plt.subplots(figsize=(9, 5))

    if wrap_angle:
        sidereal_values = np.mod(
            result.sidereal_angle * RAD_TO_DEG,
            360.0,
        )
        ylabel = r"$\theta \bmod 360^\circ$ [deg]"
        title = "Wrapped Sidereal Angle"
    else:
        sidereal_values = result.sidereal_angle
        ylabel = r"$\theta$ [rad]"
        title = "Accumulated Sidereal Angle"

    axis.plot(
        result.time_days,
        sidereal_values,
        linewidth=1.2,
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)

    figure.tight_layout()

    return figure, axis


def plot_rigid_earth_summary(
    result: SimulationResult,
) -> tuple[Figure, np.ndarray]:
    """Create a four-panel summary of the rigid-Earth simulation."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    angular_velocity = result.angular_velocity_body
    omega_reference = angular_velocity[0, 2]

    omega_1_normalized = angular_velocity[:, 0] / omega_reference
    omega_2_normalized = angular_velocity[:, 1] / omega_reference

    axes[0, 0].plot(
        omega_1_normalized,
        omega_2_normalized,
        linewidth=1.0,
    )
    axes[0, 0].set_xlabel(r"$\omega_1/\omega_\oplus$")
    axes[0, 0].set_ylabel(r"$\omega_2/\omega_\oplus$")
    axes[0, 0].set_title("Polar Motion")
    axes[0, 0].grid(True)
    axes[0, 0].axis("equal")

    axes[0, 1].plot(
        result.time_days,
        result.psi * RAD_TO_ARCSEC,
        linewidth=1.0,
    )
    axes[0, 1].set_xlabel("Time [days]")
    axes[0, 1].set_ylabel(r"$\psi$ [arcsec]")
    axes[0, 1].set_title("Precession / Nutation in Longitude")
    axes[0, 1].grid(True)

    axes[1, 0].plot(
        result.time_days,
        (
            result.obliquity
            - result.obliquity[0]
        ) * RAD_TO_ARCSEC,
        linewidth=1.0,
    )
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_ylabel(
        r"$\varepsilon-\varepsilon_0$ [arcsec]"
    )
    axes[1, 0].set_title("Obliquity Variation")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        result.time_days,
        result.sidereal_angle,
        linewidth=1.0,
    )
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_ylabel(r"$\theta$ [rad]")
    axes[1, 1].set_title("Sidereal Angle")
    axes[1, 1].grid(True)

    figure.suptitle(
        "Rigid-Earth Rotation under Lunar Gravity-Gradient Torque"
    )
    figure.tight_layout()

    return figure, axes


def show_all_plots(result: SimulationResult) -> None:
    """Create the standard rigid-Earth plots and display them."""
    plot_polar_motion(result)
    plot_precession_angle(result)
    plot_obliquity(result)
    plot_sidereal_angle(result)

    plt.show()