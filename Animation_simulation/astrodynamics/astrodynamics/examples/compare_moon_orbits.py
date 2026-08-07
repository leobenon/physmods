"""Compare circular and elliptical lunar-orbit models."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from astrodynamics.bodies.moon import MOON
from astrodynamics.simulation import simulate_rigid_earth


ARCSECONDS_PER_RADIAN = 206_264.80624709636
SECONDS_PER_HOUR = 3600.0


def detrend_linear(
    time_days: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Remove the best-fit linear trend."""
    coefficients = np.polyfit(
        time_days,
        signal,
        deg=1,
    )

    trend = np.polyval(
        coefficients,
        time_days,
    )

    return signal - trend


def linear_rate_arcsec_per_year(
    time_days: np.ndarray,
    signal_arcsec: np.ndarray,
) -> float:
    """Return the best-fit secular rate in arcsec/year."""
    slope, _ = np.polyfit(
        time_days,
        signal_arcsec,
        deg=1,
    )

    return float(
        slope * 365.25
    )


def dominant_period_days(
    time_days: np.ndarray,
    signal: np.ndarray,
) -> float | None:
    """Return the strongest nonzero FFT period in days."""
    if time_days.size < 3:
        return None

    time_step = np.diff(time_days)

    if not np.allclose(
        time_step,
        time_step[0],
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ValueError(
            "FFT analysis requires evenly spaced output times."
        )

    centered = signal - np.mean(signal)

    spectrum = np.abs(
        np.fft.rfft(centered)
    )

    frequencies = np.fft.rfftfreq(
        centered.size,
        d=time_step[0],
    )

    spectrum = spectrum[1:]
    frequencies = frequencies[1:]

    if frequencies.size == 0:
        return None

    dominant_index = int(
        np.argmax(spectrum)
    )

    frequency = frequencies[
        dominant_index
    ]

    if frequency <= 0.0:
        return None

    return float(
        1.0 / frequency
    )


def main() -> None:
    duration_days = 365.0
    output_step = 6.0 * SECONDS_PER_HOUR

    circular_moon = replace(
        MOON,
        orbit_model="circular",
    )

    elliptical_moon = replace(
        MOON,
        orbit_model="elliptical",
    )

    circular = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        moon=circular_moon,
        include_lunar_torque=True,
        include_solar_torque=False,
    )

    elliptical = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        moon=elliptical_moon,
        include_lunar_torque=True,
        include_solar_torque=False,
    )

    if not circular.success:
        raise RuntimeError(circular.message)

    if not elliptical.success:
        raise RuntimeError(elliptical.message)

    if not np.array_equal(
        circular.time,
        elliptical.time,
    ):
        raise RuntimeError(
            "Simulation output times do not match."
        )

    time_days = circular.time_days

    circular_distance = np.linalg.norm(
        circular.moon_position_inertial,
        axis=1,
    )

    elliptical_distance = np.linalg.norm(
        elliptical.moon_position_inertial,
        axis=1,
    )

    circular_torque = np.linalg.norm(
        circular.normalized_lunar_torque,
        axis=1,
    )

    elliptical_torque = np.linalg.norm(
        elliptical.normalized_lunar_torque,
        axis=1,
    )

    psi_difference = (
        elliptical.psi
        - circular.psi
    ) * ARCSECONDS_PER_RADIAN

    epsilon_difference = (
        elliptical.obliquity
        - circular.obliquity
    ) * ARCSECONDS_PER_RADIAN

    circular_psi = (
        circular.psi
        - circular.psi[0]
    ) * ARCSECONDS_PER_RADIAN

    elliptical_psi = (
        elliptical.psi
        - elliptical.psi[0]
    ) * ARCSECONDS_PER_RADIAN

    circular_epsilon = (
        circular.obliquity
        - circular.obliquity[0]
    ) * ARCSECONDS_PER_RADIAN

    elliptical_epsilon = (
        elliptical.obliquity
        - elliptical.obliquity[0]
    ) * ARCSECONDS_PER_RADIAN

    circular_psi_residual = detrend_linear(
        time_days,
        circular_psi,
    )

    elliptical_psi_residual = detrend_linear(
        time_days,
        elliptical_psi,
    )

    circular_period = dominant_period_days(
        time_days,
        circular_psi_residual,
    )

    elliptical_period = dominant_period_days(
        time_days,
        elliptical_psi_residual,
    )



    print("Circular Moon")
    print("-------------")
    print(
        f"Mean distance        = "
        f"{np.mean(circular_distance) / 1.0e3:.3f} km"
    )
    print(
        f"Min distance         = "
        f"{np.min(circular_distance) / 1.0e3:.3f} km"
    )
    print(
        f"Max distance         = "
        f"{np.max(circular_distance) / 1.0e3:.3f} km"
    )
    print(
        f"Max normalized torque = "
        f"{np.max(circular_torque):.6e} s^-2"
    )
    print(
        f"Secular psi rate      = "
        f"{linear_rate_arcsec_per_year(time_days, circular_psi):.6f} "
        "arcsec/year"
    )
    print(
        f"Periodic psi amplitude = "
        f"{np.max(np.abs(circular_psi_residual)):.6f} arcsec"
    )

    if circular_period is not None:
        print(
            f"Dominant residual period = "
            f"{circular_period:.6f} days"
        )

    print()

    print("Elliptical Moon")
    print("---------------")
    print(
        f"Mean distance        = "
        f"{np.mean(elliptical_distance) / 1.0e3:.3f} km"
    )
    print(
        f"Perigee distance     = "
        f"{np.min(elliptical_distance) / 1.0e3:.3f} km"
    )
    print(
        f"Apogee distance      = "
        f"{np.max(elliptical_distance) / 1.0e3:.3f} km"
    )
    print(
        f"Max normalized torque = "
        f"{np.max(elliptical_torque):.6e} s^-2"
    )

    print(
        f"Secular psi rate      = "
        f"{linear_rate_arcsec_per_year(time_days, elliptical_psi):.6f} "
        "arcsec/year"
    )
    print(
        f"Periodic psi amplitude = "
        f"{np.max(np.abs(elliptical_psi_residual)):.6f} arcsec"
    )

    if elliptical_period is not None:
        print(
            f"Dominant residual period = "
            f"{elliptical_period:.6f} days"
        )

    print()

    print("Elliptical - Circular")
    print("---------------------")
    print(
        f"Max |delta psi|      = "
        f"{np.max(np.abs(psi_difference)):.6f} arcsec"
    )
    print(
        f"Max |delta epsilon|  = "
        f"{np.max(np.abs(epsilon_difference)):.6f} arcsec"
    )
    print(
        f"Torque max ratio     = "
        f"{np.max(elliptical_torque) / np.max(circular_torque):.6f}"
    )

    # Moon-Earth distance
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        circular_distance / 1.0e3,
        label="Circular Moon",
    )

    axis.plot(
        time_days,
        elliptical_distance / 1.0e3,
        label="Elliptical Moon",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel("Moon-Earth distance [km]")
    axis.set_title(
        "Circular and Elliptical Lunar-Orbit Distance"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    # Torque
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        circular_torque,
        label="Circular Moon",
    )

    axis.plot(
        time_days,
        elliptical_torque,
        label="Elliptical Moon",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"Normalized lunar torque magnitude [s$^{-2}$]"
    )
    axis.set_title(
        "Lunar Gravity-Gradient Torque"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    # Precession
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        circular_psi,
        label="Circular Moon",
    )

    axis.plot(
        time_days,
        elliptical_psi,
        label="Elliptical Moon",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"$\Delta\psi$ [arcsec]"
    )
    axis.set_title(
        "Lunar Contribution to Precession"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    # Detrended precession
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        circular_psi_residual,
        label="Circular Moon",
    )

    axis.plot(
        time_days,
        elliptical_psi_residual,
        label="Elliptical Moon",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"Detrended $\Delta\psi$ [arcsec]"
    )
    axis.set_title(
        "Periodic Precession: Circular vs Elliptical Moon"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    # Obliquity
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        circular_epsilon,
        label="Circular Moon",
    )

    axis.plot(
        time_days,
        elliptical_epsilon,
        label="Elliptical Moon",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"$\Delta\epsilon$ [arcsec]"
    )
    axis.set_title(
        "Lunar Contribution to Obliquity"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    # Difference produced specifically by ellipticity
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        psi_difference,
        label=r"$\Delta\psi_{\rm ell}-\Delta\psi_{\rm circ}$",
    )

    axis.plot(
        time_days,
        epsilon_difference,
        label=r"$\Delta\epsilon_{\rm ell}-\Delta\epsilon_{\rm circ}$",
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel("Difference [arcsec]")
    axis.set_title(
        "Effect of Lunar-Orbit Ellipticity"
    )
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()