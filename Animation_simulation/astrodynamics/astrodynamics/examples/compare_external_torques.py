"""Compare lunar and solar gravity-gradient torque contributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from astrodynamics.simulation import simulate_rigid_earth


ARCSECONDS_PER_RADIAN = 206_264.80624709636
SECONDS_PER_HOUR = 3600.0


def simulate_configuration(
    *,
    duration_days: float,
    output_step: float,
    include_lunar_torque: bool,
    include_solar_torque: bool,
):
    """Run one rigid-Earth torque configuration."""
    result = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        include_lunar_torque=include_lunar_torque,
        include_solar_torque=include_solar_torque,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result


def linear_rate_arcsec_per_year(
    time_days: np.ndarray,
    angle_arcsec: np.ndarray,
) -> float:
    """Return the best-fit linear angular rate in arcsec/year."""
    slope_arcsec_per_day, _ = np.polyfit(
        time_days,
        angle_arcsec,
        deg=1,
    )

    return float(
        slope_arcsec_per_day * 365.25
    )


def detrend_linear(
    time_days: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Remove the best-fit linear trend from a signal."""
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


def estimate_dominant_period(
    time_days: np.ndarray,
    signal: np.ndarray,
) -> float | None:
    """Estimate the dominant nonzero FFT period in days."""
    if time_days.size < 3:
        return None

    time_steps = np.diff(time_days)

    if not np.allclose(
        time_steps,
        time_steps[0],
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise ValueError(
            "FFT analysis requires evenly spaced output times."
        )

    centered_signal = signal - np.mean(signal)

    spectrum = np.abs(
        np.fft.rfft(centered_signal)
    )

    frequencies = np.fft.rfftfreq(
        centered_signal.size,
        d=time_steps[0],
    )

    # Ignore zero frequency.
    spectrum = spectrum[1:]
    frequencies = frequencies[1:]

    if frequencies.size == 0:
        return None

    dominant_index = int(
        np.argmax(spectrum)
    )

    dominant_frequency = frequencies[dominant_index]

    if dominant_frequency <= 0.0:
        return None

    return float(
        1.0 / dominant_frequency
    )


def print_configuration_summary(
    *,
    name: str,
    time_days: np.ndarray,
    delta_psi_arcsec: np.ndarray,
    delta_epsilon_arcsec: np.ndarray,
    torque_magnitude: np.ndarray,
) -> None:
    """Print the main diagnostics for one torque contribution."""
    psi_residual = detrend_linear(
        time_days,
        delta_psi_arcsec,
    )

    psi_period = estimate_dominant_period(
        time_days,
        psi_residual,
    )

    epsilon_period = estimate_dominant_period(
        time_days,
        delta_epsilon_arcsec,
    )

    print(f"\n{name}")
    print("-" * len(name))

    print(
        "Secular precession rate: "
        f"{linear_rate_arcsec_per_year(time_days, delta_psi_arcsec):.6f} "
        "arcsec/year"
    )

    print(
        "Maximum |delta psi|: "
        f"{np.max(np.abs(delta_psi_arcsec)):.6f} arcsec"
    )

    print(
        "Periodic psi amplitude: "
        f"{np.max(np.abs(psi_residual)):.6f} arcsec"
    )

    print(
        "Maximum |delta epsilon|: "
        f"{np.max(np.abs(delta_epsilon_arcsec)):.6f} arcsec"
    )

    print(
        "Maximum normalized torque: "
        f"{np.max(torque_magnitude):.6e} s^-2"
    )

    if psi_period is not None:
        print(
            "Dominant detrended-psi period: "
            f"{psi_period:.6f} days"
        )

    if epsilon_period is not None:
        print(
            "Dominant epsilon period: "
            f"{epsilon_period:.6f} days"
        )


def plot_angle_contributions(
    time_days: np.ndarray,
    lunar_difference: np.ndarray,
    solar_difference: np.ndarray,
    combined_difference: np.ndarray,
    *,
    ylabel: str,
    title: str,
) -> None:
    """Compare lunar, solar, and combined angular contributions."""
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        lunar_difference,
        label="Moon only",
    )

    axis.plot(
        time_days,
        solar_difference,
        label="Sun only",
    )

    axis.plot(
        time_days,
        combined_difference,
        label="Moon + Sun",
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True)
    axis.legend()

    figure.tight_layout()


def plot_detrended_precession_contributions(
    time_days: np.ndarray,
    lunar_psi: np.ndarray,
    solar_psi: np.ndarray,
    combined_psi: np.ndarray,
) -> None:
    """Compare periodic precession after secular trends are removed."""
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        detrend_linear(time_days, lunar_psi),
        label="Moon only",
    )

    axis.plot(
        time_days,
        detrend_linear(time_days, solar_psi),
        label="Sun only",
    )

    axis.plot(
        time_days,
        detrend_linear(time_days, combined_psi),
        label="Moon + Sun",
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"Detrended $\Delta\psi$ [arcsec]"
    )

    axis.set_title(
        "Periodic Precession Contributions"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()


def plot_torque_magnitudes(
    time_days: np.ndarray,
    lunar_torque: np.ndarray,
    solar_torque: np.ndarray,
    total_torque: np.ndarray,
) -> None:
    """Plot lunar, solar, and total normalized torque magnitudes."""
    figure, axis = plt.subplots(
        figsize=(9.5, 5.0),
    )

    axis.plot(
        time_days,
        np.linalg.norm(lunar_torque, axis=1),
        label="Lunar torque",
    )

    axis.plot(
        time_days,
        np.linalg.norm(solar_torque, axis=1),
        label="Solar torque",
    )

    axis.plot(
        time_days,
        np.linalg.norm(total_torque, axis=1),
        label="Total torque",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"Normalized torque magnitude [s$^{-2}$]"
    )

    axis.set_title(
        "External Gravity-Gradient Torques"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()


def main() -> None:
    """Run and compare all external-torque configurations."""
    duration_days = 365.0
    output_step = 6.0 * SECONDS_PER_HOUR

    no_torque = simulate_configuration(
        duration_days=duration_days,
        output_step=output_step,
        include_lunar_torque=False,
        include_solar_torque=False,
    )

    moon_only = simulate_configuration(
        duration_days=duration_days,
        output_step=output_step,
        include_lunar_torque=True,
        include_solar_torque=False,
    )

    sun_only = simulate_configuration(
        duration_days=duration_days,
        output_step=output_step,
        include_lunar_torque=False,
        include_solar_torque=True,
    )

    moon_and_sun = simulate_configuration(
        duration_days=duration_days,
        output_step=output_step,
        include_lunar_torque=True,
        include_solar_torque=True,
    )

    time_days = no_torque.time_days

    for result in (
        moon_only,
        sun_only,
        moon_and_sun,
    ):
        if not np.array_equal(
            result.time,
            no_torque.time,
        ):
            raise RuntimeError(
                "Simulation output times do not match."
            )

    lunar_delta_psi = (
        moon_only.psi - no_torque.psi
    ) * ARCSECONDS_PER_RADIAN

    solar_delta_psi = (
        sun_only.psi - no_torque.psi
    ) * ARCSECONDS_PER_RADIAN

    combined_delta_psi = (
        moon_and_sun.psi - no_torque.psi
    ) * ARCSECONDS_PER_RADIAN

    lunar_delta_epsilon = (
        moon_only.obliquity
        - no_torque.obliquity
    ) * ARCSECONDS_PER_RADIAN

    solar_delta_epsilon = (
        sun_only.obliquity
        - no_torque.obliquity
    ) * ARCSECONDS_PER_RADIAN

    combined_delta_epsilon = (
        moon_and_sun.obliquity
        - no_torque.obliquity
    ) * ARCSECONDS_PER_RADIAN

    print_configuration_summary(
        name="Moon-only contribution",
        time_days=time_days,
        delta_psi_arcsec=lunar_delta_psi,
        delta_epsilon_arcsec=lunar_delta_epsilon,
        torque_magnitude=np.linalg.norm(
            moon_only.normalized_lunar_torque,
            axis=1,
        ),
    )

    print_configuration_summary(
        name="Sun-only contribution",
        time_days=time_days,
        delta_psi_arcsec=solar_delta_psi,
        delta_epsilon_arcsec=solar_delta_epsilon,
        torque_magnitude=np.linalg.norm(
            sun_only.normalized_solar_torque,
            axis=1,
        ),
    )

    print_configuration_summary(
        name="Moon-and-Sun contribution",
        time_days=time_days,
        delta_psi_arcsec=combined_delta_psi,
        delta_epsilon_arcsec=combined_delta_epsilon,
        torque_magnitude=np.linalg.norm(
            moon_and_sun.normalized_total_torque,
            axis=1,
        ),
    )

    plot_angle_contributions(
        time_days,
        lunar_delta_psi,
        solar_delta_psi,
        combined_delta_psi,
        ylabel=r"$\Delta\psi$ [arcsec]",
        title="External-Torque Contributions to Precession",
    )

    plot_angle_contributions(
        time_days,
        lunar_delta_epsilon,
        solar_delta_epsilon,
        combined_delta_epsilon,
        ylabel=r"$\Delta\epsilon$ [arcsec]",
        title="External-Torque Contributions to Obliquity",
    )

    plot_detrended_precession_contributions(
        time_days,
        lunar_delta_psi,
        solar_delta_psi,
        combined_delta_psi,
    )

    plot_torque_magnitudes(
        time_days,
        moon_and_sun.normalized_lunar_torque,
        moon_and_sun.normalized_solar_torque,
        moon_and_sun.normalized_total_torque,
    )

    plt.show()


if __name__ == "__main__":
    main()