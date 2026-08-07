"""Compare rigid-Earth motion with and without lunar torque."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from astrodynamics.simulation import (
    default_initial_state,
    simulate_rigid_earth,
)


ARCSECONDS_PER_RADIAN = 206_264.80624709636
SECONDS_PER_HOUR = 3600.0


def run_comparison(
    *,
    duration_days: float = 365.0,
    output_step_hours: float = 6.0,
) -> None:
    """Compare forced and unforced rigid-Earth simulations."""

    output_step = output_step_hours * SECONDS_PER_HOUR

    with_torque = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        include_lunar_torque=True,
    )

    without_torque = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        include_lunar_torque=False,
    )

    if not with_torque.success:
        raise RuntimeError(with_torque.message)

    if not without_torque.success:
        raise RuntimeError(without_torque.message)

    if not np.array_equal(
        with_torque.time,
        without_torque.time,
    ):
        raise RuntimeError(
            "The two simulations returned different output times."
        )

    time_days = with_torque.time_days

    delta_psi_arcsec = (
        with_torque.psi
        - without_torque.psi
    ) * ARCSECONDS_PER_RADIAN

    delta_epsilon_arcsec = (
        with_torque.obliquity
        - without_torque.obliquity
    ) * ARCSECONDS_PER_RADIAN

    epsilon_period_days = estimate_dominant_period(
        time_days,
        delta_epsilon_arcsec,
    )

    if epsilon_period_days is not None:
        print(
            f"Dominant delta-epsilon period = "
            f"{epsilon_period_days:.6f} days"
        )

    delta_omega = (
        with_torque.angular_velocity_body
        - without_torque.angular_velocity_body
    )

    normalized_omega_difference = (
        delta_omega
        / with_torque.angular_velocity_body[0, 2]
    ) * ARCSECONDS_PER_RADIAN

    torque_magnitude = np.linalg.norm(
        with_torque.normalized_lunar_torque,
        axis=1,
    )

    print(
        "Maximum lunar-torque contribution over "
        f"{duration_days:.1f} days:"
    )

    print(
        f"  max |Δpsi|     = "
        f"{np.max(np.abs(delta_psi_arcsec)):.6e} arcsec"
    )

    print(
        f"  max |Δepsilon| = "
        f"{np.max(np.abs(delta_epsilon_arcsec)):.6e} arcsec"
    )

    print(
        f"  max |d_B|      = "
        f"{np.max(torque_magnitude):.6e} s^-2"
    )

    plot_angle_difference(
        time_days,
        delta_psi_arcsec,
        symbol=r"$\Delta\psi$",
        title="Lunar-Torque Contribution to Precession",
    )

    plot_detrended_precession(
        time_days,
        delta_psi_arcsec,
    )

    plot_angle_difference(
        time_days,
        delta_epsilon_arcsec,
        symbol=r"$\Delta\epsilon$",
        title="Lunar-Torque Contribution to Obliquity",
    )

    plot_polar_motion_difference(
        normalized_omega_difference,
    )

    plot_torque_magnitude(
        time_days,
        torque_magnitude,
    )

    

    plt.show()


def plot_angle_difference(
    time_days: np.ndarray,
    difference_arcsec: np.ndarray,
    *,
    symbol: str,
    title: str,
) -> None:
    """Plot one torque-on minus torque-off Euler-angle difference."""

    figure, axis = plt.subplots(
        figsize=(9, 4.8),
    )

    axis.plot(
        time_days,
        difference_arcsec,
        linewidth=1.3,
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(f"{symbol} [arcsec]")
    axis.set_title(title)
    axis.grid(True)

    figure.tight_layout()


def plot_polar_motion_difference(
    normalized_omega_difference: np.ndarray,
) -> None:
    """Plot the lunar-torque contribution in the polar-motion plane."""

    figure, axis = plt.subplots(
        figsize=(7, 7),
    )

    axis.plot(
        normalized_omega_difference[:, 0],
        normalized_omega_difference[:, 1],
        linewidth=1.2,
    )

    axis.scatter(
        normalized_omega_difference[0, 0],
        normalized_omega_difference[0, 1],
        s=35.0,
        label="Initial difference",
    )

    axis.scatter(
        normalized_omega_difference[-1, 0],
        normalized_omega_difference[-1, 1],
        s=35.0,
        marker="x",
        label="Final difference",
    )

    axis.set_xlabel(
        r"$\Delta\omega_1/\omega_0$ [arcsec]"
    )

    axis.set_ylabel(
        r"$\Delta\omega_2/\omega_0$ [arcsec]"
    )

    axis.set_title(
        "Lunar-Torque Contribution to Polar Motion"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()


def plot_torque_magnitude(
    time_days: np.ndarray,
    torque_magnitude: np.ndarray,
) -> None:
    """Plot the magnitude of the normalized lunar torque."""

    figure, axis = plt.subplots(
        figsize=(9, 4.8),
    )

    axis.plot(
        time_days,
        torque_magnitude,
        linewidth=1.2,
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(r"$|\mathbf{d}_B|$ [s$^{-2}$]")
    axis.set_title("Normalized Lunar-Torque Magnitude")
    axis.grid(True)

    figure.tight_layout()

def plot_detrended_precession(
    time_days: np.ndarray,
    delta_psi_arcsec: np.ndarray,
) -> None:
    """Separate secular drift from the periodic precession residual."""

    fit_coefficients = np.polyfit(
        time_days,
        delta_psi_arcsec,
        deg=1,
    )

    slope_arcsec_per_day = fit_coefficients[0]
    intercept_arcsec = fit_coefficients[1]

    fitted_trend = np.polyval(
        fit_coefficients,
        time_days,
    )

    residual_arcsec = (
        delta_psi_arcsec - fitted_trend
    )

    dominant_period_days = estimate_dominant_period(
        time_days,
        residual_arcsec,
    )

    if dominant_period_days is not None:
        print(
            f"  dominant residual period = "
            f"{dominant_period_days:.6f} days"
        )

    slope_arcsec_per_year = (
        slope_arcsec_per_day * 365.25
    )

    print(
        "Linear fit to lunar contribution to psi:"
    )
    print(
        f"  secular rate = "
        f"{slope_arcsec_per_day:.6e} arcsec/day"
    )
    print(
        f"               = "
        f"{slope_arcsec_per_year:.6e} arcsec/year"
    )
    print(
        f"  residual peak amplitude = "
        f"{np.max(np.abs(residual_arcsec)):.6e} arcsec"
    )

    figure, axis = plt.subplots(
        figsize=(9, 4.8),
    )

    axis.plot(
        time_days,
        delta_psi_arcsec,
        linewidth=1.2,
        label=r"$\Delta\psi$",
    )

    axis.plot(
        time_days,
        fitted_trend,
        linewidth=1.4,
        linestyle="--",
        label="Linear secular trend",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(r"$\Delta\psi$ [arcsec]")
    axis.set_title(
        "Secular and Periodic Lunar Contributions to Precession"
    )
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure, axis = plt.subplots(
        figsize=(9, 4.8),
    )

    axis.plot(
        time_days,
        residual_arcsec,
        linewidth=1.2,
    )

    axis.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel(
        r"$\Delta\psi-\Delta\psi_{\mathrm{linear}}$ "
        "[arcsec]"
    )
    axis.set_title(
        "Periodic Precession Residual after Linear Detrending"
    )
    axis.grid(True)

    figure.tight_layout()


def run_aligned_comparison(
    *,
    duration_days: float = 365.0,
    output_step_hours: float = 6.0,
) -> None:
    """Compare torque-on and torque-off runs without initial free wobble."""

    initial_state = default_initial_state()
    initial_state[0] = 0.0
    initial_state[1] = 0.0

    output_step = output_step_hours * SECONDS_PER_HOUR

    with_torque = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        initial_state=initial_state,
        include_lunar_torque=True,
    )

    without_torque = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=output_step,
        max_step=SECONDS_PER_HOUR,
        initial_state=initial_state,
        include_lunar_torque=False,
    )

    delta_psi_arcsec = (
        with_torque.psi
        - without_torque.psi
    ) * ARCSECONDS_PER_RADIAN

    delta_epsilon_arcsec = (
        with_torque.obliquity
        - without_torque.obliquity
    ) * ARCSECONDS_PER_RADIAN

    figure, axis = plt.subplots(
        figsize=(9, 4.8),
    )

    axis.plot(
        with_torque.time_days,
        delta_psi_arcsec,
        label=r"$\Delta\psi$",
    )

    axis.plot(
        with_torque.time_days,
        delta_epsilon_arcsec,
        label=r"$\Delta\epsilon$",
    )

    axis.set_xlabel("Time [days]")
    axis.set_ylabel("Torque-induced difference [arcsec]")

    axis.set_title(
        "Forced Response with Initial Free Wobble Removed"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()
    plt.show()

def estimate_dominant_period(
    time_days: np.ndarray,
    signal: np.ndarray,
) -> float | None:
    """Estimate the dominant nonzero period of an evenly sampled signal."""

    if time_days.size < 3:
        return None

    time_steps = np.diff(time_days)

    if not np.allclose(
        time_steps,
        time_steps[0],
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Time samples must be evenly spaced for FFT analysis."
        )

    centered_signal = signal - np.mean(signal)

    frequencies = np.fft.rfftfreq(
        centered_signal.size,
        d=time_steps[0],
    )

    spectrum = np.abs(
        np.fft.rfft(centered_signal)
    )

    # Ignore the zero-frequency component.
    frequencies = frequencies[1:]
    spectrum = spectrum[1:]

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


def main() -> None:
    """Run the standard lunar-torque comparison."""

    run_comparison(
        duration_days=365.0,
        output_step_hours=6.0,
    )

    
    
    run_aligned_comparison(
         duration_days=365.0,
         output_step_hours=6.0,
    )


if __name__ == "__main__":
    main()