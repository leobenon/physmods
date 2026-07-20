"""Faithful Python translation of the MATLAB files rot2.m and deqrot2.m.

The model integrates a rigid, axisymmetric Earth under a simplified lunar
gravity-gradient torque. The state is
    y = [omega_1, omega_2, omega_3, psi, epsilon, theta]
where the angular-velocity components are expressed in the Earth-fixed
principal-axis frame and the last three entries are the lecture's Euler angles.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.typing import NDArray

# Conversion factor used in the MATLAB scripts
RAD_TO_ARCSEC = 206_264.8
SECONDS_PER_DAY = 86_400.0

# Model constants copied from deqrot2.m
GM_MOON = 398.6e12 / 81.3          # m^3/s^2
OMEGA_EARTH = 7_292_115.1467e-11  # rad/s
OMEGA_MOON = 2.661707223e-6       # rad/s
MOON_ORBIT_TILT = np.deg2rad(28.0)
MOON_DISTANCE = 3.8e8              # m

# Dimensionless inertia-difference ratios
GAMMA_1 = 0.003295669
GAMMA_2 = -0.003295669
GAMMA_3 = 0.0


def rotation_x(angle: float) -> NDArray[np.float64]:
    """Return the x-axis rotation matrix used by the MATLAB model."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=float,
    )


def inertial_to_earth_fixed_z(theta: float) -> NDArray[np.float64]:
    """Return the z rotation B from the MATLAB script.

    This matrix maps the already tilted inertial Moon vector into the rotating
    Earth-fixed frame using the current sidereal angle theta.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[c, s, 0.0],
         [-s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def moon_position_inertial(t: float) -> NDArray[np.float64]:
    """Simplified circular inertial Moon orbit used by the professor's code."""
    phase = OMEGA_MOON * t
    return np.array(
        [MOON_DISTANCE * np.cos(phase),
         MOON_DISTANCE * np.sin(phase),
         0.0],
        dtype=float,
    )


def rigid_earth_rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Differential equations translated from deqrot2.m."""
    omega_1, omega_2, omega_3, psi, epsilon, theta = y

    # Moon position in the Earth-fixed frame: x_mf = B(theta) A(eps) x_mi
    x_mi = moon_position_inertial(t)
    x_mf = (
        inertial_to_earth_fixed_z(theta)
        @ rotation_x(MOON_ORBIT_TILT)
        @ x_mi
    )
    x1, x2, x3 = x_mf

    # Lunar gravity-gradient forcing terms (units of angular acceleration)
    d1 = 3.0 * GM_MOON * GAMMA_1 * x3 * x2 / MOON_DISTANCE**5
    d2 = 3.0 * GM_MOON * GAMMA_2 * x1 * x3 / MOON_DISTANCE**5
    d3 = 3.0 * GM_MOON * GAMMA_3 * x2 * x1 / MOON_DISTANCE**5

    # Euler equations in the Earth-fixed principal-axis frame
    domega_1 = -GAMMA_1 * omega_2 * omega_3 + d1
    domega_2 = -GAMMA_2 * omega_3 * omega_1 + d2
    domega_3 = -GAMMA_3 * omega_1 * omega_2 + d3

    # Euler-angle kinematics, preserving the MATLAB sign/order convention
    common = omega_1 * np.sin(theta) + omega_2 * np.cos(theta)
    dpsi = -common / np.sin(epsilon)
    depsilon = -(omega_1 * np.cos(theta) - omega_2 * np.sin(theta))
    dtheta = common / np.tan(epsilon) + OMEGA_EARTH

    return np.array(
        [domega_1, domega_2, domega_3, dpsi, depsilon, dtheta],
        dtype=float,
    )


def run_simulation(days: float = 70.0) -> solve_ivp:
    """Integrate the professor's rigid-Earth model for the requested duration."""
    t_end = days * SECONDS_PER_DAY

    y0 = np.array(
        [
            1e-6 * OMEGA_EARTH,  # omega_1
            0.0,                 # omega_2
            OMEGA_EARTH,         # omega_3
            0.0,                 # psi
            np.deg2rad(23.5),    # epsilon
            0.0,                 # theta
        ],
        dtype=float,
    )

    # DOP853 is a high-order adaptive explicit method. It is not identical to
    # MATLAB ode113, but is well suited to reproducing this smooth problem.
    solution = solve_ivp(
        rigid_earth_rhs,
        (0.0, t_end),
        y0,
        method="DOP853",
        max_step=3600.0,
        rtol=1e-10,
        atol=1e-13,
        dense_output=False,
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    return solution


def plot_results(solution: solve_ivp) -> None:
    """Reproduce the four plots from rot2.m."""
    t_days = solution.t / SECONDS_PER_DAY
    w = solution.y.T

    # 1. Polar wobble in body-frame angular-velocity coordinates
    plt.figure(figsize=(7, 7))
    plt.plot(
        RAD_TO_ARCSEC * w[:, 0] / OMEGA_EARTH,
        RAD_TO_ARCSEC * w[:, 1] / OMEGA_EARTH,
    )
    plt.xlabel(r"$\omega_1/\omega_0$ [arcsec]")
    plt.ylabel(r"$\omega_2/\omega_0$ [arcsec]")
    plt.title("Rigid-Earth polar wobble")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    # 2. Psi
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_days, RAD_TO_ARCSEC * w[:, 3])
    plt.xlabel("Time [days]")
    plt.ylabel(r"$\psi$ [arcsec]")
    plt.title(r"Euler angle $\psi$")
    plt.grid(True)
    plt.tight_layout()

    # 3. Epsilon (absolute value, exactly as in MATLAB)
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_days, RAD_TO_ARCSEC * w[:, 4])
    plt.xlabel("Time [days]")
    plt.ylabel(r"$\epsilon$ [arcsec]")
    plt.title(r"Euler angle $\epsilon$")
    plt.grid(True)
    plt.tight_layout()

    # 4. Theta (unwrapped, exactly as in MATLAB)
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_days, RAD_TO_ARCSEC * w[:, 5])
    plt.xlabel("Time [days]")
    plt.ylabel(r"$\theta$ [arcsec]")
    plt.title(r"Euler angle $\theta$")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    result = run_simulation(days=70.0)
    plot_results(result)
