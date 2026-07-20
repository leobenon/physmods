"""Simulation utilities for the rigid-Earth rotation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from astrodynamics.bodies.earth import EARTH, EarthParameters
from astrodynamics.bodies.moon import MOON, Moon
from astrodynamics.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from astrodynamics.dynamics.earth_rotation import (
    rigid_earth_state_derivative,
)


StateVector = NDArray[np.float64]
StateHistory = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Numerical result of a rigid-Earth rotation simulation."""

    time: NDArray[np.float64]
    state: StateHistory
    success: bool
    message: str
    function_evaluations: int

    @property
    def angular_velocity_body(self) -> StateHistory:
        """Return omega_1, omega_2 and omega_3 for all output times."""
        return self.state[:, :3]

    @property
    def psi(self) -> NDArray[np.float64]:
        """Return the precession angle history."""
        return self.state[:, 3]

    @property
    def obliquity(self) -> NDArray[np.float64]:
        """Return the obliquity history."""
        return self.state[:, 4]

    @property
    def sidereal_angle(self) -> NDArray[np.float64]:
        """Return the sidereal-angle history."""
        return self.state[:, 5]

    @property
    def time_days(self) -> NDArray[np.float64]:
        """Return output times in days."""
        return self.time / SECONDS_PER_DAY


def default_initial_state(
    earth: EarthParameters = EARTH,
) -> StateVector:
    """Return the initial state used by the original MATLAB script.

    State ordering:

        [omega_1, omega_2, omega_3, psi, epsilon, theta]
    """
    return np.array(
        [
            1.0e-6 * earth.rotation_rate,
            0.0,
            earth.rotation_rate,
            0.0,
            earth.obliquity,
            0.0,
        ],
        dtype=float,
    )


def simulate_rigid_earth(
    *,
    duration_days: float = 70.0,
    output_step: float = SECONDS_PER_HOUR,
    max_step: float = SECONDS_PER_HOUR,
    initial_state: NDArray[np.floating] | None = None,
    earth: EarthParameters = EARTH,
    moon: Moon = MOON,
    method: str = "DOP853",
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float | NDArray[np.floating] = 1.0e-13,
) -> SimulationResult:
    """Integrate the rigid-Earth equations of motion.

    Parameters
    ----------
    duration_days
        Total simulation duration in days.
    output_step
        Time interval between returned output samples in seconds.
        This does not force the solver to use a fixed internal step.
    max_step
        Maximum internal integration step in seconds.
    initial_state
        Optional six-component initial state. If omitted, the original
        MATLAB initial conditions are used.
    earth
        Earth model parameters.
    moon
        Moon model parameters.
    method
        Integration method accepted by scipy.integrate.solve_ivp.
    relative_tolerance
        Relative local-error tolerance.
    absolute_tolerance
        Absolute local-error tolerance.

    Returns
    -------
    SimulationResult
        Time and state histories together with solver status information.
    """
    if duration_days <= 0.0:
        raise ValueError("duration_days must be positive.")

    if output_step <= 0.0:
        raise ValueError("output_step must be positive.")

    if max_step <= 0.0:
        raise ValueError("max_step must be positive.")

    if initial_state is None:
        state_0 = default_initial_state(earth)
    else:
        state_0 = np.asarray(initial_state, dtype=float)

    if state_0.shape != (6,):
        raise ValueError(
            "initial_state must have shape (6,), "
            f"but received {state_0.shape}."
        )

    final_time = duration_days * SECONDS_PER_DAY

    output_times = np.arange(
        0.0,
        final_time,
        output_step,
        dtype=float,
    )

    # Ensure the exact final time is included.
    if output_times.size == 0 or output_times[-1] < final_time:
        output_times = np.append(output_times, final_time)

    solution = solve_ivp(
        fun=lambda time, state: rigid_earth_state_derivative(
            time=time,
            state=state,
            earth=earth,
            moon=moon,
        ),
        t_span=(0.0, final_time),
        y0=state_0,
        method=method,
        t_eval=output_times,
        max_step=max_step,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )

    return SimulationResult(
        time=solution.t,
        state=solution.y.T,
        success=solution.success,
        message=solution.message,
        function_evaluations=solution.nfev,
    )