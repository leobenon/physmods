"""Simulation utilities for the rigid-Earth rotation model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from astrodynamics.dynamics.rigid_body import (
    angular_velocity_derivative,
)
from astrodynamics.dynamics.torques import (
    matlab_gravity_gradient_acceleration,
)

from astrodynamics.bodies.earth import EARTH, EarthParameters
from astrodynamics.bodies.moon import MOON, Moon
from astrodynamics.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from astrodynamics.dynamics.earth_rotation import (
    rigid_earth_state_derivative,
)


StateVector = NDArray[np.float64]
StateHistory = NDArray[np.float64]
VectorHistory = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Numerical and derived results of a rigid-Earth simulation."""

    time: NDArray[np.float64]
    state: StateHistory

    moon_position_inertial: VectorHistory
    moon_position_body: VectorHistory
    normalized_lunar_torque: VectorHistory
    angular_acceleration_body: VectorHistory
    rotation_axis_body: VectorHistory
    figure_axis_body: VectorHistory

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
    state_history = solution.y.T

    (
        moon_position_inertial,
        moon_position_body,
        normalized_lunar_torque,
        angular_acceleration_body,
        rotation_axis_body,
        figure_axis_body,
    ) = compute_derived_histories(
        time=solution.t,
        state=state_history,
        earth=earth,
        moon=moon,
    )

    return SimulationResult(
        time=solution.t,
        state=state_history,
        moon_position_inertial=moon_position_inertial,
        moon_position_body=moon_position_body,
        normalized_lunar_torque=normalized_lunar_torque,
        angular_acceleration_body=angular_acceleration_body,
        rotation_axis_body=rotation_axis_body,
        figure_axis_body=figure_axis_body,
        success=solution.success,
        message=solution.message,
        function_evaluations=solution.nfev,
    )

def save_simulation_csv(
    result: SimulationResult,
    filename: str,
) -> None:
    """Save a simulation result to a CSV file."""
    data = np.column_stack(
        [
            result.time,
            result.state,
            result.moon_position_inertial,
            result.moon_position_body,
            result.normalized_lunar_torque,
            result.angular_acceleration_body,
            result.rotation_axis_body,
            result.figure_axis_body,
        ]
    )

    header = (
        "time_s,"
        "omega_1_rad_s,"
        "omega_2_rad_s,"
        "omega_3_rad_s,"
        "psi_rad,"
        "epsilon_rad,"
        "theta_rad,"
        "moon_x_inertial_m,"
        "moon_y_inertial_m,"
        "moon_z_inertial_m,"
        "moon_x_body_m,"
        "moon_y_body_m,"
        "moon_z_body_m,"
        "normalized_torque_1_s-2,"
        "normalized_torque_2_s-2,"
        "normalized_torque_3_s-2,"
        "omega_dot_1_s-2,"
        "omega_dot_2_s-2,"
        "omega_dot_3_s-2,"
        "rotation_axis_body_x,"
        "rotation_axis_body_y,"
        "rotation_axis_body_z,"
        "figure_axis_body_x,"
        "figure_axis_body_y,"
        "figure_axis_body_z"
    )

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

def compute_derived_histories(
    time: NDArray[np.floating],
    state: NDArray[np.floating],
    *,
    earth: EarthParameters = EARTH,
    moon: Moon = MOON,
) -> tuple[
    VectorHistory,
    VectorHistory,
    VectorHistory,
    VectorHistory,
    VectorHistory,
    VectorHistory,
]:
    """Compute quantities derived from an integrated state history.

    Returns
    -------
    tuple
        Histories of:

        1. Moon position in the inertial frame
        2. Moon position in the body-fixed frame
        3. Normalized lunar torque
        4. Body-frame angular acceleration
        5. Body-frame rotation-axis direction
        6. Body-frame figure-axis direction
    """
    time = np.asarray(time, dtype=float)
    state = np.asarray(state, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be one-dimensional.")

    if state.ndim != 2 or state.shape[1] != 6:
        raise ValueError(
            "state must have shape (number_of_samples, 6)."
        )

    if state.shape[0] != time.size:
        raise ValueError(
            "time and state must contain the same number of samples."
        )

    number_of_samples = time.size

    moon_position_inertial = np.empty(
        (number_of_samples, 3),
        dtype=float,
    )
    moon_position_body = np.empty_like(
        moon_position_inertial
    )
    normalized_lunar_torque = np.empty_like(
        moon_position_inertial
    )
    angular_acceleration_body = np.empty_like(
        moon_position_inertial
    )
    rotation_axis_body = np.empty_like(
        moon_position_inertial
    )

    # In the body-fixed principal-axis frame, the figure axis is e3.
    figure_axis_body = np.tile(
        np.array([0.0, 0.0, 1.0]),
        (number_of_samples, 1),
    )

    for index, current_time in enumerate(time):
        angular_velocity = state[index, :3]
        sidereal_angle = state[index, 5]

        moon_position_inertial[index] = (
            moon.position_inertial(current_time)
        )

        moon_position_body[index] = (
            moon.position_body_fixed(
                time=current_time,
                sidereal_angle=sidereal_angle,
            )
        )

        normalized_lunar_torque[index] = (
            matlab_gravity_gradient_acceleration(
                position_body=moon_position_body[index],
                gravitational_parameter=(
                    moon.gravitational_parameter
                ),
                gamma_1=earth.gamma_1,
                gamma_2=earth.gamma_2,
                gamma_3=earth.gamma_3,
            )
        )

        angular_acceleration_body[index] = (
            angular_velocity_derivative(
                angular_velocity_body=angular_velocity,
                normalized_torque_body=(
                    normalized_lunar_torque[index]
                ),
                gamma_1=earth.gamma_1,
                gamma_2=earth.gamma_2,
                gamma_3=earth.gamma_3,
            )
        )

        angular_speed = np.linalg.norm(
            angular_velocity
        )

        if angular_speed == 0.0:
            rotation_axis_body[index] = np.zeros(3)
        else:
            rotation_axis_body[index] = (
                angular_velocity / angular_speed
            )

    return (
        moon_position_inertial,
        moon_position_body,
        normalized_lunar_torque,
        angular_acceleration_body,
        rotation_axis_body,
        figure_axis_body,
    )