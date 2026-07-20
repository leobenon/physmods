"""Complete differential equations for the rigid-Earth rotation model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astrodynamics.bodies.earth import EARTH, EarthParameters
from astrodynamics.bodies.moon import MOON, Moon
from astrodynamics.dynamics.rigid_body import (
    angular_velocity_derivative,
)
from astrodynamics.dynamics.torques import (
    matlab_gravity_gradient_acceleration,
)


StateVector = NDArray[np.float64]
Vector3 = NDArray[np.float64]


def euler_angle_derivative(
    angular_velocity_body: NDArray[np.floating],
    obliquity: float,
    sidereal_angle: float,
    earth_rotation_rate: float,
) -> Vector3:
    """Return [psi_dot, epsilon_dot, theta_dot].

    This function reproduces the Euler-angle kinematics used in the
    original MATLAB model.

    Parameters
    ----------
    angular_velocity_body
        Earth angular-velocity components [omega_1, omega_2, omega_3]
        expressed in the body-fixed frame, in rad/s.
    obliquity
        Euler angle epsilon in radians.
    sidereal_angle
        Euler angle theta in radians.
    earth_rotation_rate
        Nominal Earth rotation rate in rad/s.

    Returns
    -------
    numpy.ndarray
        [psi_dot, epsilon_dot, theta_dot] in rad/s.

    Raises
    ------
    ValueError
        If the angular-velocity vector does not have shape (3,), or if
        sin(obliquity) is too close to zero.
    """
    angular_velocity_body = np.asarray(
        angular_velocity_body,
        dtype=float,
    )

    if angular_velocity_body.shape != (3,):
        raise ValueError(
            "angular_velocity_body must have shape (3,), "
            f"but received {angular_velocity_body.shape}."
        )

    sine_obliquity = np.sin(obliquity)

    if np.isclose(sine_obliquity, 0.0, atol=1.0e-14):
        raise ValueError(
            "Euler-angle kinematics are singular when "
            "sin(obliquity) is zero."
        )

    omega_1, omega_2, _ = angular_velocity_body

    projected_component = (
        omega_1 * np.sin(sidereal_angle)
        + omega_2 * np.cos(sidereal_angle)
    )

    psi_dot = -projected_component / sine_obliquity

    obliquity_dot = -(
        omega_1 * np.cos(sidereal_angle)
        - omega_2 * np.sin(sidereal_angle)
    )

    sidereal_angle_dot = (
        projected_component / np.tan(obliquity)
        + earth_rotation_rate
    )

    return np.array(
        [
            psi_dot,
            obliquity_dot,
            sidereal_angle_dot,
        ],
        dtype=float,
    )


def rigid_earth_state_derivative(
    time: float,
    state: NDArray[np.floating],
    *,
    earth: EarthParameters = EARTH,
    moon: Moon = MOON,
) -> StateVector:
    """Return the derivative of the complete rigid-Earth state.

    The state ordering is

        [omega_1, omega_2, omega_3, psi, epsilon, theta].

    Parameters
    ----------
    time
        Simulation time in seconds.
    state
        Six-component rigid-Earth state vector.
    earth
        Earth model parameters.
    moon
        Moon model parameters.

    Returns
    -------
    numpy.ndarray
        Six-component state derivative.
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (6,):
        raise ValueError(
            "state must have shape (6,), "
            f"but received {state.shape}."
        )

    angular_velocity_body = state[:3]
    obliquity = state[4]
    sidereal_angle = state[5]

    moon_position_body = moon.position_body_fixed(
        time=time,
        sidereal_angle=sidereal_angle,
    )

    normalized_lunar_torque = (
        matlab_gravity_gradient_acceleration(
            position_body=moon_position_body,
            gravitational_parameter=moon.gravitational_parameter,
            gamma_1=earth.gamma_1,
            gamma_2=earth.gamma_2,
            gamma_3=earth.gamma_3,
        )
    )

    angular_velocity_dot = angular_velocity_derivative(
        angular_velocity_body=angular_velocity_body,
        normalized_torque_body=normalized_lunar_torque,
        gamma_1=earth.gamma_1,
        gamma_2=earth.gamma_2,
        gamma_3=earth.gamma_3,
    )

    euler_angles_dot = euler_angle_derivative(
        angular_velocity_body=angular_velocity_body,
        obliquity=obliquity,
        sidereal_angle=sidereal_angle,
        earth_rotation_rate=earth.rotation_rate,
    )

    return np.concatenate(
        [
            angular_velocity_dot,
            euler_angles_dot,
        ]
    )