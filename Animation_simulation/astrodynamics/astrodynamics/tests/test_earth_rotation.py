"""Tests for the complete rigid-Earth differential equations."""

import numpy as np
import pytest

from astrodynamics.bodies.earth import EARTH
from astrodynamics.bodies.moon import MOON
from astrodynamics.dynamics.earth_rotation import (
    euler_angle_derivative,
    rigid_earth_state_derivative,
)
from astrodynamics.dynamics.rigid_body import (
    angular_velocity_derivative,
)
from astrodynamics.dynamics.torques import (
    matlab_gravity_gradient_acceleration,
)


def test_euler_angle_derivative_matches_matlab_equations() -> None:
    angular_velocity = np.array([1.2e-8, -2.3e-8, 7.29e-5])
    obliquity = np.deg2rad(23.5)
    theta = 0.7

    projected = (
        angular_velocity[0] * np.sin(theta)
        + angular_velocity[1] * np.cos(theta)
    )

    expected = np.array(
        [
            -projected / np.sin(obliquity),
            -(
                angular_velocity[0] * np.cos(theta)
                - angular_velocity[1] * np.sin(theta)
            ),
            projected / np.tan(obliquity)
            + EARTH.rotation_rate,
        ]
    )

    actual = euler_angle_derivative(
        angular_velocity_body=angular_velocity,
        obliquity=obliquity,
        sidereal_angle=theta,
        earth_rotation_rate=EARTH.rotation_rate,
    )

    assert np.allclose(actual, expected, atol=1.0e-20)


def test_nominal_spin_gives_nominal_sidereal_rate() -> None:
    derivative = euler_angle_derivative(
        angular_velocity_body=np.array(
            [0.0, 0.0, EARTH.rotation_rate]
        ),
        obliquity=np.deg2rad(23.5),
        sidereal_angle=0.0,
        earth_rotation_rate=EARTH.rotation_rate,
    )

    expected = np.array(
        [
            0.0,
            0.0,
            EARTH.rotation_rate,
        ]
    )

    assert np.allclose(derivative, expected, atol=1.0e-20)


def test_complete_derivative_has_six_components() -> None:
    state = np.array(
        [
            1.0e-6 * EARTH.rotation_rate,
            0.0,
            EARTH.rotation_rate,
            0.0,
            np.deg2rad(23.5),
            0.0,
        ]
    )

    derivative = rigid_earth_state_derivative(
        time=0.0,
        state=state,
    )

    assert derivative.shape == (6,)
    assert np.all(np.isfinite(derivative))


def test_complete_derivative_matches_individual_components() -> None:
    time = 1.2e5

    state = np.array(
        [
            1.0e-9,
            -2.0e-9,
            EARTH.rotation_rate,
            0.1,
            np.deg2rad(23.5),
            0.8,
        ]
    )

    moon_position_body = MOON.position_body_fixed(
        time=time,
        sidereal_angle=state[5],
    )

    normalized_torque = matlab_gravity_gradient_acceleration(
        position_body=moon_position_body,
        gravitational_parameter=MOON.gravitational_parameter,
        gamma_1=EARTH.gamma_1,
        gamma_2=EARTH.gamma_2,
        gamma_3=EARTH.gamma_3,
    )

    expected_omega_dot = angular_velocity_derivative(
        angular_velocity_body=state[:3],
        normalized_torque_body=normalized_torque,
        gamma_1=EARTH.gamma_1,
        gamma_2=EARTH.gamma_2,
        gamma_3=EARTH.gamma_3,
    )

    expected_angles_dot = euler_angle_derivative(
        angular_velocity_body=state[:3],
        obliquity=state[4],
        sidereal_angle=state[5],
        earth_rotation_rate=EARTH.rotation_rate,
    )

    expected = np.concatenate(
        [
            expected_omega_dot,
            expected_angles_dot,
        ]
    )

    actual = rigid_earth_state_derivative(
        time=time,
        state=state,
    )

    assert np.allclose(actual, expected, atol=1.0e-20)


def test_axisymmetric_model_keeps_third_spin_derivative_zero() -> None:
    state = np.array(
        [
            1.0e-9,
            2.0e-9,
            EARTH.rotation_rate,
            0.0,
            np.deg2rad(23.5),
            0.4,
        ]
    )

    derivative = rigid_earth_state_derivative(
        time=5.0e4,
        state=state,
    )

    assert derivative[2] == 0.0


@pytest.mark.parametrize(
    "invalid_state",
    [
        np.zeros(5),
        np.zeros(7),
        np.zeros((1, 6)),
    ],
)
def test_invalid_state_shape_raises_error(invalid_state) -> None:
    with pytest.raises(ValueError, match="shape"):
        rigid_earth_state_derivative(
            time=0.0,
            state=invalid_state,
        )


def test_euler_angle_singularity_raises_error() -> None:
    with pytest.raises(ValueError, match="singular"):
        euler_angle_derivative(
            angular_velocity_body=np.zeros(3),
            obliquity=0.0,
            sidereal_angle=0.0,
            earth_rotation_rate=EARTH.rotation_rate,
        )