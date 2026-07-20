"""Tests for gravity-gradient torque models."""

import numpy as np
import pytest

from astrodynamics.bodies.earth import EARTH
from astrodynamics.bodies.moon import MOON
from astrodynamics.dynamics.torques import (
    gravity_gradient_torque,
    matlab_gravity_gradient_acceleration,
)


def test_matlab_acceleration_matches_explicit_equations() -> None:
    position_body = np.array([2.0e8, -2.5e8, 1.8e8])

    x_position, y_position, z_position = position_body
    radius = np.linalg.norm(position_body)

    expected = np.array(
        [
            3.0
            * MOON.gravitational_parameter
            * EARTH.gamma_1
            * z_position
            * y_position
            / radius**5,
            3.0
            * MOON.gravitational_parameter
            * EARTH.gamma_2
            * x_position
            * z_position
            / radius**5,
            3.0
            * MOON.gravitational_parameter
            * EARTH.gamma_3
            * y_position
            * x_position
            / radius**5,
        ]
    )

    actual = matlab_gravity_gradient_acceleration(
        position_body=position_body,
        gravitational_parameter=MOON.gravitational_parameter,
        gamma_1=EARTH.gamma_1,
        gamma_2=EARTH.gamma_2,
        gamma_3=EARTH.gamma_3,
    )

    assert np.allclose(actual, expected, atol=1.0e-30)


def test_axisymmetric_model_has_zero_third_component() -> None:
    position_body = np.array([2.0e8, 2.5e8, 1.0e8])

    acceleration = matlab_gravity_gradient_acceleration(
        position_body=position_body,
        gravitational_parameter=MOON.gravitational_parameter,
        gamma_1=EARTH.gamma_1,
        gamma_2=EARTH.gamma_2,
        gamma_3=EARTH.gamma_3,
    )

    assert acceleration[2] == 0.0


def test_zero_torque_along_principal_axis() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    position_body = np.array([10.0, 0.0, 0.0])

    torque = gravity_gradient_torque(
        position_body=position_body,
        gravitational_parameter=1.0,
        inertia_tensor=inertia,
    )

    assert np.allclose(torque, np.zeros(3), atol=1.0e-15)


def test_spherical_body_has_zero_gravity_gradient_torque() -> None:
    inertia = 4.0 * np.eye(3)
    position_body = np.array([2.0, -3.0, 5.0])

    torque = gravity_gradient_torque(
        position_body=position_body,
        gravitational_parameter=7.0,
        inertia_tensor=inertia,
    )

    assert np.allclose(torque, np.zeros(3), atol=1.0e-15)


def test_general_torque_matches_principal_axis_formula() -> None:
    moment_a = 2.0
    moment_b = 3.0
    moment_c = 5.0

    inertia = np.diag([moment_a, moment_b, moment_c])
    position_body = np.array([2.0, 3.0, 4.0])
    gravitational_parameter = 7.0

    x_position, y_position, z_position = position_body
    radius = np.linalg.norm(position_body)
    factor = 3.0 * gravitational_parameter / radius**5

    expected = factor * np.array(
        [
            (moment_c - moment_b) * y_position * z_position,
            (moment_a - moment_c) * z_position * x_position,
            (moment_b - moment_a) * x_position * y_position,
        ]
    )

    actual = gravity_gradient_torque(
        position_body=position_body,
        gravitational_parameter=gravitational_parameter,
        inertia_tensor=inertia,
    )

    assert np.allclose(actual, expected, atol=1.0e-15)


def test_torque_is_perpendicular_to_position_vector() -> None:
    inertia = np.diag([2.0, 3.0, 5.0])
    position_body = np.array([2.0, -3.0, 5.0])

    torque = gravity_gradient_torque(
        position_body=position_body,
        gravitational_parameter=7.0,
        inertia_tensor=inertia,
    )

    assert np.isclose(
        np.dot(position_body, torque),
        0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "position",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0]]),
    ],
)
def test_invalid_position_shape_raises_error(position) -> None:
    with pytest.raises(ValueError, match="shape"):
        matlab_gravity_gradient_acceleration(
            position_body=position,
            gravitational_parameter=1.0,
            gamma_1=1.0,
            gamma_2=1.0,
            gamma_3=1.0,
        )


def test_zero_position_raises_error() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        matlab_gravity_gradient_acceleration(
            position_body=np.zeros(3),
            gravitational_parameter=1.0,
            gamma_1=1.0,
            gamma_2=1.0,
            gamma_3=1.0,
        )