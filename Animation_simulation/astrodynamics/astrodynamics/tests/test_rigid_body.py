"""Tests for rigid-body rotational dynamics."""

import numpy as np
import pytest

from astrodynamics.dynamics.rigid_body import (
    angular_velocity_derivative,
)


def test_derivative_matches_explicit_euler_equations() -> None:
    omega = np.array([1.0, 2.0, 3.0])
    normalized_torque = np.array([0.1, -0.2, 0.3])

    gamma_1 = 0.4
    gamma_2 = -0.5
    gamma_3 = 0.2

    expected = np.array(
        [
            -gamma_1 * omega[1] * omega[2] + normalized_torque[0],
            -gamma_2 * omega[2] * omega[0] + normalized_torque[1],
            -gamma_3 * omega[0] * omega[1] + normalized_torque[2],
        ]
    )

    actual = angular_velocity_derivative(
        angular_velocity_body=omega,
        normalized_torque_body=normalized_torque,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
    )

    assert np.allclose(actual, expected, atol=1.0e-15)


def test_spherical_body_has_no_coupling() -> None:
    omega = np.array([1.0, -2.0, 3.0])
    normalized_torque = np.array([0.1, 0.2, 0.3])

    derivative = angular_velocity_derivative(
        angular_velocity_body=omega,
        normalized_torque_body=normalized_torque,
        gamma_1=0.0,
        gamma_2=0.0,
        gamma_3=0.0,
    )

    assert np.allclose(
        derivative,
        normalized_torque,
        atol=1.0e-15,
    )


def test_torque_free_rotation_about_principal_axis_is_steady() -> None:
    omega = np.array([0.0, 0.0, 7.0])

    derivative = angular_velocity_derivative(
        angular_velocity_body=omega,
        normalized_torque_body=np.zeros(3),
        gamma_1=0.3,
        gamma_2=-0.2,
        gamma_3=0.1,
    )

    assert np.allclose(derivative, np.zeros(3), atol=1.0e-15)


def test_axisymmetric_body_has_zero_third_derivative_without_torque() -> None:
    omega = np.array([1.0, 2.0, 3.0])

    derivative = angular_velocity_derivative(
        angular_velocity_body=omega,
        normalized_torque_body=np.zeros(3),
        gamma_1=0.4,
        gamma_2=-0.4,
        gamma_3=0.0,
    )

    assert derivative[2] == 0.0


def test_zero_angular_velocity_returns_applied_acceleration() -> None:
    normalized_torque = np.array([1.0e-8, -2.0e-8, 3.0e-8])

    derivative = angular_velocity_derivative(
        angular_velocity_body=np.zeros(3),
        normalized_torque_body=normalized_torque,
        gamma_1=0.3,
        gamma_2=-0.3,
        gamma_3=0.0,
    )

    assert np.allclose(
        derivative,
        normalized_torque,
        atol=1.0e-20,
    )


@pytest.mark.parametrize(
    "angular_velocity",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0]]),
    ],
)
def test_invalid_angular_velocity_shape_raises_error(
    angular_velocity,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        angular_velocity_derivative(
            angular_velocity_body=angular_velocity,
            normalized_torque_body=np.zeros(3),
            gamma_1=0.0,
            gamma_2=0.0,
            gamma_3=0.0,
        )


@pytest.mark.parametrize(
    "normalized_torque",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0]]),
    ],
)
def test_invalid_torque_shape_raises_error(
    normalized_torque,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        angular_velocity_derivative(
            angular_velocity_body=np.zeros(3),
            normalized_torque_body=normalized_torque,
            gamma_1=0.0,
            gamma_2=0.0,
            gamma_3=0.0,
        )