import numpy as np
import pytest

from astrodynamics.frames.rotation import (
    is_rotation_matrix,
    rotation_x,
    rotation_y,
    rotation_z,
)


@pytest.mark.parametrize(
    "rotation_function",
    [rotation_x, rotation_y, rotation_z],
)
@pytest.mark.parametrize(
    "angle",
    [0.0, 0.25, -0.75, np.pi / 2.0, np.pi],
)
def test_elementary_rotation_is_proper(
    rotation_function,
    angle: float,
) -> None:
    matrix = rotation_function(angle)

    assert matrix.shape == (3, 3)
    assert is_rotation_matrix(matrix)


@pytest.mark.parametrize(
    "rotation_function",
    [rotation_x, rotation_y, rotation_z],
)
@pytest.mark.parametrize(
    "angle",
    [0.0, 0.4, -1.2, np.pi],
)
def test_inverse_equals_transpose(
    rotation_function,
    angle: float,
) -> None:
    matrix = rotation_function(angle)

    assert np.allclose(
        np.linalg.inv(matrix),
        matrix.T,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "rotation_function",
    [rotation_x, rotation_y, rotation_z],
)
def test_zero_angle_returns_identity(rotation_function) -> None:
    assert np.allclose(
        rotation_function(0.0),
        np.eye(3),
        atol=1.0e-12,
    )


def test_rotation_x_quarter_turn() -> None:
    vector = np.array([0.0, 1.0, 0.0])

    rotated = rotation_x(np.pi / 2.0) @ vector

    expected = np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotated, expected, atol=1.0e-12)


def test_rotation_y_quarter_turn() -> None:
    vector = np.array([0.0, 0.0, 1.0])

    rotated = rotation_y(np.pi / 2.0) @ vector

    expected = np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, expected, atol=1.0e-12)


def test_rotation_z_quarter_turn() -> None:
    vector = np.array([1.0, 0.0, 0.0])

    rotated = rotation_z(np.pi / 2.0) @ vector

    expected = np.array([0.0, 1.0, 0.0])
    assert np.allclose(rotated, expected, atol=1.0e-12)


def test_matlab_x_rotation_matrix() -> None:
    epsilon = np.deg2rad(28.0)

    matlab_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(epsilon), -np.sin(epsilon)],
            [0.0, np.sin(epsilon), np.cos(epsilon)],
        ]
    )

    assert np.allclose(
        rotation_x(epsilon),
        matlab_matrix,
        atol=1.0e-12,
    )


def test_matlab_z_rotation_matrix() -> None:
    theta = 0.73

    matlab_matrix = np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(
        rotation_z(-theta),
        matlab_matrix,
        atol=1.0e-12,
    )