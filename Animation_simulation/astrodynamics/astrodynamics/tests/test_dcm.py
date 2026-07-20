"""Tests for direction cosine matrices and frame transformations."""

import numpy as np
import pytest

from astrodynamics.frames.dcm import (
    inertial_to_body_dcm,
    transform_body_to_inertial,
    transform_inertial_to_body,
)
from astrodynamics.frames.rotation import (
    is_rotation_matrix,
    rotation_x,
    rotation_z,
)


def test_inertial_to_body_dcm_matches_matlab_product() -> None:
    theta = 0.73
    inclination = np.deg2rad(28.0)

    expected = (
        rotation_z(-theta)
        @ rotation_x(inclination)
    )

    actual = inertial_to_body_dcm(
        sidereal_angle=theta,
        lunar_orbit_inclination=inclination,
    )

    assert np.allclose(actual, expected, atol=1.0e-12)


def test_inertial_to_body_dcm_is_proper_rotation() -> None:
    dcm = inertial_to_body_dcm(
        sidereal_angle=1.2,
        lunar_orbit_inclination=np.deg2rad(28.0),
    )

    assert is_rotation_matrix(dcm)


def test_transformation_preserves_vector_norm() -> None:
    vector_inertial = np.array([2.0, -3.0, 5.0])

    vector_body = transform_inertial_to_body(
        vector_inertial=vector_inertial,
        sidereal_angle=0.8,
        lunar_orbit_inclination=0.4,
    )

    assert np.isclose(
        np.linalg.norm(vector_body),
        np.linalg.norm(vector_inertial),
        atol=1.0e-12,
    )


def test_body_to_inertial_is_inverse_transformation() -> None:
    vector_inertial = np.array([3.0, 4.0, -2.0])

    vector_body = transform_inertial_to_body(
        vector_inertial=vector_inertial,
        sidereal_angle=1.1,
        lunar_orbit_inclination=0.3,
    )

    recovered = transform_body_to_inertial(
        vector_body=vector_body,
        sidereal_angle=1.1,
        lunar_orbit_inclination=0.3,
    )

    assert np.allclose(
        recovered,
        vector_inertial,
        atol=1.0e-12,
    )


def test_zero_angles_return_unchanged_vector() -> None:
    vector = np.array([1.0, 2.0, 3.0])

    transformed = transform_inertial_to_body(
        vector_inertial=vector,
        sidereal_angle=0.0,
        lunar_orbit_inclination=0.0,
    )

    assert np.allclose(transformed, vector, atol=1.0e-12)


def test_matlab_transformation_for_known_vector() -> None:
    theta = 0.6
    inclination = np.deg2rad(28.0)
    vector_inertial = np.array([3.8e8, 0.0, 0.0])

    matrix_a = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(inclination), -np.sin(inclination)],
            [0.0, np.sin(inclination), np.cos(inclination)],
        ]
    )

    matrix_b = np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    expected = matrix_b @ matrix_a @ vector_inertial

    actual = transform_inertial_to_body(
        vector_inertial=vector_inertial,
        sidereal_angle=theta,
        lunar_orbit_inclination=inclination,
    )

    assert np.allclose(actual, expected, atol=1.0e-6)


def test_invalid_vector_shape_raises_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        transform_inertial_to_body(
            vector_inertial=np.array([1.0, 2.0]),
            sidereal_angle=0.0,
            lunar_orbit_inclination=0.0,
        )