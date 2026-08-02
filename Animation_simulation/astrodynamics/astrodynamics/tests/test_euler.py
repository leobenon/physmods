import numpy as np

from astrodynamics.frames.euler import (
    body_to_inertial_euler_dcm,
    inertial_to_body_euler_dcm,
)
from astrodynamics.frames.rotation import is_rotation_matrix


def test_euler_dcm_is_proper_rotation_matrix() -> None:
    matrix = inertial_to_body_euler_dcm(
        psi=0.2,
        epsilon=0.4,
        theta=0.7,
    )

    assert is_rotation_matrix(matrix)


def test_body_and_inertial_dcms_are_inverses() -> None:
    body_to_inertial = body_to_inertial_euler_dcm(
        psi=0.2,
        epsilon=0.4,
        theta=0.7,
    )

    inertial_to_body = inertial_to_body_euler_dcm(
        psi=0.2,
        epsilon=0.4,
        theta=0.7,
    )

    assert np.allclose(
        inertial_to_body @ body_to_inertial,
        np.eye(3),
        atol=1.0e-14,
    )

def test_figure_axis_has_obliquity_from_inertial_z() -> None:
    epsilon = np.deg2rad(23.5)

    body_to_inertial = body_to_inertial_euler_dcm(
        psi=0.0,
        epsilon=epsilon,
        theta=0.0,
    )

    figure_axis_inertial = (
        body_to_inertial
        @ np.array([0.0, 0.0, 1.0])
    )

    angle = np.arccos(
        np.clip(
            figure_axis_inertial[2],
            -1.0,
            1.0,
        )
    )

    assert np.isclose(
        angle,
        epsilon,
        atol=1.0e-14,
    )