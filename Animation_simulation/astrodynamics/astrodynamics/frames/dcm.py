"""Direction cosine matrices used by the rigid-Earth model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astrodynamics.frames.rotation import rotation_x, rotation_z


Vector3 = NDArray[np.float64]
DirectionCosineMatrix = NDArray[np.float64]


def inertial_to_body_dcm(
    sidereal_angle: float,
    lunar_orbit_inclination: float,
) -> DirectionCosineMatrix:
    """Return the inertial-to-body DCM used by the MATLAB model.

    The original MATLAB transformation is

        r_body = B @ A @ r_inertial

    where

        A = R_x(inclination)
        B = R_z(-sidereal_angle)

    Parameters
    ----------
    sidereal_angle
        Earth's sidereal rotation angle in radians.
    lunar_orbit_inclination
        Inclination angle used by the simplified lunar model, in radians.

    Returns
    -------
    numpy.ndarray
        The 3 x 3 direction cosine matrix mapping inertial coordinates
        into body-fixed coordinates.
    """
    return (
        rotation_z(-sidereal_angle)
        @ rotation_x(lunar_orbit_inclination)
    )


def transform_inertial_to_body(
    vector_inertial: NDArray[np.floating],
    sidereal_angle: float,
    lunar_orbit_inclination: float,
) -> Vector3:
    """Express an inertial vector in the body-fixed frame."""
    vector_inertial = np.asarray(vector_inertial, dtype=float)

    if vector_inertial.shape != (3,):
        raise ValueError(
            "vector_inertial must have shape (3,), "
            f"but received {vector_inertial.shape}."
        )

    dcm = inertial_to_body_dcm(
        sidereal_angle=sidereal_angle,
        lunar_orbit_inclination=lunar_orbit_inclination,
    )

    return dcm @ vector_inertial


def transform_body_to_inertial(
    vector_body: NDArray[np.floating],
    sidereal_angle: float,
    lunar_orbit_inclination: float,
) -> Vector3:
    """Express a body-fixed vector in the inertial frame."""
    vector_body = np.asarray(vector_body, dtype=float)

    if vector_body.shape != (3,):
        raise ValueError(
            "vector_body must have shape (3,), "
            f"but received {vector_body.shape}."
        )

    dcm = inertial_to_body_dcm(
        sidereal_angle=sidereal_angle,
        lunar_orbit_inclination=lunar_orbit_inclination,
    )

    return dcm.T @ vector_body