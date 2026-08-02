"""Euler-angle transformations for the rigid-Earth model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astrodynamics.frames.rotation import (
    rotation_x,
    rotation_z,
)


DirectionCosineMatrix = NDArray[np.float64]


def inertial_to_body_euler_dcm(
    psi: float,
    epsilon: float,
    theta: float,
) -> DirectionCosineMatrix:
    """Return the inertial-to-body DCM for the model's 3-1-3 sequence."""
    return (
        rotation_z(-theta)
        @ rotation_x(epsilon)
        @ rotation_z(-psi)
    )


def body_to_inertial_euler_dcm(
    psi: float,
    epsilon: float,
    theta: float,
) -> DirectionCosineMatrix:
    """Return the body-to-inertial DCM."""
    return inertial_to_body_euler_dcm(
        psi=psi,
        epsilon=epsilon,
        theta=theta,
    ).T


def transform_body_to_inertial_euler(
    vector_body: NDArray[np.floating],
    *,
    psi: float,
    epsilon: float,
    theta: float,
) -> NDArray[np.float64]:
    """Express a body-frame vector in inertial coordinates."""
    vector_body = np.asarray(vector_body, dtype=float)

    if vector_body.shape != (3,):
        raise ValueError(
            "vector_body must have shape (3,), "
            f"but received {vector_body.shape}."
        )

    return (
        body_to_inertial_euler_dcm(
            psi=psi,
            epsilon=epsilon,
            theta=theta,
        )
        @ vector_body
    )