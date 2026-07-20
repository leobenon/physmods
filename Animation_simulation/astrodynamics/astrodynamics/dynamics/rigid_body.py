"""Rigid-body rotational dynamics in principal axes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Vector3 = NDArray[np.float64]


def angular_velocity_derivative(
    angular_velocity_body: NDArray[np.floating],
    normalized_torque_body: NDArray[np.floating],
    gamma_1: float,
    gamma_2: float,
    gamma_3: float,
) -> Vector3:
    """Return the body-frame angular-velocity derivative.

    Parameters
    ----------
    angular_velocity_body
        Components [omega_1, omega_2, omega_3] in rad/s.
    normalized_torque_body
        Components [d_1, d_2, d_3] in rad/s^2, where each physical
        torque component has been divided by its corresponding
        principal moment of inertia.
    gamma_1, gamma_2, gamma_3
        Principal-inertia difference ratios.

    Returns
    -------
    numpy.ndarray
        Angular acceleration [omega_dot_1, omega_dot_2, omega_dot_3].
    """
    angular_velocity_body = np.asarray(
        angular_velocity_body,
        dtype=float,
    )
    normalized_torque_body = np.asarray(
        normalized_torque_body,
        dtype=float,
    )

    if angular_velocity_body.shape != (3,):
        raise ValueError(
            "angular_velocity_body must have shape (3,), "
            f"but received {angular_velocity_body.shape}."
        )

    if normalized_torque_body.shape != (3,):
        raise ValueError(
            "normalized_torque_body must have shape (3,), "
            f"but received {normalized_torque_body.shape}."
        )

    omega_1, omega_2, omega_3 = angular_velocity_body
    d_1, d_2, d_3 = normalized_torque_body

    return np.array(
        [
            -gamma_1 * omega_2 * omega_3 + d_1,
            -gamma_2 * omega_3 * omega_1 + d_2,
            -gamma_3 * omega_1 * omega_2 + d_3,
        ],
        dtype=float,
    )