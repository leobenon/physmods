"""External torque models for rotational dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Vector3 = NDArray[np.float64]


def gravity_gradient_torque(
    position_body: NDArray[np.floating],
    gravitational_parameter: float,
    inertia_tensor: NDArray[np.floating],
) -> Vector3:
    """Return the physical gravity-gradient torque in the body frame.

    Parameters
    ----------
    position_body
        Position of the disturbing body relative to the rigid body,
        expressed in the body-fixed frame, in metres.
    gravitational_parameter
        Gravitational parameter of the disturbing body in m^3/s^2.
    inertia_tensor
        Body inertia tensor in kg m^2, expressed in the body frame.

    Returns
    -------
    numpy.ndarray
        Physical gravity-gradient torque in N m.
    """
    position_body = np.asarray(position_body, dtype=float)
    inertia_tensor = np.asarray(inertia_tensor, dtype=float)

    if position_body.shape != (3,):
        raise ValueError(
            "position_body must have shape (3,), "
            f"but received {position_body.shape}."
        )

    if inertia_tensor.shape != (3, 3):
        raise ValueError(
            "inertia_tensor must have shape (3, 3), "
            f"but received {inertia_tensor.shape}."
        )

    radius = np.linalg.norm(position_body)

    if radius == 0.0:
        raise ValueError("position_body must be non-zero.")

    return (
        3.0
        * gravitational_parameter
        / radius**5
        * np.cross(position_body, inertia_tensor @ position_body)
    )


def matlab_gravity_gradient_acceleration(
    position_body: NDArray[np.floating],
    gravitational_parameter: float,
    gamma_1: float,
    gamma_2: float,
    gamma_3: float,
) -> Vector3:
    """Return the normalized lunar torque terms used by the MATLAB model.

    These are the physical torque components divided by the corresponding
    principal moments of inertia. Their units are rad/s^2, conventionally
    written as s^-2 because radians are dimensionless.

    The implemented equations are

        d1 = 3 mu gamma1 y z / r^5
        d2 = 3 mu gamma2 z x / r^5
        d3 = 3 mu gamma3 x y / r^5
    """
    position_body = np.asarray(position_body, dtype=float)

    if position_body.shape != (3,):
        raise ValueError(
            "position_body must have shape (3,), "
            f"but received {position_body.shape}."
        )

    radius = np.linalg.norm(position_body)

    if radius == 0.0:
        raise ValueError("position_body must be non-zero.")

    x_position, y_position, z_position = position_body
    factor = 3.0 * gravitational_parameter / radius**5

    return factor * np.array(
        [
            gamma_1 * y_position * z_position,
            gamma_2 * z_position * x_position,
            gamma_3 * x_position * y_position,
        ],
        dtype=float,
    )