"""Elementary three-dimensional rotation matrices.

The functions in this module return active, right-handed rotation matrices
for column vectors:

    v_rotated = R @ v

For a passive coordinate transformation, use the transpose:

    v_new_frame = R.T @ v_old_frame
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


RotationMatrix = NDArray[np.float64]


def rotation_x(angle: float) -> RotationMatrix:
    """Return the active right-handed rotation matrix about the x-axis.

    Parameters
    ----------
    angle
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarrays
        A 3 x 3 rotation matrix.
    """
    cosine = np.cos(angle)
    sine = np.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=float,
    )


def rotation_y(angle: float) -> RotationMatrix:
    """Return the active right-handed rotation matrix about the y-axis."""
    cosine = np.cos(angle)
    sine = np.sin(angle)

    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=float,
    )


def rotation_z(angle: float) -> RotationMatrix:
    """Return the active right-handed rotation matrix about the z-axis."""
    cosine = np.cos(angle)
    sine = np.sin(angle)

    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def is_rotation_matrix(
    matrix: NDArray[np.floating],
    *,
    atol: float = 1.0e-12,
) -> bool:
    """Return whether a matrix is a proper three-dimensional rotation matrix.

    A proper rotation matrix satisfies

        R.T @ R = I

    and

        det(R) = +1.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        return False

    identity = np.eye(3)
    orthogonal = np.allclose(matrix.T @ matrix, identity, atol=atol)
    proper = np.isclose(np.linalg.det(matrix), 1.0, atol=atol)

    return bool(orthogonal and proper)