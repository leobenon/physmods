# astrodynamics/bodies/earth.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astrodynamics.constants import DEG_TO_RAD


@dataclass(frozen=True, slots=True)
class EarthParameters:
    """Parameters used by the rigid-Earth rotation model."""

    rotation_rate: float = 7.2921151467e-5  # rad/s
    obliquity: float = 23.5 * DEG_TO_RAD  # rad

    # Dimensionless principal-inertia difference ratios.
    gamma_1: float = 0.003295669
    gamma_2: float = -0.003295669
    gamma_3: float = 0.0

    # Principal moments of inertia in kg m^2.
    #
    # The current model assumes an axisymmetric Earth:
    # A = B, with C slightly larger.
    moment_a: float = 8.01010135727e37
    moment_b: float = 8.01010135727e37
    moment_c: float = 8.0365e37

    @property
    def principal_moments(self) -> NDArray[np.float64]:
        """Return the principal moments [A, B, C] in kg m^2."""
        return np.array(
            [
                self.moment_a,
                self.moment_b,
                self.moment_c,
            ],
            dtype=float,
        )


EARTH = EarthParameters()