# astrodynamics/bodies/earth.py

from __future__ import annotations

from dataclasses import dataclass

from astrodynamics.constants import DEG_TO_RAD


@dataclass(frozen=True, slots=True)
class EarthParameters:
    """Parameters used by the rigid-Earth rotation model."""

    rotation_rate: float = 7.2921151467e-5  # rad/s
    obliquity: float = 23.5 * DEG_TO_RAD    # rad

    # Dimensionless principal-inertia difference ratios
    gamma_1: float = 0.003295669
    gamma_2: float = -0.003295669
    gamma_3: float = 0.0


EARTH = EarthParameters()