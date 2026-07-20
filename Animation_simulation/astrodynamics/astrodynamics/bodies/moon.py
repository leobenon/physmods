# astrodynamics/bodies/moon.py

from __future__ import annotations

from dataclasses import dataclass

from astrodynamics.constants import DEG_TO_RAD


@dataclass(frozen=True, slots=True)
class MoonParameters:
    """Parameters used by the simplified circular lunar-orbit model."""

    gravitational_parameter: float = 398.6e12 / 81.3  # m^3/s^2
    orbital_radius: float = 3.8e8                     # m
    orbital_angular_rate: float = 2.661707223e-6      # rad/s
    orbital_inclination: float = 28.0 * DEG_TO_RAD    # rad


MOON = MoonParameters()