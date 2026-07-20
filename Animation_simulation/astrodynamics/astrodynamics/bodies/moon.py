# astrodynamics/bodies/moon.py

"""Simplified Moon model used by the rigid-Earth simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astrodynamics.constants import DEG_TO_RAD

from astrodynamics.frames.dcm import transform_inertial_to_body


Vector3 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Moon:
    """Parameters and inertial position model for a circular lunar orbit.

    This model reproduces the assumptions in the original MATLAB code:

    - constant orbital radius,
    - constant angular rate,
    - circular motion in the inertial xy-plane,
    - orbital inclination applied later during the frame transformation.
    """

    gravitational_parameter: float = 398.6e12 / 81.3
    orbital_radius: float = 3.8e8
    orbital_angular_rate: float = 2.661707223e-6
    orbital_inclination: float = 28.0 * DEG_TO_RAD

    def position_inertial(self, time: float) -> Vector3:
        """Return the Moon position in the inertial reference frame.

        Parameters
        ----------
        time
            Simulation time in seconds.

        Returns
        -------
        numpy.ndarray
            Moon position vector in metres.
        """
        phase = self.orbital_angular_rate * time

        return np.array(
            [
                self.orbital_radius * np.cos(phase),
                self.orbital_radius * np.sin(phase),
                0.0,
            ],
            dtype=float,
        )

    @property
    def orbital_period(self) -> float:
        """Return the orbital period in seconds."""
        return 2.0 * np.pi / self.orbital_angular_rate
    
    def position_body_fixed(
        self,
        time: float,
        sidereal_angle: float,
    ) -> Vector3:
        """Return the Moon position expressed in the Earth body-fixed frame.

        Parameters
        ----------
        time
            Simulation time in seconds.
        sidereal_angle
            Earth's sidereal rotation angle in radians.
        """
        position_inertial = self.position_inertial(time)

        return transform_inertial_to_body(
            vector_inertial=position_inertial,
            sidereal_angle=sidereal_angle,
            lunar_orbit_inclination=self.orbital_inclination,
        )


MOON = Moon()