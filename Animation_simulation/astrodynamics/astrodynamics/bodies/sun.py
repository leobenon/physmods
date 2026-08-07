"""Simplified solar position model for rigid-Earth simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astrodynamics.constants import DEG_TO_RAD
from astrodynamics.frames.dcm import (
    transform_inertial_to_body,
)


Vector3 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Sun:
    """Parameters for a simplified circular Earth-Sun model.

    The Sun is represented as moving on a circular geocentric orbit.
    This is equivalent, for the present torque calculation, to using
    the Earth's circular heliocentric orbit and reversing the position
    vector.

    The inertial x-y plane is the ecliptic plane.
    """

    gravitational_parameter: float = 1.32712440018e20
    distance: float = 149_597_870_700.0

    orbital_period: float = (
        365.256363004 * 86_400.0
    )

    ecliptic_obliquity: float = (
        23.4392911 * DEG_TO_RAD
    )

    phase_at_epoch: float = 0.0

    # Elliptical Earth-orbit parameters
    semi_major_axis: float = 149_597_870_700.0
    eccentricity: float = 0.0167086

    orbit_model: str = "circular"

    @property
    def angular_rate(self) -> float:
        """Return the constant orbital angular rate in rad/s."""
        return 2.0 * np.pi / self.orbital_period

    def _eccentric_anomaly(
        self,
        mean_anomaly: float,
        *,
        tolerance: float = 1.0e-13,
        maximum_iterations: int = 50,
    ) -> float:
        """Solve Kepler's equation for the eccentric anomaly."""
        mean_anomaly = mean_anomaly % (2.0 * np.pi)

        eccentric_anomaly = mean_anomaly

        for _ in range(maximum_iterations):
            function_value = (
                eccentric_anomaly
                - self.eccentricity * np.sin(eccentric_anomaly)
                - mean_anomaly
            )

            derivative = (
                1.0
                - self.eccentricity * np.cos(eccentric_anomaly)
            )

            correction = function_value / derivative

            eccentric_anomaly -= correction

            if abs(correction) < tolerance:
                return eccentric_anomaly

        raise RuntimeError(
            "Kepler equation did not converge."
        )
    
    def _position_ecliptic_circular(
        self,
        time: float,
    ) -> Vector3:
        """Return the circular geocentric Sun position in ecliptic coordinates."""
        phase = (
            self.phase_at_epoch
            + self.angular_rate * time
        )

        return self.distance * np.array(
            [
                np.cos(phase),
                np.sin(phase),
                0.0,
            ],
            dtype=float,
        )
    
    def _position_ecliptic_elliptical(
        self,
        time: float,
    ) -> Vector3:
        """Return the elliptical geocentric Sun position in ecliptic coordinates."""
        mean_anomaly = (
            self.phase_at_epoch
            + self.angular_rate * time
        )

        eccentric_anomaly = self._eccentric_anomaly(
            mean_anomaly
        )

        x_ecliptic = (
            self.semi_major_axis
            * (
                np.cos(eccentric_anomaly)
                - self.eccentricity
            )
        )

        y_ecliptic = (
            self.semi_major_axis
            * np.sqrt(
                1.0 - self.eccentricity**2
            )
            * np.sin(eccentric_anomaly)
        )

        return np.array(
            [
                x_ecliptic,
                y_ecliptic,
                0.0,
            ],
            dtype=float,
        )
    
    def position_ecliptic(
        self,
        time: float,
    ) -> Vector3:
        """Return the geocentric Sun position in ecliptic coordinates."""
        if self.orbit_model == "circular":
            return self._position_ecliptic_circular(
                time
            )

        if self.orbit_model == "elliptical":
            return self._position_ecliptic_elliptical(
                time
            )

        raise ValueError(
            "orbit_model must be 'circular' or 'elliptical', "
            f"but received {self.orbit_model!r}."
        )

    def position_inertial(
        self,
        time: float,
    ) -> Vector3:
        """Return the Sun position in the equatorial inertial frame."""
        position_ecliptic = self.position_ecliptic(
            time
        )

        cosine = np.cos(
            self.ecliptic_obliquity
        )
        sine = np.sin(
            self.ecliptic_obliquity
        )

        ecliptic_to_equatorial = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ],
            dtype=float,
        )

        return (
            ecliptic_to_equatorial
            @ position_ecliptic
        )

    def position_body_fixed(
        self,
        *,
        time: float,
        sidereal_angle: float,
    ) -> Vector3:
        """Return the Sun position in the Earth-fixed body frame."""
        return transform_inertial_to_body(
            vector_inertial=self.position_inertial(
                time
            ),
            sidereal_angle=sidereal_angle,
            lunar_orbit_inclination=0.0,
        )


SUN = Sun()