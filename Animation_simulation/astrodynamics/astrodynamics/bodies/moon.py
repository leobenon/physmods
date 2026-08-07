"""Moon models used by the rigid-Earth simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astrodynamics.constants import DEG_TO_RAD
from astrodynamics.frames.dcm import transform_inertial_to_body
from astrodynamics.frames.rotation import (
    rotation_x,
    rotation_z,
)


Vector3 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Moon:
    """Parameters and position models for the Moon.

    Two orbit models are supported:

    ``"circular"``
        Reproduces the simplified circular orbit from the original
        MATLAB model.

    ``"elliptical"``
        Uses a fixed Keplerian ellipse with lunar eccentricity.
    """

    gravitational_parameter: float = 398.6e12 / 81.3

    # Circular-model parameters
    orbital_radius: float = 3.8e8
    orbital_angular_rate: float = 2.661707223e-6

    # Shared / simplified orbital orientation
    orbital_inclination: float = 28.0 * DEG_TO_RAD

    # Elliptical-model parameters
    semi_major_axis: float = 384_400_000.0
    eccentricity: float = 0.0549

    longitude_of_ascending_node: float = 0.0
    argument_of_periapsis: float = 0.0
    mean_anomaly_at_epoch: float = 0.0

    orbit_model: str = "circular"

    @property
    def orbital_period(self) -> float:
        """Return the orbital period in seconds."""
        return (
            2.0 * np.pi
            / self.orbital_angular_rate
        )

    def _mean_anomaly(
        self,
        time: float,
    ) -> float:
        """Return the mean anomaly for the elliptical model."""
        return (
            self.mean_anomaly_at_epoch
            + self.orbital_angular_rate * time
        )

    def _eccentric_anomaly(
        self,
        mean_anomaly: float,
        *,
        tolerance: float = 1.0e-13,
        maximum_iterations: int = 50,
    ) -> float:
        """Solve Kepler's equation M = E - e sin(E)."""
        mean_anomaly = (
            mean_anomaly % (2.0 * np.pi)
        )

        eccentric_anomaly = mean_anomaly

        for _ in range(maximum_iterations):
            function_value = (
                eccentric_anomaly
                - self.eccentricity
                * np.sin(eccentric_anomaly)
                - mean_anomaly
            )

            derivative = (
                1.0
                - self.eccentricity
                * np.cos(eccentric_anomaly)
            )

            correction = (
                function_value / derivative
            )

            eccentric_anomaly -= correction

            if abs(correction) < tolerance:
                return eccentric_anomaly

        raise RuntimeError(
            "Kepler equation did not converge."
        )

    def _circular_position_orbital(
        self,
        time: float,
    ) -> Vector3:
        """Return circular-orbit position in the orbital plane."""
        phase = (
            self.orbital_angular_rate * time
        )

        return np.array(
            [
                self.orbital_radius
                * np.cos(phase),

                self.orbital_radius
                * np.sin(phase),

                0.0,
            ],
            dtype=float,
        )

    def _elliptical_position_orbital(
        self,
        time: float,
    ) -> Vector3:
        """Return Keplerian position in the orbital plane."""
        mean_anomaly = self._mean_anomaly(
            time
        )

        eccentric_anomaly = (
            self._eccentric_anomaly(
                mean_anomaly
            )
        )

        x_orbital = (
            self.semi_major_axis
            * (
                np.cos(eccentric_anomaly)
                - self.eccentricity
            )
        )

        y_orbital = (
            self.semi_major_axis
            * np.sqrt(
                1.0
                - self.eccentricity**2
            )
            * np.sin(eccentric_anomaly)
        )

        return np.array(
            [
                x_orbital,
                y_orbital,
                0.0,
            ],
            dtype=float,
        )

    def _orbital_to_inertial(
        self,
        position_orbital: Vector3,
    ) -> Vector3:
        """Rotate an orbital-plane position into the inertial frame."""
        return (
            rotation_z(
                self.longitude_of_ascending_node
            )
            @ rotation_x(
                self.orbital_inclination
            )
            @ rotation_z(
                self.argument_of_periapsis
            )
            @ position_orbital
        )

    def position_inertial(
        self,
        time: float,
    ) -> Vector3:
        """Return Moon position in the inertial reference frame."""
        if self.orbit_model == "circular":
            position_orbital = (
                self._circular_position_orbital(
                    time
                )
            )

        elif self.orbit_model == "elliptical":
            position_orbital = (
                self._elliptical_position_orbital(
                    time
                )
            )

        else:
            raise ValueError(
                "orbit_model must be "
                "'circular' or 'elliptical', "
                f"but received "
                f"{self.orbit_model!r}."
            )

        return self._orbital_to_inertial(
            position_orbital
        )

    def position_body_fixed(
        self,
        *,
        time: float,
        sidereal_angle: float,
    ) -> Vector3:
        """Return Moon position in the Earth body-fixed frame."""
        position_inertial = (
            self.position_inertial(time)
        )

        return transform_inertial_to_body(
            vector_inertial=position_inertial,
            sidereal_angle=sidereal_angle,

            # Inclination is already included
            # in position_inertial().
            lunar_orbit_inclination=0.0,
        )


MOON = Moon()