"""Tests for the simplified Moon model."""

import numpy as np

from astrodynamics.bodies.moon import MOON, Moon
from astrodynamics.frames.rotation import rotation_x, rotation_z

def test_default_moon_parameters_match_matlab() -> None:
    assert np.isclose(MOON.gravitational_parameter, 398.6e12 / 81.3)
    assert np.isclose(MOON.orbital_radius, 3.8e8)
    assert np.isclose(MOON.orbital_angular_rate, 2.661707223e-6)
    assert np.isclose(MOON.orbital_inclination, np.deg2rad(28.0))


def test_position_at_zero_time() -> None:
    position = MOON.position_inertial(0.0)

    expected = np.array([MOON.orbital_radius, 0.0, 0.0])

    assert np.allclose(position, expected, atol=1.0e-8)


def test_position_after_quarter_orbit() -> None:
    time = MOON.orbital_period / 4.0

    position = MOON.position_inertial(time)

    expected = np.array([0.0, MOON.orbital_radius, 0.0])

    assert np.allclose(position, expected, atol=1.0e-6)


def test_position_after_half_orbit() -> None:
    time = MOON.orbital_period / 2.0

    position = MOON.position_inertial(time)

    expected = np.array([-MOON.orbital_radius, 0.0, 0.0])

    assert np.allclose(position, expected, atol=1.0e-6)


def test_orbital_radius_is_constant() -> None:
    sample_times = np.linspace(0.0, MOON.orbital_period, 50)

    for time in sample_times:
        position = MOON.position_inertial(time)

        assert np.isclose(
            np.linalg.norm(position),
            MOON.orbital_radius,
            rtol=1.0e-12,
        )


def test_position_remains_in_inertial_xy_plane() -> None:
    sample_times = np.linspace(0.0, MOON.orbital_period, 20)

    for time in sample_times:
        position = MOON.position_inertial(time)

        assert np.isclose(position[2], 0.0)


def test_custom_moon_parameters() -> None:
    moon = Moon(
        gravitational_parameter=1.0,
        orbital_radius=10.0,
        orbital_angular_rate=2.0,
        orbital_inclination=0.25,
    )

    position = moon.position_inertial(np.pi / 4.0)

    expected = np.array([0.0, 10.0, 0.0])

    assert np.allclose(position, expected, atol=1.0e-12)

def test_body_fixed_position_matches_explicit_transformation() -> None:
    time = 1.5e5
    sidereal_angle = 0.9

    position_inertial = MOON.position_inertial(time)

    expected = (
        rotation_z(-sidereal_angle)
        @ rotation_x(MOON.orbital_inclination)
        @ position_inertial
    )

    actual = MOON.position_body_fixed(
        time=time,
        sidereal_angle=sidereal_angle,
    )

    assert np.allclose(actual, expected, atol=1.0e-6)