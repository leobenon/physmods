"""Tests for the simplified Moon model."""

from dataclasses import replace

import numpy as np

from astrodynamics.bodies.moon import MOON, Moon
from astrodynamics.frames.dcm import transform_inertial_to_body


def test_default_moon_parameters_match_matlab() -> None:
    assert np.isclose(
        MOON.gravitational_parameter,
        398.6e12 / 81.3,
    )
    assert np.isclose(
        MOON.orbital_radius,
        3.8e8,
    )
    assert np.isclose(
        MOON.orbital_angular_rate,
        2.661707223e-6,
    )
    assert np.isclose(
        MOON.orbital_inclination,
        np.deg2rad(28.0),
    )


def test_position_at_zero_time() -> None:
    position = MOON.position_inertial(0.0)

    expected = np.array(
        [
            MOON.orbital_radius,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(
        position,
        expected,
        atol=1.0e-8,
    )


def test_position_after_quarter_orbit() -> None:
    time = MOON.orbital_period / 4.0

    position = MOON.position_inertial(time)

    expected = np.array(
        [
            0.0,
            MOON.orbital_radius
            * np.cos(MOON.orbital_inclination),
            MOON.orbital_radius
            * np.sin(MOON.orbital_inclination),
        ]
    )

    assert np.allclose(
        position,
        expected,
        atol=1.0e-6,
    )


def test_position_after_half_orbit() -> None:
    time = MOON.orbital_period / 2.0

    position = MOON.position_inertial(time)

    expected = np.array(
        [
            -MOON.orbital_radius,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(
        position,
        expected,
        atol=1.0e-6,
    )


def test_circular_orbit_has_constant_radius() -> None:
    moon = replace(
        MOON,
        orbit_model="circular",
    )

    times = np.linspace(
        0.0,
        moon.orbital_period,
        100,
    )

    distances = np.array(
        [
            np.linalg.norm(
                moon.position_inertial(time)
            )
            for time in times
        ]
    )

    assert np.allclose(
        distances,
        moon.orbital_radius,
        rtol=1.0e-12,
    )


def test_circular_orbit_is_inclined_in_inertial_frame() -> None:
    moon = replace(
        MOON,
        orbit_model="circular",
    )

    position = moon.position_inertial(
        0.25 * moon.orbital_period
    )

    assert not np.isclose(
        position[2],
        0.0,
        atol=1.0e-12,
    )


def test_custom_moon_parameters() -> None:
    moon = Moon(
        gravitational_parameter=1.0,
        orbital_radius=10.0,
        orbital_angular_rate=2.0,
        orbital_inclination=0.25,
    )

    time = np.pi / 4.0

    position = moon.position_inertial(
        time
    )

    expected = np.array(
        [
            0.0,
            moon.orbital_radius
            * np.cos(moon.orbital_inclination),
            moon.orbital_radius
            * np.sin(moon.orbital_inclination),
        ]
    )

    assert np.allclose(
        position,
        expected,
        atol=1.0e-12,
    )


def test_body_fixed_position_matches_explicit_transformation() -> None:
    time = 1.5e5
    sidereal_angle = 0.9

    position_inertial = MOON.position_inertial(
        time
    )

    expected = transform_inertial_to_body(
        vector_inertial=position_inertial,
        sidereal_angle=sidereal_angle,
        lunar_orbit_inclination=0.0,
    )

    actual = MOON.position_body_fixed(
        time=time,
        sidereal_angle=sidereal_angle,
    )

    assert np.allclose(
        actual,
        expected,
    )


def test_elliptical_orbit_has_correct_perigee() -> None:
    moon = replace(
        MOON,
        orbit_model="elliptical",
        mean_anomaly_at_epoch=0.0,
    )

    distance = np.linalg.norm(
        moon.position_inertial(0.0)
    )

    expected = (
        moon.semi_major_axis
        * (1.0 - moon.eccentricity)
    )

    assert np.isclose(
        distance,
        expected,
        rtol=1.0e-12,
    )


def test_elliptical_orbit_has_correct_apogee() -> None:
    moon = replace(
        MOON,
        orbit_model="elliptical",
        mean_anomaly_at_epoch=0.0,
    )

    distance = np.linalg.norm(
        moon.position_inertial(
            0.5 * moon.orbital_period
        )
    )

    expected = (
        moon.semi_major_axis
        * (1.0 + moon.eccentricity)
    )

    assert np.isclose(
        distance,
        expected,
        rtol=1.0e-10,
    )


def test_elliptical_orbit_repeats_after_one_period() -> None:
    moon = replace(
        MOON,
        orbit_model="elliptical",
    )

    position_0 = moon.position_inertial(
        0.0
    )

    position_1 = moon.position_inertial(
        moon.orbital_period
    )

    assert np.allclose(
        position_0,
        position_1,
        rtol=1.0e-12,
        atol=1.0e-3,
    )