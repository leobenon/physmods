"""Tests for the simplified solar position model."""

from __future__ import annotations

import numpy as np

from astrodynamics.bodies.sun import SUN

import pytest
from dataclasses import replace


def test_circular_sun_inertial_position_has_constant_distance() -> None:
    sun = replace(
        SUN,
        orbit_model="circular",
    )

    times = np.array(
        [
            0.0,
            30.0 * 86_400.0,
            180.0 * 86_400.0,
            365.0 * 86_400.0,
        ]
    )

    distances = np.array(
        [
            np.linalg.norm(
                sun.position_inertial(time)
            )
            for time in times
        ]
    )

    assert np.allclose(
        distances,
        sun.distance,
        rtol=1.0e-14,
        atol=1.0e-3,
    )


def test_circular_sun_position_repeats_after_one_orbit() -> None:
    sun = replace(
        SUN,
        orbit_model="circular",
    )

    initial_position = sun.position_inertial(
        0.0
    )

    final_position = sun.position_inertial(
        sun.orbital_period
    )

    assert np.allclose(
        final_position,
        initial_position,
        rtol=1.0e-14,
        atol=1.0e-3,
    )


def test_sun_body_position_has_correct_shape() -> None:
    position = SUN.position_body_fixed(
        time=0.0,
        sidereal_angle=0.0,
    )

    assert position.shape == (3,)


def test_sun_angular_rate_is_positive() -> None:
    assert SUN.angular_rate > 0.0

def test_elliptical_sun_has_correct_perihelion() -> None:
    sun = replace(
        SUN,
        orbit_model="elliptical",
        phase_at_epoch=0.0,
    )

    distance = np.linalg.norm(
        sun.position_inertial(0.0)
    )

    expected = (
        sun.semi_major_axis
        * (1.0 - sun.eccentricity)
    )

    assert np.isclose(
        distance,
        expected,
        rtol=1.0e-12,
    )

def test_elliptical_sun_has_correct_aphelion() -> None:
    sun = replace(
        SUN,
        orbit_model="elliptical",
        phase_at_epoch=0.0,
    )

    distance = np.linalg.norm(
        sun.position_inertial(
            0.5 * sun.orbital_period
        )
    )

    expected = (
        sun.semi_major_axis
        * (1.0 + sun.eccentricity)
    )

    assert np.isclose(
        distance,
        expected,
        rtol=1.0e-12,
    )

def test_elliptical_sun_position_repeats_after_one_orbit() -> None:
    sun = replace(
        SUN,
        orbit_model="elliptical",
    )

    initial_position = sun.position_inertial(
        0.0
    )

    final_position = sun.position_inertial(
        sun.orbital_period
    )

    assert np.allclose(
        final_position,
        initial_position,
        rtol=1.0e-12,
        atol=1.0e-3,
    )

def test_elliptical_sun_distance_varies() -> None:
    sun = replace(
        SUN,
        orbit_model="elliptical",
    )

    times = np.linspace(
        0.0,
        sun.orbital_period,
        100,
    )

    distances = np.array(
        [
            np.linalg.norm(
                sun.position_inertial(time)
            )
            for time in times
        ]
    )

    assert np.ptp(distances) > 0.0

def test_invalid_sun_orbit_model_raises_error() -> None:
    sun = replace(
        SUN,
        orbit_model="invalid",
    )


    with pytest.raises(
        ValueError,
        match="orbit_model",
    ):
        sun.position_inertial(0.0)