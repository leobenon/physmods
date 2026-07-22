"""Tests for derived rigid-Earth simulation quantities."""

import numpy as np

from astrodynamics.bodies.moon import MOON
from astrodynamics.simulation import simulate_rigid_earth


def short_result():
    return simulate_rigid_earth(
        duration_days=0.02,
        output_step=300.0,
        max_step=300.0,
    )


def test_derived_histories_have_expected_shapes() -> None:
    result = short_result()
    number_of_samples = result.time.size

    histories = (
        result.moon_position_inertial,
        result.moon_position_body,
        result.normalized_lunar_torque,
        result.angular_acceleration_body,
        result.rotation_axis_body,
        result.figure_axis_body,
    )

    for history in histories:
        assert history.shape == (number_of_samples, 3)
        assert np.all(np.isfinite(history))


def test_moon_distance_is_preserved_between_frames() -> None:
    result = short_result()

    inertial_distance = np.linalg.norm(
        result.moon_position_inertial,
        axis=1,
    )
    body_distance = np.linalg.norm(
        result.moon_position_body,
        axis=1,
    )

    assert np.allclose(
        inertial_distance,
        MOON.orbital_radius,
        rtol=1.0e-12,
    )

    assert np.allclose(
        body_distance,
        inertial_distance,
        rtol=1.0e-12,
    )


def test_rotation_axis_is_unit_length() -> None:
    result = short_result()

    axis_norm = np.linalg.norm(
        result.rotation_axis_body,
        axis=1,
    )

    assert np.allclose(
        axis_norm,
        1.0,
        atol=1.0e-12,
    )


def test_figure_axis_is_body_third_axis() -> None:
    result = short_result()

    expected = np.tile(
        np.array([0.0, 0.0, 1.0]),
        (result.time.size, 1),
    )

    assert np.allclose(
        result.figure_axis_body,
        expected,
        atol=0.0,
    )


def test_third_normalized_torque_is_zero() -> None:
    result = short_result()

    assert np.all(
        result.normalized_lunar_torque[:, 2] == 0.0
    )


def test_angular_acceleration_matches_state_equation() -> None:
    result = short_result()

    # The first three state derivatives are the angular accelerations.
    finite_difference = np.gradient(
        result.angular_velocity_body,
        result.time,
        axis=0,
    )

    assert np.allclose(
        finite_difference[1:-1],
        result.angular_acceleration_body[1:-1],
        rtol=5.0e-3,
        atol=1.0e-16,
    )