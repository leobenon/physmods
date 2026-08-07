"""Tests for the rigid-Earth simulation interface."""

import numpy as np
import pytest


from astrodynamics.bodies.earth import EARTH
from astrodynamics.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from astrodynamics.simulation import (
    default_initial_state,
    simulate_rigid_earth,
)
from astrodynamics.dynamics.earth_rotation import (
    rigid_earth_state_derivative,
)


def short_result():
    """Return a small simulation result for derived-history tests."""
    return simulate_rigid_earth(
        duration_days=0.01,
        output_step=300.0,
        max_step=300.0,
    )


def test_default_initial_state_matches_matlab() -> None:
    state = default_initial_state()

    expected = np.array(
        [
            1.0e-6 * EARTH.rotation_rate,
            0.0,
            EARTH.rotation_rate,
            0.0,
            np.deg2rad(23.5),
            0.0,
        ]
    )

    assert state.shape == (6,)
    assert np.allclose(state, expected, atol=1.0e-20)


def test_short_simulation_succeeds() -> None:
    result = simulate_rigid_earth(
        duration_days=0.1,
        output_step=600.0,
        max_step=600.0,
    )

    assert result.success
    assert result.time.ndim == 1
    assert result.state.ndim == 2
    assert result.state.shape[1] == 6
    assert result.state.shape[0] == result.time.size
    assert np.all(np.isfinite(result.state))


def test_initial_state_is_preserved_at_first_output() -> None:
    initial_state = default_initial_state()

    result = simulate_rigid_earth(
        duration_days=0.01,
        output_step=60.0,
        max_step=60.0,
        initial_state=initial_state,
    )

    assert np.allclose(
        result.state[0],
        initial_state,
        atol=1.0e-20,
    )


def test_final_time_matches_requested_duration() -> None:
    duration_days = 0.125

    result = simulate_rigid_earth(
        duration_days=duration_days,
        output_step=SECONDS_PER_HOUR,
        max_step=SECONDS_PER_HOUR,
    )

    assert np.isclose(
        result.time[-1],
        duration_days * SECONDS_PER_DAY,
        atol=1.0e-12,
    )


def test_result_properties_have_expected_shapes() -> None:
    result = simulate_rigid_earth(
        duration_days=0.05,
        output_step=600.0,
        max_step=600.0,
    )

    number_of_samples = result.time.size

    assert result.angular_velocity_body.shape == (
        number_of_samples,
        3,
    )
    assert result.psi.shape == (number_of_samples,)
    assert result.obliquity.shape == (number_of_samples,)
    assert result.sidereal_angle.shape == (number_of_samples,)
    assert result.time_days.shape == (number_of_samples,)


def test_axisymmetric_model_keeps_omega_3_constant() -> None:
    result = simulate_rigid_earth(
        duration_days=0.2,
        output_step=600.0,
        max_step=600.0,
    )

    omega_3 = result.angular_velocity_body[:, 2]

    assert np.allclose(
        omega_3,
        omega_3[0],
        rtol=1.0e-12,
        atol=1.0e-15,
    )


@pytest.mark.parametrize(
    ("argument_name", "argument_value"),
    [
        ("duration_days", 0.0),
        ("duration_days", -1.0),
        ("output_step", 0.0),
        ("output_step", -10.0),
        ("max_step", 0.0),
        ("max_step", -10.0),
    ],
)
def test_invalid_positive_argument_raises_error(
    argument_name: str,
    argument_value: float,
) -> None:
    arguments = {
        "duration_days": 0.1,
        "output_step": 600.0,
        "max_step": 600.0,
    }

    arguments[argument_name] = argument_value

    with pytest.raises(ValueError, match="positive"):
        simulate_rigid_earth(**arguments)


def test_invalid_initial_state_shape_raises_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        simulate_rigid_earth(
            duration_days=0.1,
            initial_state=np.zeros(5),
        )
def test_angular_momentum_axis_has_unit_norm() -> None:
    result = short_result()

    momentum_magnitudes = np.linalg.norm(
        result.angular_momentum_body,
        axis=1,
    )

    axis_norms = np.linalg.norm(
        result.angular_momentum_axis_body,
        axis=1,
    )

    nonzero = momentum_magnitudes > 0.0

    assert np.allclose(
        axis_norms[nonzero],
        1.0,
        atol=1.0e-14,
    )

def test_angular_momentum_matches_inertia_times_omega() -> None:
    result = short_result()

    expected = (
        result.angular_velocity_body
        * EARTH.principal_moments
    )

    assert np.allclose(
        result.angular_momentum_body,
        expected,
        rtol=1.0e-14,
        atol=0.0,
    )

def test_rotation_and_momentum_axes_differ_when_transverse_spin_exists() -> None:
    result = short_result()

    angular_velocity = result.angular_velocity_body

    expected_momentum = (
        angular_velocity * EARTH.principal_moments
    )

    expected_axis = (
        expected_momentum
        / np.linalg.norm(
            expected_momentum,
            axis=1,
            keepdims=True,
        )
    )

    assert np.allclose(
        result.angular_momentum_axis_body,
        expected_axis,
        rtol=1.0e-14,
        atol=1.0e-15,
    )

def test_lunar_torque_can_be_disabled() -> None:
    result = simulate_rigid_earth(
        duration_days=0.01,
        output_step=300.0,
        max_step=300.0,
        include_lunar_torque=False,
    )

    assert result.include_lunar_torque is False

    assert np.allclose(
        result.normalized_lunar_torque,
        0.0,
        atol=0.0,
    )
def test_lunar_torque_is_enabled_by_default() -> None:
    result = simulate_rigid_earth(
        duration_days=0.01,
        output_step=300.0,
        max_step=300.0,
    )

    assert result.include_lunar_torque is True

def test_lunar_torque_changes_the_simulation() -> None:
    with_torque = simulate_rigid_earth(
        duration_days=5.0,
        output_step=3600.0,
        max_step=3600.0,
        include_lunar_torque=True,
    )

    without_torque = simulate_rigid_earth(
        duration_days=5.0,
        output_step=3600.0,
        max_step=3600.0,
        include_lunar_torque=False,
    )

    assert not np.array_equal(
        with_torque.state,
        without_torque.state,
    )

def test_solar_torque_is_disabled_by_default() -> None:
    result = short_result()

    assert result.include_solar_torque is False

    assert np.allclose(
        result.normalized_solar_torque,
        0.0,
        atol=0.0,
    )

def test_solar_torque_can_be_enabled() -> None:
    result = simulate_rigid_earth(
        duration_days=0.1,
        output_step=600.0,
        max_step=600.0,
        include_lunar_torque=False,
        include_solar_torque=True,
    )

    assert result.include_solar_torque is True

    assert np.any(
        result.normalized_solar_torque != 0.0
    )

def test_total_torque_is_sum_of_contributions() -> None:
    result = simulate_rigid_earth(
        duration_days=0.1,
        output_step=600.0,
        max_step=600.0,
        include_lunar_torque=True,
        include_solar_torque=True,
    )

    assert np.allclose(
        result.normalized_total_torque,
        (
            result.normalized_lunar_torque
            + result.normalized_solar_torque
        ),
        rtol=1.0e-14,
        atol=1.0e-30,
    )

def test_all_torques_are_zero_when_disabled() -> None:
    result = simulate_rigid_earth(
        duration_days=0.1,
        output_step=600.0,
        max_step=600.0,
        include_lunar_torque=False,
        include_solar_torque=False,
    )

    assert np.allclose(
        result.normalized_total_torque,
        0.0,
        atol=0.0,
    )
def test_combined_weak_torque_response_is_nearly_additive() -> None:
    settings = {
        "duration_days": 5.0,
        "output_step": 3600.0,
        "max_step": 3600.0,
    }

    no_torque = simulate_rigid_earth(
        **settings,
        include_lunar_torque=False,
        include_solar_torque=False,
    )

    moon_only = simulate_rigid_earth(
        **settings,
        include_lunar_torque=True,
        include_solar_torque=False,
    )

    sun_only = simulate_rigid_earth(
        **settings,
        include_lunar_torque=False,
        include_solar_torque=True,
    )

    combined = simulate_rigid_earth(
        **settings,
        include_lunar_torque=True,
        include_solar_torque=True,
    )

    expected_combined_change = (
        moon_only.state
        + sun_only.state
        - no_torque.state
    )

    assert np.allclose(
        combined.state,
        expected_combined_change,
        rtol=1.0e-8,
        atol=1.0e-12,
    )