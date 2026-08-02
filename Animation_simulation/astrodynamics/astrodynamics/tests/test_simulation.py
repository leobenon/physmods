"""Tests for the rigid-Earth simulation interface."""

import numpy as np
import pytest


from astrodynamics.bodies.earth import EARTH
from astrodynamics.constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from astrodynamics.simulation import (
    default_initial_state,
    simulate_rigid_earth,
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