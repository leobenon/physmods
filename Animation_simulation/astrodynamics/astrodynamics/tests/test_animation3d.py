"""Tests for three-dimensional rigid-Earth visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from astrodynamics.simulation import simulate_rigid_earth
from astrodynamics.visualization.animation3d import (
    RigidEarthViewer,
    exaggerate_direction,
    normalize_vector,
    plot_rigid_earth_state_3d,
)


def short_result():
    return simulate_rigid_earth(
        duration_days=0.01,
        output_step=300.0,
        max_step=300.0,
    )


def test_normalize_vector_returns_unit_vector() -> None:
    vector = np.array([3.0, 4.0, 0.0])

    normalized = normalize_vector(vector)

    assert np.allclose(
        normalized,
        np.array([0.6, 0.8, 0.0]),
        atol=1.0e-15,
    )

    assert np.isclose(
        np.linalg.norm(normalized),
        1.0,
        atol=1.0e-15,
    )


def test_normalize_zero_vector_returns_zero() -> None:
    normalized = normalize_vector(np.zeros(3))

    assert np.array_equal(
        normalized,
        np.zeros(3),
    )


def test_invalid_vector_shape_raises_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_vector(np.array([1.0, 2.0]))


def test_plot_returns_figure_and_3d_axis() -> None:
    result = short_result()

    figure, axis = plot_rigid_earth_state_3d(
        result,
        sample_index=0,
    )

    assert figure is not None
    assert axis.name == "3d"

    plt.close(figure)


def test_negative_sample_index_is_supported() -> None:
    result = short_result()

    figure, axis = plot_rigid_earth_state_3d(
        result,
        sample_index=-1,
    )

    assert axis.name == "3d"

    plt.close(figure)


def test_invalid_sample_index_raises_error() -> None:
    result = short_result()

    with pytest.raises(IndexError, match="outside"):
        plot_rigid_earth_state_3d(
            result,
            sample_index=result.time.size,
        )


def test_viewer_initializes_at_requested_index() -> None:
    result = short_result()

    viewer = RigidEarthViewer(
        result,
        initial_index=1,
        rotation_axis_exaggeration=1.0e5,
    )

    assert viewer.current_index == 1
    assert viewer.axis.name == "3d"
    assert viewer.time_slider.val == result.time_days[1]

    plt.close(viewer.figure)

def test_viewer_set_index_updates_current_state() -> None:
    result = short_result()

    viewer = RigidEarthViewer(result)

    viewer.set_index(-1)

    assert viewer.current_index == result.time.size - 1

    plt.close(viewer.figure)

def test_viewer_exaggeration_can_be_changed() -> None:
    result = short_result()

    viewer = RigidEarthViewer(
        result,
        rotation_axis_exaggeration=1.0,
    )

    viewer.set_rotation_axis_exaggeration(1.0e6)

    assert viewer.rotation_axis_exaggeration == 1.0e6

    plt.close(viewer.figure)

def test_invalid_viewer_exaggeration_raises_error() -> None:
    result = short_result()

    with pytest.raises(ValueError, match="positive"):
        RigidEarthViewer(
            result,
            rotation_axis_exaggeration=0.0,
        )
def test_exaggeration_factor_one_preserves_direction() -> None:
    direction = np.array([0.1, 0.2, 1.0])
    reference = np.array([0.0, 0.0, 1.0])

    actual = exaggerate_direction(
        direction=direction,
        reference_axis=reference,
        factor=1.0,
    )

    expected = normalize_vector(direction)

    assert np.allclose(actual, expected, atol=1.0e-15)


def test_exaggeration_increases_transverse_tilt() -> None:
    direction = normalize_vector(
        np.array([1.0e-6, 0.0, 1.0])
    )
    reference = np.array([0.0, 0.0, 1.0])

    exaggerated = exaggerate_direction(
        direction=direction,
        reference_axis=reference,
        factor=1.0e5,
    )

    assert abs(exaggerated[0]) > abs(direction[0])
    assert np.isclose(
        np.linalg.norm(exaggerated),
        1.0,
        atol=1.0e-15,
    )


def test_invalid_exaggeration_factor_raises_error() -> None:
    with pytest.raises(ValueError, match="positive"):
        exaggerate_direction(
            direction=np.array([0.0, 0.0, 1.0]),
            reference_axis=np.array([0.0, 0.0, 1.0]),
            factor=0.0,
        )

def test_viewer_play_and_pause() -> None:
    result = short_result()

    viewer = RigidEarthViewer(result)

    viewer.play()

    assert viewer.is_playing
    assert viewer.play_button.label.get_text() == "Pause"

    viewer.pause()

    assert not viewer.is_playing
    assert viewer.play_button.label.get_text() == "Play"

    plt.close(viewer.figure)

def test_advance_frame_wraps_to_start() -> None:
    result = short_result()

    viewer = RigidEarthViewer(
        result,
        initial_index=-1,
    )

    viewer.is_playing = True
    viewer._advance_frame()

    assert viewer.current_index == 0

    plt.close(viewer.figure)

def test_playback_interval_can_be_changed() -> None:
    result = short_result()

    viewer = RigidEarthViewer(result)

    viewer.set_playback_interval(100)

    assert viewer.timer_interval_ms == 100
    assert viewer.timer.interval == 100

    plt.close(viewer.figure)

def test_invalid_playback_interval_raises_error() -> None:
    result = short_result()

    viewer = RigidEarthViewer(result)

    with pytest.raises(ValueError, match="positive"):
        viewer.set_playback_interval(0)

    plt.close(viewer.figure)

def test_playback_speed_can_be_changed() -> None:
    result = short_result()

    viewer = RigidEarthViewer(result)

    viewer.set_playback_speed(20.0)

    assert viewer.speed_slider.val == 20.0
    assert viewer.timer_interval_ms == 50

    plt.close(viewer.figure)

def test_invalid_playback_speed_raises_error() -> None:
    result = short_result()

    viewer = RigidEarthViewer(result)

    with pytest.raises(ValueError, match="positive"):
        viewer.set_playback_speed(0.0)

    plt.close(viewer.figure)

def test_visibility_can_be_toggled() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.set_visibility("moon_direction", False)

    assert not viewer.visibility["moon_direction"]

    plt.close(viewer.figure)

def test_unknown_visibility_element_raises_error() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    with pytest.raises(ValueError, match="Unknown"):
        viewer.set_visibility("not_an_element", False)

    plt.close(viewer.figure)

def test_rotation_axis_trail_points_have_correct_shape() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    points = viewer._rotation_axis_trail_points(
        result.time.size - 1
    )

    assert points.shape == (result.time.size, 3)

    plt.close(viewer.figure)

def test_rotation_axis_trail_length_is_limited() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.set_rotation_axis_trail_length(2)

    points = viewer._rotation_axis_trail_points(
        result.time.size - 1
    )

    assert points.shape == (2, 3)

    plt.close(viewer.figure)

def test_rotation_axis_trail_can_be_disabled() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.set_rotation_axis_trail_enabled(False)

    assert not viewer.trails["rotation_axis"]["enabled"]

    plt.close(viewer.figure)

def test_invalid_rotation_axis_trail_length_raises_error() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    with pytest.raises(ValueError, match="at least 1"):
        viewer.set_rotation_axis_trail_length(0)

    plt.close(viewer.figure)

def test_trail_length_slider_updates_setting() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.trail_length_slider.set_val(2)

    assert viewer.trails["rotation_axis"]["length"] == 2

    plt.close(viewer.figure)

def test_public_trail_length_setter_updates_slider() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.set_rotation_axis_trail_length(2)

    assert viewer.trail_length_slider.val == 2
    assert viewer.trails["rotation_axis"]["length"] == 2

    plt.close(viewer.figure)

def test_rotation_axis_trail_visibility_can_be_changed() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.set_rotation_axis_trail_enabled(False)

    assert not viewer.visibility["rotation_axis_trail"]
    assert not viewer.trails["rotation_axis"]["enabled"]

    plt.close(viewer.figure)

def test_trail_can_remain_visible_without_current_axis() -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    viewer.set_visibility("rotation_axis", False)

    assert not viewer.visibility["rotation_axis"]
    assert viewer.visibility["rotation_axis_trail"]

    plt.close(viewer.figure)

def test_save_animation_rejects_unknown_extension(
    tmp_path,
) -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    with pytest.raises(ValueError, match="mp4.*gif"):
        viewer.save_animation(
            tmp_path / "animation.avi",
        )

    plt.close(viewer.figure)

def test_save_animation_rejects_invalid_fps(
    tmp_path,
) -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    with pytest.raises(ValueError, match="fps must be positive"):
        viewer.save_animation(
            tmp_path / "animation.gif",
            fps=0,
        )

    plt.close(viewer.figure)

def test_save_animation_rejects_invalid_frame_step(
    tmp_path,
) -> None:
    result = short_result()
    viewer = RigidEarthViewer(result)

    with pytest.raises(
        ValueError,
        match="frame_step must be positive",
    ):
        viewer.save_animation(
            tmp_path / "animation.gif",
            frame_step=0,
        )

    plt.close(viewer.figure)