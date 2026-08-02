"""Three-dimensional visualization of rigid-Earth rotation."""

from __future__ import annotations
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter,FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import to_rgba

from astrodynamics.frames.euler import (body_to_inertial_euler_dcm,)
from astrodynamics.simulation import SimulationResult


Vector3 = np.ndarray


def normalize_vector(
    vector: np.ndarray,
    *,
    tolerance: float = 1.0e-15,
) -> np.ndarray:
    """Return a unit vector, or zero for a near-zero input vector."""
    vector = np.asarray(vector, dtype=float)

    if vector.shape != (3,):
        raise ValueError(
            "vector must have shape (3,), "
            f"but received {vector.shape}."
        )

    magnitude = np.linalg.norm(vector)

    if magnitude < tolerance:
        return np.zeros(3, dtype=float)

    return vector / magnitude


def set_axes_equal_3d(
    axis: Axes3D,
    *,
    limit: float = 1.8,
) -> None:
    """Set equal symmetric limits on a three-dimensional axis."""
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)

    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def draw_vector(
    axis: Axes3D,
    vector: np.ndarray,
    *,
    length: float,
    label: str,
    color: str,
    linewidth: float = 2.0,
    arrow_length_ratio: float = 0.12,
    linestyle: str = "-",
    normalization_tolerance: float = 1.0e-15,
) -> None:
    """Draw a normalized vector from the origin."""
    direction = normalize_vector(
        vector,
        tolerance=normalization_tolerance,
    )

    if np.allclose(direction, 0.0):
        return

    plotted_vector = length * direction

    axis.quiver(
        0.0,
        0.0,
        0.0,
        plotted_vector[0],
        plotted_vector[1],
        plotted_vector[2],
        linewidth=linewidth,
        color=color,
        arrow_length_ratio=arrow_length_ratio,
        linestyle=linestyle,
        label=label,
    )


def draw_body_axes(
    axis: Axes3D,
    *,
    directions: np.ndarray | None = None,
    length: float = 1.35,
) -> None:
    """Draw the body-fixed principal axes in the selected frame."""
    if directions is None:
        directions = np.eye(3)

    directions = np.asarray(directions, dtype=float)

    if directions.shape != (3, 3):
        raise ValueError(
            "directions must have shape (3, 3)."
        )

    labels = [
        r"$\mathbf{e}_1^B$",
        r"$\mathbf{e}_2^B$",
        r"$\mathbf{e}_3^B$",
    ]

    colors = [
        "red",
        "green",
        "blue",
    ]

    for index in range(3):
        draw_vector(
            axis,
            directions[:, index],
            length=length,
            label=labels[index],
            color=colors[index],
            linewidth=1.2,
            arrow_length_ratio=0.08,
        )


def draw_oblate_earth(
    axis: Axes3D,
    *,
    equatorial_radius: float = 1.0,
    polar_radius: float = 0.96,
    resolution: int = 60,
) -> None:
    """Draw a dimensionless oblate Earth ellipsoid."""
    longitude = np.linspace(0.0, 2.0 * np.pi, resolution)
    latitude = np.linspace(0.0, np.pi, resolution)

    longitude_grid, latitude_grid = np.meshgrid(
        longitude,
        latitude,
    )

    x_values = (
        equatorial_radius
        * np.sin(latitude_grid)
        * np.cos(longitude_grid)
    )
    y_values = (
        equatorial_radius
        * np.sin(latitude_grid)
        * np.sin(longitude_grid)
    )
    z_values = polar_radius * np.cos(latitude_grid)

    axis.plot_surface(
        x_values,
        y_values,
        z_values,
        alpha=0.20,
        linewidth=0.2,
        antialiased=True,
    )

    equator_angle = np.linspace(0.0, 2.0 * np.pi, 200)

    axis.plot(
        equatorial_radius * np.cos(equator_angle),
        equatorial_radius * np.sin(equator_angle),
        np.zeros_like(equator_angle),
        color="navy",
        linewidth=0.9,
    )


def plot_rigid_earth_state_3d(
    result: SimulationResult,
    sample_index: int = 0,
) -> tuple[Figure, Axes3D]:
    """Plot one rigid-Earth simulation state in the body-fixed frame.

    Parameters
    ----------
    result
        Enriched rigid-Earth simulation result.
    sample_index
        Index of the output sample to visualize.

    Returns
    -------
    tuple
        Matplotlib figure and three-dimensional axis.
    """
    number_of_samples = result.time.size

    if not -number_of_samples <= sample_index < number_of_samples:
        raise IndexError(
            f"sample_index {sample_index} is outside the valid range "
            f"[-{number_of_samples}, {number_of_samples - 1}]."
        )

    sample_index %= number_of_samples

    rotation_axis = result.rotation_axis_body[sample_index]
    figure_axis = result.figure_axis_body[sample_index]
    moon_direction = result.moon_position_body[sample_index]
    torque_direction = result.normalized_lunar_torque[sample_index]

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    draw_oblate_earth(axis)
    draw_body_axes(axis)

    draw_vector(
        axis,
        figure_axis,
        length=1.55,
        color = "black",
        label="Figure axis",
        linewidth=2.4,
        linestyle="--",
    )

    draw_vector(
        axis,
        rotation_axis,
        length=1.65,
        color = "magenta",
        label=r"Rotation axis $\hat{\omega}$",
        linewidth=2.6,
    )

    draw_vector(
        axis,
        moon_direction,
        length=1.75,
        color = "orange",
        label="Moon direction",
        linewidth=2.0,
    )

    draw_vector(
        axis,
        torque_direction,
        length=1.25,
        color = "purple",
        label="Normalized lunar torque",
        linewidth=2.0,
    )

    time_days = result.time_days[sample_index]
    angular_velocity = result.angular_velocity_body[sample_index]
    normalized_torque = result.normalized_lunar_torque[sample_index]

    axis.set_title(
        "Rigid Earth in the Body-Fixed Frame\n"
        f"Time = {time_days:.3f} days"
    )

    axis.set_xlabel(r"$x_B$")
    axis.set_ylabel(r"$y_B$")
    axis.set_zlabel(r"$z_B$")

    set_axes_equal_3d(axis)

    axis.view_init(
        elev=24.0,
        azim=38.0,
    )

    axis.legend(
        loc="upper left",
        fontsize=9,
        
    )

    information = (
        rf"$\omega_B=$"
        f" [{angular_velocity[0]:.3e}, "
        f"{angular_velocity[1]:.3e}, "
        f"{angular_velocity[2]:.3e}] rad/s\n"
        rf"$d_B=$"
        f" [{normalized_torque[0]:.3e}, "
        f"{normalized_torque[1]:.3e}, "
        f"{normalized_torque[2]:.3e}] s$^{{-2}}$"
    )

    figure.text(
        0.03,
        0.03,
        information,
        fontsize=9,
        family="monospace",
    )

    figure.tight_layout()

    return figure, axis


def exaggerate_direction(
    direction: np.ndarray,
    reference_axis: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Exaggerate a direction's component perpendicular to a reference axis.

    A factor of 1 leaves the direction unchanged. Only the displayed
    direction changes; the simulation values remain untouched.
    """
    if factor <= 0.0:
        raise ValueError("factor must be positive.")

    direction_unit = normalize_vector(direction)
    reference_unit = normalize_vector(reference_axis)

    if np.allclose(direction_unit, 0.0):
        return direction_unit

    if np.allclose(reference_unit, 0.0):
        raise ValueError("reference_axis must be non-zero.")

    parallel_component = (
        np.dot(direction_unit, reference_unit)
        * reference_unit
    )

    perpendicular_component = (
        direction_unit - parallel_component
    )

    exaggerated = (
        parallel_component
        + factor * perpendicular_component
    )

    return normalize_vector(exaggerated)

class RigidEarthViewer:
    """Interactive three-dimensional viewer for a rigid-Earth simulation."""

    def __init__(
        self,
        result: SimulationResult,
        *,
        initial_index: int = 0,
        rotation_axis_exaggeration: float = 1.0e6,
        angular_momentum_axis_exaggeration: float | None = None,
        figure_axis_exaggeration: float = 1.0e4,
        elevation: float = 24.0,
        azimuth: float = 38.0,

    ) -> None:
        
        self.reference_frame = "body"
        
        self.result = result
        self.number_of_samples = result.time.size

        if self.number_of_samples == 0:
            raise ValueError("result contains no output samples.")

        if not -self.number_of_samples <= initial_index < self.number_of_samples:
            raise IndexError(
                f"initial_index {initial_index} is outside the valid range."
            )

        if rotation_axis_exaggeration <= 0.0:
            raise ValueError(
                "rotation_axis_exaggeration must be positive."
            )
        
        if angular_momentum_axis_exaggeration is None:
            angular_momentum_axis_exaggeration = (
                rotation_axis_exaggeration
            )

        if angular_momentum_axis_exaggeration <= 0.0:
            raise ValueError(
                "angular_momentum_axis_exaggeration must be positive."
            )

        self.angular_momentum_axis_exaggeration = float(
            angular_momentum_axis_exaggeration
        )

        if figure_axis_exaggeration <= 0.0:
            raise ValueError(
                "figure_axis_exaggeration must be positive."
            )

        self.figure_axis_exaggeration = float(
            figure_axis_exaggeration
        )

        self.current_index = initial_index % self.number_of_samples
        self.rotation_axis_exaggeration = rotation_axis_exaggeration
        self.elevation = float(elevation)
        self.azimuth = float(azimuth)
        self.is_playing = False
        self.animation_step = 1
        self.timer_interval_ms = 50

        self.visibility = {
            "body_axes": True,
            "figure_axis": True,
            "rotation_axis": True,
            "angular_momentum_axis": True,
            "moon_direction": True,
            "torque": True,
            "rotation_axis_trail": True,
            "angular_momentum_trail": True,
            "figure_axis_trail": True,
        }

        self.visibility_label_to_key = {
            "Body axes": "body_axes",
            "Figure axis": "figure_axis",
            "Rotation axis": "rotation_axis",
            "Angular momentum": "angular_momentum_axis",
            "Moon direction": "moon_direction",
            "Torque": "torque",
            "Rotation-axis trail": "rotation_axis_trail",
            "Momentum-axis trail": "angular_momentum_trail",
            "Figure-axis trail": "figure_axis_trail",

        }


        self.trails = {
            "rotation_axis": {
                "enabled": True,
                "length": 1500,
                "color": "magenta",
                "minimum_alpha": 0.05,
                "maximum_alpha": 0.90,
                "minimum_linewidth": 0.3,
                "maximum_linewidth": 1.8,
                "show_marker": True,
                "marker_size": 45.0,
            },
            "angular_momentum_axis": {
                "enabled": True,
                "length": 1500,
                "color": "cyan",
                "minimum_alpha": 0.05,
                "maximum_alpha": 0.90,
                "minimum_linewidth": 0.3,
                "maximum_linewidth": 1.8,
                "show_marker": True,
                "marker_size": 40.0,
            },
            "figure_axis": {
            "enabled": True,
            "length": 1500,
            "color": "black",
            "minimum_alpha": 0.05,
            "maximum_alpha": 0.90,
            "minimum_linewidth": 0.3,
            "maximum_linewidth": 1.8,
            "show_marker": True,
            "marker_size": 40.0,
        },
        }
        
        self.figure = plt.figure(figsize=(13, 8))
        self.axis = self.figure.add_subplot(111, projection="3d")

        self.axis.view_init(
            elev=self.elevation,
            azim=self.azimuth,
        )

        self.figure.subplots_adjust(
            left=0.08,
            right=0.95,
            top=0.90,
            bottom=0.35,
        )

        self.timer = self.figure.canvas.new_timer(
            interval=self.timer_interval_ms
        )

        self.timer.add_callback(self._advance_frame)

        self.information_text = self.figure.text(
            0.01,
            0.98,
            "",
            fontsize=9,
            family="monospace",
            horizontalalignment="left",
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "0.7",
            },
        )

        self.slider_axis = self.figure.add_axes(
            [0.18, 0.14, 0.64, 0.035]
        )

        self.time_slider = Slider(
            ax=self.slider_axis,
            label="Time [days]",
            valmin=float(result.time_days[0]),
            valmax=float(result.time_days[-1]),
            valinit=float(result.time_days[self.current_index]),
            valstep=result.time_days,
        )

        self.time_slider.on_changed(self._on_slider_changed)

        maximum_trail_length = self.number_of_samples

        initial_trail_length = min(
            int(self.trails["rotation_axis"]["length"]),
            maximum_trail_length,
        )

        self.trails["rotation_axis"]["length"] = initial_trail_length

        self.trail_length_slider_axis = self.figure.add_axes(
            [0.30, 0.018, 0.40, 0.025]
        )

        self.trail_length_slider = Slider(
            ax=self.trail_length_slider_axis,
            label="Trail length [samples]",
            valmin=1,
            valmax=maximum_trail_length,
            valinit=initial_trail_length,
            valstep=1,
        )

        self.trail_length_slider.on_changed(
            self._on_trail_length_changed
        )

        self.previous_button_axis = self.figure.add_axes(
            [0.28, 0.075, 0.10, 0.04]
        )

        self.play_button_axis = self.figure.add_axes(
            [0.40, 0.075, 0.20, 0.04]
        )

        self.next_button_axis = self.figure.add_axes(
            [0.62, 0.075, 0.10, 0.04]
        )

        self.previous_button = Button(
            self.previous_button_axis,
            "Previous",
        )

        self.play_button = Button(
            self.play_button_axis,
            "Play",
        )

        self.next_button = Button(
            self.next_button_axis,
            "Next",
        )

        self.previous_button.on_clicked(
            self._on_previous_clicked
        )

        self.play_button.on_clicked(
            self._on_play_pause_clicked
        )

        self.next_button.on_clicked(
            self._on_next_clicked
        )

        self.speed_slider_axis = self.figure.add_axes(
            [0.30, 0.040, 0.40, 0.025]
        )

        self.speed_slider = Slider(
            ax=self.speed_slider_axis,
            label="Playback speed [fps]",
            valmin=1.0,
            valmax=30.0,
            valinit=1000.0 / self.timer_interval_ms,
            valstep=1.0,
        )

        self.speed_slider.on_changed(
            self._on_speed_changed
        )

        self.visibility_axis = self.figure.add_axes(
            [0.76, 0.78, 0.235, 0.19]
        )

        visibility_labels = [
            "Body axes",
            "Figure axis",
            "Rotation axis",
            "Angular momentum",
            "Moon direction",
            "Torque",
            "Rotation-axis trail",
            "Momentum-axis trail",
            "Figure-axis trail",
        ]

        visibility_actives = [
            self.visibility["body_axes"],
            self.visibility["figure_axis"],
            self.visibility["rotation_axis"],
            self.visibility["angular_momentum_axis"],
            self.visibility["moon_direction"],
            self.visibility["torque"],
            self.visibility["rotation_axis_trail"],
            self.visibility["angular_momentum_trail"],
            self.visibility["figure_axis_trail"],
        ]

        number_of_controls = len(visibility_labels)

        self.visibility_checkboxes = CheckButtons(
            ax=self.visibility_axis,
            labels=visibility_labels,
            actives=visibility_actives,
            label_props={
                "fontsize": [12] * number_of_controls,
            },
            frame_props={
                "s": [100] * number_of_controls,
                "linewidth": [1.5] * number_of_controls,
            },
            check_props={
                "s": [100] * number_of_controls,
                "linewidth": [2.0] * number_of_controls,
            },
        )

        self.visibility_axis.set_title(
            "Visible vectors",
            fontsize=11,
        )

        self.visibility_checkboxes.on_clicked(
            self._on_visibility_changed
        )


        
        self.legend = None

        self.draw_state(self.current_index)

    def _initial_figure_axis_inertial(self) -> np.ndarray:
        """Return the initial physical figure axis in inertial coordinates."""
        body_to_inertial = self._body_to_inertial_dcm(0)

        return normalize_vector(
            body_to_inertial
            @ self.result.figure_axis_body[0]
        )

    def _body_to_inertial_dcm(
        self,
        sample_index: int,
    ) -> np.ndarray:
        """Return the Earth body-to-inertial DCM at one sample."""
        return body_to_inertial_euler_dcm(
            psi=float(self.result.psi[sample_index]),
            epsilon=float(self.result.obliquity[sample_index]),
            theta=float(self.result.sidereal_angle[sample_index]),
        )

    def _axis_trail_points(
        self,
        sample_index: int,
        *,
        vector_key: str,
        settings_key: str,
        exaggeration_factor: float,
        display_length: float,
        reference_mode: str = "instantaneous_figure_axis",
    ) -> np.ndarray:
        """Return displayed endpoints for one axis trail."""
        trail_length = int(
            self.trails[settings_key]["length"]
        )

        start_index = max(
            0,
            sample_index - trail_length + 1,
        )

        valid_reference_modes = {
            "initial_inertial_figure_axis",
            "instantaneous_figure_axis",
        }

        if reference_mode not in valid_reference_modes:
            raise ValueError(
                f"Unsupported reference_mode: "
                f"{reference_mode!r}."
            )

        initial_reference_axis = None

        if reference_mode == "initial_inertial_figure_axis":
            initial_reference_axis = (
                self._initial_figure_axis_inertial()
            )

        points = []

        for index in range(start_index, sample_index + 1):
            vectors = self._scene_vectors(index)
            physical_axis = vectors[vector_key]

            if reference_mode == "initial_inertial_figure_axis":
                displayed_axis = exaggerate_direction(
                    direction=physical_axis,
                    reference_axis=initial_reference_axis,
                    factor=exaggeration_factor,
                )

            else:
                figure_axis = vectors["figure_axis"]

                if self.reference_frame == "body":
                    displayed_axis = exaggerate_direction(
                        direction=physical_axis,
                        reference_axis=figure_axis,
                        factor=exaggeration_factor,
                    )
                else:
                    displayed_axis = normalize_vector(
                        physical_axis
                    )

            points.append(
                display_length * displayed_axis
            )

        return np.asarray(points, dtype=float)
    
    def _draw_axis_trail(
        self,
        sample_index: int,
        *,
        vector_key: str,
        settings_key: str,
        exaggeration_factor: float,
        display_length: float,
        label: str,
        reference_mode: str = "instantaneous_figure_axis",
    ) -> None:
        """Draw a fading, variable-width axis trail."""
        settings = self.trails[settings_key]

        if not settings["enabled"]:
            return

        points = self._axis_trail_points(
            sample_index,
            vector_key=vector_key,
            settings_key=settings_key,
            exaggeration_factor=exaggeration_factor,
            display_length=display_length,
            reference_mode=reference_mode,
        )

        if points.shape[0] < 2:
            return

        segments = np.stack(
            [points[:-1], points[1:]],
            axis=1,
        )

        number_of_segments = segments.shape[0]

        fade_fraction = np.linspace(
            0.0,
            1.0,
            number_of_segments,
        )

        alpha_values = (
            settings["minimum_alpha"]
            + (
                settings["maximum_alpha"]
                - settings["minimum_alpha"]
            )
            * fade_fraction**2
        )

        linewidth_values = (
            settings["minimum_linewidth"]
            + (
                settings["maximum_linewidth"]
                - settings["minimum_linewidth"]
            )
            * fade_fraction**2
        )

        base_color = np.array(
            to_rgba(settings["color"])
        )

        segment_colors = np.tile(
            base_color,
            (number_of_segments, 1),
        )
        segment_colors[:, 3] = alpha_values

        collection = Line3DCollection(
            segments,
            colors=segment_colors,
            linewidths=linewidth_values,
            label=label,
        )

        self.axis.add_collection3d(collection)

    def _on_speed_changed(
        self,
        frames_per_second: float,
    ) -> None:
        self.set_playback_speed(frames_per_second)

    def set_playback_interval(
        self,
        interval_ms: int,
    ) -> None:
        """Set the animation timer interval in milliseconds."""
        if interval_ms <= 0:
            raise ValueError(
                "interval_ms must be positive."
            )

        self.timer_interval_ms = interval_ms
        self.timer.interval = interval_ms

    def set_playback_speed(
        self,
        frames_per_second: float,
    ) -> None:
        if frames_per_second <= 0.0:
            raise ValueError(
                "frames_per_second must be positive."
            )

        if self.speed_slider.val != frames_per_second:
            self.speed_slider.set_val(frames_per_second)

        interval_ms = int(round(1000.0 / frames_per_second))
        self.set_playback_interval(interval_ms)

    def _advance_frame(self) -> None:
        """Advance the viewer by one animation step."""
        if not self.is_playing:
            return

        next_index = (
            self.current_index + self.animation_step
        )

        if next_index >= self.number_of_samples:
            next_index = 0

        self.set_index(next_index)


    def _on_visibility_changed(
        self,
        label: str,
    ) -> None:
        """Toggle the visibility of a scene element."""

        key = self.visibility_label_to_key[label]
        self.visibility[key] = not self.visibility[key]

        if key == "rotation_axis_trail":
            self.trails["rotation_axis"]["enabled"] = self.visibility[key]

    
        self.draw_state(self.current_index)

    def _on_play_pause_clicked(self, event) -> None:
        """Toggle automatic playback."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _configure_axis(
        self,
        *,
        elevation: float,
        azimuth: float,
    ) -> None:
        """Configure labels, limits, and camera orientation."""
        set_axes_equal_3d(self.axis)

        frame_suffix = {
            "body": "B",
            "inertial": "I",
        }[self.reference_frame]

        self.axis.set_xlabel(
            rf"$x_{{{frame_suffix}}}$"
        )
        self.axis.set_ylabel(
            rf"$y_{{{frame_suffix}}}$"
        )
        self.axis.set_zlabel(
            rf"$z_{{{frame_suffix}}}$"
        )

        self.axis.view_init(
            elev=elevation,
            azim=azimuth,
        )

    def _on_previous_clicked(self, event) -> None:
        """Move one sample backward."""
        self.pause()

        previous_index = (
            self.current_index - self.animation_step
        ) % self.number_of_samples

        self.set_index(previous_index)

    
    def _on_next_clicked(self, event) -> None:
        """Move one sample forward."""
        self.pause()

        next_index = (
            self.current_index + self.animation_step
        ) % self.number_of_samples

        self.set_index(next_index)

    def play(self) -> None:
        """Start automatic playback."""
        if self.is_playing:
            return

        self.is_playing = True
        self.play_button.label.set_text("Pause")
        self.timer.start()
    
    def pause(self) -> None:
        """Pause automatic playback."""
        if not self.is_playing:
            return

        self.is_playing = False
        self.play_button.label.set_text("Play")
        self.timer.stop()

    def _rotation_axis_label(self) -> str:
        """Return the rotation-axis legend label."""
        label = r"Rotation axis $\hat{\omega}$"

        if (
            self.reference_frame == "body"
            and self.rotation_axis_exaggeration != 1.0
        ):
            label += (
                f" (tilt ×{self.rotation_axis_exaggeration:.0e})"
            )

        return label
    
    def _draw_axis_marker(
        self,
        axis_display: np.ndarray,
        *,
        settings_key: str,
        display_length: float,
    ) -> None:
        """Draw a marker at a displayed axis endpoint."""
        settings = self.trails[settings_key]

        if not settings["show_marker"]:
            return

        direction = normalize_vector(axis_display)

        if np.allclose(direction, 0.0):
            return

        endpoint = display_length * direction

        marker_size = float(settings["marker_size"])

        self.axis.scatter(
            float(endpoint[0]),
            float(endpoint[1]),
            float(endpoint[2]),
            s=marker_size,
            color=settings["color"],
            edgecolors="black",
            linewidths=0.6,
            depthshade=True,
            label="_nolegend_",
            zorder=10,
        )

    def _scene_vectors(
        self,
        sample_index: int,
    ) -> dict[str, np.ndarray]:
        """Return scene vectors in the selected reference frame."""
        if not 0 <= sample_index < self.number_of_samples:
            raise IndexError(
                f"sample_index {sample_index} is outside the valid range."
            )

        if self.reference_frame == "body":
            return {
                "body_axes": np.eye(3),
                "figure_axis": (
                    self.result.figure_axis_body[sample_index]
                ),
                "rotation_axis": (
                    self.result.rotation_axis_body[sample_index]
                ),
                "angular_momentum_axis": (
                    self.result.angular_momentum_axis_body[
                        sample_index
                    ]
                ),
                "moon_direction": (
                    self.result.moon_position_body[sample_index]
                ),
                "torque": (
                    self.result.normalized_lunar_torque[
                        sample_index
                    ]
                ),
            }

        if self.reference_frame == "inertial":
            body_to_inertial = self._body_to_inertial_dcm(
                sample_index
            )

            return {
                "body_axes": body_to_inertial,
                "figure_axis": (
                    body_to_inertial
                    @ self.result.figure_axis_body[sample_index]
                ),
                "rotation_axis": (
                    body_to_inertial
                    @ self.result.rotation_axis_body[sample_index]
                ),
                "angular_momentum_axis": (
                    body_to_inertial
                    @ self.result.angular_momentum_axis_body[
                        sample_index
                    ]
                ),
                "moon_direction": (
                    self.result.moon_position_inertial[sample_index]
                ),
                "torque": (
                    body_to_inertial
                    @ self.result.normalized_lunar_torque[
                        sample_index
                    ]
                ),
            }

        raise ValueError(
            f"Unsupported reference frame: "
            f"{self.reference_frame!r}."
        )
    
    
    def draw_state(self, sample_index: int) -> None:
        """Draw one simulation sample, replacing the previous scene."""
        if not 0 <= sample_index < self.number_of_samples:
            raise IndexError(
                f"sample_index {sample_index} is outside the valid range."
            )

        self.current_index = sample_index

        elevation = getattr(self.axis, "elev", self.elevation)
        azimuth = getattr(self.axis, "azim", self.azimuth)

        self.axis.clear()

        self._configure_axis(
            elevation=elevation,
            azimuth=azimuth,
        )

        draw_oblate_earth(self.axis)

        vectors = self._scene_vectors(sample_index)

        if self.visibility["body_axes"]:
            draw_body_axes(
                self.axis,
                directions=vectors["body_axes"],
            )

        figure_axis = vectors["figure_axis"]
        rotation_axis_physical = vectors["rotation_axis"]
        angular_momentum_axis_physical = (vectors["angular_momentum_axis"])
        moon_direction = vectors["moon_direction"]
        torque_direction = vectors["torque"]

        if self.reference_frame == "inertial":
            initial_figure_axis = (
                self._initial_figure_axis_inertial()
            )

            figure_axis_display = exaggerate_direction(
                direction=figure_axis,
                reference_axis=initial_figure_axis,
                factor=self.figure_axis_exaggeration,
            )
        else:
            figure_axis_display = figure_axis

        if self.reference_frame == "body":
            rotation_axis_display = exaggerate_direction(
                direction=rotation_axis_physical,
                reference_axis=figure_axis,
                factor=self.rotation_axis_exaggeration,
            )

            angular_momentum_axis_display = exaggerate_direction(
                direction=angular_momentum_axis_physical,
                reference_axis=figure_axis,
                factor=self.angular_momentum_axis_exaggeration,
            )

        else:
            rotation_axis_display = normalize_vector(
                rotation_axis_physical
            )

            angular_momentum_axis_display = normalize_vector(
                angular_momentum_axis_physical
            )

        time_days = self.result.time_days[sample_index]

        angular_velocity_body = (
            self.result.angular_velocity_body[sample_index]
        )

        angular_momentum_body = (
            self.result.angular_momentum_body[sample_index]
        )

        normalized_torque_body = (
            self.result.normalized_lunar_torque[sample_index]
        )

        if self.reference_frame == "body":
            angular_velocity = angular_velocity_body
            angular_momentum = angular_momentum_body
            normalized_torque = normalized_torque_body
            frame_suffix = "B"

        else:
            body_to_inertial = self._body_to_inertial_dcm(
                sample_index
            )

            angular_velocity = (
                body_to_inertial @ angular_velocity_body
            )

            angular_momentum = (
                body_to_inertial @ angular_momentum_body
            )

            normalized_torque = (
                body_to_inertial @ normalized_torque_body
            )

            frame_suffix = "I"

        angular_momentum_magnitude = np.linalg.norm(
            angular_momentum
        )

        frame_title = {
            "body": "Body-Fixed Frame",
            "inertial": "Inertial Frame",
        }[self.reference_frame]

        self.axis.set_title(
            f"Rigid Earth in the {frame_title}\n"
            f"Time = {time_days:.3f} days"
        )
        


        psi = self.result.psi[sample_index]
        epsilon = self.result.obliquity[sample_index]
        theta = self.result.sidereal_angle[sample_index]

        psi_degrees = np.rad2deg(psi)
        epsilon_degrees = np.rad2deg(epsilon)
        theta_degrees = np.rad2deg(theta)

        epsilon_change_arcsec = np.rad2deg(
            epsilon - self.result.obliquity[0]
        ) * 3600.0

        psi_change_arcsec = np.rad2deg(
            psi - self.result.psi[0]
        ) * 3600.0

        theta_wrapped_degrees = np.rad2deg(theta) % 360.0


        if self.visibility["figure_axis"]:
            figure_axis_label = "Figure axis"

            if (
                self.reference_frame == "inertial"
                and self.figure_axis_exaggeration != 1.0
            ):
                figure_axis_label += (
                    f" (motion ×{self.figure_axis_exaggeration:.0e})"
                )

            draw_vector(
                self.axis,
                figure_axis_display,
                length=1.55,
                color="black",
                label=figure_axis_label,
                linewidth=2.4,
                linestyle="--",
            )
        if self.reference_frame == "inertial":
            self._draw_axis_marker(
                axis_display=figure_axis_display,
                settings_key="figure_axis",
                display_length=1.55,
            )


        if self.visibility["rotation_axis"]:
            draw_vector(
                self.axis,
                rotation_axis_display,
                length=1.65,
                color="magenta",
                label=self._rotation_axis_label(),
                linewidth=2.6,
            )

            self._draw_axis_marker(
                axis_display=rotation_axis_display,
                settings_key="rotation_axis",
                display_length=1.65,
            )

        if self.visibility["angular_momentum_axis"]:
            draw_vector(
                self.axis,
                angular_momentum_axis_display,
                length=1.58,
                color="cyan",
                label=r"Angular-momentum axis $\hat{\mathbf{H}}$",
                linewidth=2.5,
            )

            self._draw_axis_marker(
                axis_display=angular_momentum_axis_display,
                settings_key="angular_momentum_axis",
                display_length=1.58,
            )
        
        if self.visibility["moon_direction"]:
            draw_vector(
                self.axis,
                moon_direction,
                length=1.75,
                color="orange",
                label="Moon direction",
                linewidth=2.0,
            )

        if self.visibility["torque"]:
            draw_vector(
                self.axis,
                torque_direction,
                length=1.25,
                color="purple",
                label="Normalized lunar torque",
                linewidth=2.0,
                normalization_tolerance=1.0e-20,
            )

        if self.visibility["rotation_axis_trail"]:
            self._draw_axis_trail(
                sample_index,
                vector_key="rotation_axis",
                settings_key="rotation_axis",
                exaggeration_factor=self.rotation_axis_exaggeration,
                display_length=1.65,
                label="Rotation-axis trail",
            )

        if self.visibility["angular_momentum_trail"]:
            self._draw_axis_trail(
                sample_index,
                vector_key="angular_momentum_axis",
                settings_key="angular_momentum_axis",
                exaggeration_factor=(
                    self.angular_momentum_axis_exaggeration
                ),
                display_length=1.58,
                label="Momentum-axis trail",
            )

        if (
            self.reference_frame == "inertial"
            and self.visibility["figure_axis_trail"]
            and self.trails["figure_axis"]["enabled"]
        ):
            self._draw_axis_trail(
                sample_index,
                vector_key="figure_axis",
                settings_key="figure_axis",
                exaggeration_factor=self.figure_axis_exaggeration,
                display_length=1.55,
                label="Figure-axis motion trail",
                reference_mode="initial_inertial_figure_axis",
            )

        trail_length = self.trails["rotation_axis"]["length"]

        self.information_text.set_text(
            rf"$\omega_{{{frame_suffix}}}=$ "
            f"[{angular_velocity[0]:.3e}, "
            f"{angular_velocity[1]:.3e}, "
            f"{angular_velocity[2]:.3e}] rad/s\n"
            rf"$\mathbf{{H}}_{{{frame_suffix}}}=$ "
            f"[{angular_momentum[0]:.3e}, "
            f"{angular_momentum[1]:.3e}, "
            f"{angular_momentum[2]:.3e}] kg m$^2$ s$^{{-1}}$\n"
            rf"$|\mathbf{{H}}|=$ "
            f"{angular_momentum_magnitude:.3e} kg m$^2$ s$^{{-1}}$\n"
            rf"$d_{{{frame_suffix}}}=$ "
            f"[{normalized_torque[0]:.3e}, "
            f"{normalized_torque[1]:.3e}, "
            f"{normalized_torque[2]:.3e}] s$^{{-2}}$\n"
            f"Trail length = {trail_length} samples\n"
            rf"$\psi={psi_degrees:.6f}^\circ$" f"\n"
            rf"$\epsilon={epsilon_degrees:.6f}^\circ$" f"\n"
            rf"$\theta={theta_wrapped_degrees:.3f}^\circ$, $\theta_{{tot}}={theta_degrees:.3f}^\circ$" f"\n"
            rf"$\Delta\psi={psi_change_arcsec:.3f}$ arcsec" f"\n"
            rf"$\Delta\epsilon={epsilon_change_arcsec:.3f}$ arcsec" f"\n"

        )

        if self.legend is not None:
            self.legend.remove()
            self.legend = None

        handles, labels = self.axis.get_legend_handles_labels()

        if handles:
            self.legend = self.figure.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.001, 0.765),
                fontsize=9,
            )



        self.figure.canvas.draw_idle()

    def set_visibility(
        self,
        element: str,
        visible: bool,
    ) -> None:
        """Set the visibility of one scene element."""
        if element not in self.visibility:
            raise ValueError(
                f"Unknown visibility element: {element!r}."
            )

        visible = bool(visible)

        if self.visibility[element] == visible:
            return

        checkbox_index = list(self.visibility).index(element)
        self.visibility_checkboxes.set_active(checkbox_index)


    def _on_slider_changed(self, value: float) -> None:
        """Update the displayed state when the slider moves."""
        sample_index = int(
            np.argmin(
                np.abs(self.result.time_days - value)
            )
        )

        if sample_index != self.current_index:
            self.draw_state(sample_index)

    def _on_trail_length_changed(
        self,
        value: float,
    ) -> None:
        """Update the rotation-axis trail length."""
        self.trails["rotation_axis"]["length"] = int(round(value))
        self.draw_state(self.current_index)

    def set_index(self, sample_index: int) -> None:
        """Display a specific stored output sample."""
        if not -self.number_of_samples <= sample_index < self.number_of_samples:
            raise IndexError(
                f"sample_index {sample_index} is outside the valid range."
            )

        sample_index %= self.number_of_samples

        self.time_slider.set_val(
            float(self.result.time_days[sample_index])
        )

    def set_time_days(self, time_days: float) -> None:
        """Display the sample nearest to a requested time."""
        clipped_time = float(
            np.clip(
                time_days,
                self.result.time_days[0],
                self.result.time_days[-1],
            )
        )

        self.time_slider.set_val(clipped_time)

    def set_rotation_axis_exaggeration(
        self,
        factor: float,
    ) -> None:
        """Set the displayed rotation-axis tilt exaggeration."""
        if factor <= 0.0:
            raise ValueError("factor must be positive.")

        self.rotation_axis_exaggeration = factor
        self.draw_state(self.current_index)

    def show(self) -> None:
        """Open the interactive Matplotlib window."""
        plt.show()

    def save_animation(
        self,
        filename: str | Path,
        *,
        fps: int = 30,
        start_index: int = 0,
        end_index: int | None = None,
        frame_step: int = 1,
        dpi: int = 150,
        close_figure: bool = True,
    ) -> Path:
        """Save the viewer animation as MP4 or GIF.

        Parameters
        ----------
        filename
            Output filename ending in ``.mp4`` or ``.gif``.
        fps
            Number of displayed frames per second.
        start_index
            First stored simulation sample to export.
        end_index
            Stop index, excluded. ``None`` uses all remaining samples.
        frame_step
            Export every nth stored simulation sample.
        dpi
            Output resolution in dots per inch.

        Returns
        -------
        pathlib.Path
            Path of the saved animation.
        """
        
        output_path = Path(filename)

        if not output_path.is_absolute() and output_path.parent == Path("."):
            output_directory = Path("astrodynamics/outputs") / "animations"
            output_directory.mkdir(parents=True, exist_ok=True)
            output_path = output_directory / output_path


        if fps <= 0:
            raise ValueError("fps must be positive.")

        if frame_step <= 0:
            raise ValueError("frame_step must be positive.")

        if not 0 <= start_index < self.number_of_samples:
            raise IndexError(
                "start_index is outside the valid sample range."
            )

        if end_index is None:
            end_index = self.number_of_samples

        if not start_index < end_index <= self.number_of_samples:
            raise IndexError(
                "end_index must be greater than start_index and "
                "within the available sample range."
            )

        suffix = output_path.suffix.lower()

        if suffix not in {".mp4", ".gif"}:
            raise ValueError(
                "filename must end with '.mp4' or '.gif'."
            )

        frame_indices = range(
            start_index,
            end_index,
            frame_step,
        )

        number_of_frames = len(frame_indices)
        duration_seconds = number_of_frames / fps

        
        print(f"Exporting {number_of_frames} frames " f"({duration_seconds:.1f} s at {fps} fps) " f"to {output_path}")

        original_index = self.current_index
        was_playing = self.is_playing

        self.pause()

        def update_frame(sample_index: int):
            self.draw_state(sample_index)
            return ()

        animation = FuncAnimation(
            self.figure,
            update_frame,
            frames=frame_indices,
            interval=1000.0 / fps,
            blit=False,
            repeat=False,
        )

        try:
            if suffix == ".gif":
                writer = PillowWriter(fps=fps)
            else:
                writer = FFMpegWriter(
                    fps=fps,
                    metadata={
                        "title": "Rigid Earth Rotation",
                        "artist": "astrodynamics",
                    },
                )

            animation.save(
                output_path,
                writer=writer,
                dpi=dpi,
            )
            if close_figure:
                plt.close(self.figure)
            

            print(f"Animation saved to:\n{output_path.resolve()}")

        finally:
            self.draw_state(original_index)

            self.time_slider.set_val(
                float(self.result.time_days[original_index])
            )

            if was_playing:
                self.play()

        return output_path
    
        


    def set_rotation_axis_trail_enabled(
        self,
        enabled: bool,
    ) -> None:
        """Show or hide the rotation-axis trail."""
        enabled = bool(enabled)

        if self.visibility["rotation_axis_trail"] == enabled:
            return

        checkbox_index = list(
            self.visibility_label_to_key.values()
        ).index("rotation_axis_trail")

        self.visibility_checkboxes.set_active(
            checkbox_index
        )

    def set_rotation_axis_trail_length(
        self,
        length: int,
    ) -> None:
        """Set the maximum number of samples in the trail."""
        if not isinstance(length, (int, np.integer)):
            raise TypeError("length must be an integer.")

        if length < 1:
            raise ValueError("length must be at least 1.")

        maximum_length = int(self.trail_length_slider.valmax)

        if length > maximum_length:
            raise ValueError(
                f"length cannot exceed the number of available "
                f"samples ({maximum_length})."
            )

        self.trail_length_slider.set_val(length)

    def set_view(
        self,
        *,
        elevation: float,
        azimuth: float,
    ) -> None:
        """Set the current camera orientation."""
        self.elevation = float(elevation)
        self.azimuth = float(azimuth)

        self.axis.view_init(
            elev=self.elevation,
            azim=self.azimuth,
        )

        self.figure.canvas.draw_idle()

    def set_reference_frame(
        self,
        reference_frame: str,
    ) -> None:
        """Set the frame in which the scene is displayed."""
        valid_frames = {
            "body",
            "inertial",
        }

        if reference_frame not in valid_frames:
            raise ValueError(
                f"reference_frame must be one of "
                f"{sorted(valid_frames)}, but received "
                f"{reference_frame!r}."
            )

        if self.reference_frame == reference_frame:
            return

        self.reference_frame = reference_frame
        self.draw_state(self.current_index)

def interactive_rigid_earth_state_3d(
    result: SimulationResult,
    *,
    initial_index: int = 0,
    rotation_axis_exaggeration: float = 1.0e6,
) -> RigidEarthViewer:
    """Create and return an interactive rigid-Earth viewer."""
    return RigidEarthViewer(
        result,
        initial_index=initial_index,
        rotation_axis_exaggeration=rotation_axis_exaggeration,
    )