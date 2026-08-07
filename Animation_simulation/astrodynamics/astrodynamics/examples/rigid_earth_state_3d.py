"""Display the rigid-Earth model in an interactive 3D viewer."""

from astrodynamics.simulation import simulate_rigid_earth
from astrodynamics.visualization.animation3d import (
    RigidEarthViewer,
)
from dataclasses import replace

from astrodynamics.bodies.moon import MOON
from astrodynamics.bodies.sun import SUN




def main() -> None:
    simulation_settings = {
        "duration_days": 100.0,
        "output_step": 1.0 * 3600.0,
        "max_step": 3600.0,
    }

    def simulation_factory(
        *,
        include_lunar_torque: bool,
        include_solar_torque: bool,
        moon_orbit_model: str,
        sun_orbit_model: str,
    ):
        moon = replace(
            MOON,
            orbit_model=moon_orbit_model,
        )

        sun = replace(
            SUN,
            orbit_model=sun_orbit_model,
        )

        return simulate_rigid_earth(
            **simulation_settings,
            moon=moon,
            sun=sun,
            include_lunar_torque=include_lunar_torque,
            include_solar_torque=include_solar_torque,
        )
    
    moon_orbit_model = "circular"
    sun_orbit_model = "elliptical"

    result = simulation_factory(
        include_lunar_torque=True,
        include_solar_torque=True,
        moon_orbit_model=moon_orbit_model,
        sun_orbit_model=sun_orbit_model,
    )

    if not result.success:
        raise RuntimeError(result.message)


    viewer = RigidEarthViewer(
        result,
        simulation_factory=simulation_factory,
        moon_orbit_model=moon_orbit_model,
        sun_orbit_model=sun_orbit_model,
        initial_index=0,
        rotation_axis_exaggeration=1.0e6,
        angular_momentum_axis_exaggeration=1.0e6,
        figure_axis_exaggeration=1.0e5,
    )

    viewer.set_reference_frame("body")
    viewer.set_view(elevation=25.0,azimuth=50.0)
    viewer.set_playback_speed(15.0)
    viewer.set_rotation_axis_trail_length(1400)
    viewer.set_rotation_axis_trail_enabled(True)
    #viewer.save_animation("rigid_earth.gif",fps=30, start_index=600 , end_index=1600,frame_step=5,dpi=130,close_figure=True)

    viewer.show()
    


if __name__ == "__main__":
    main()