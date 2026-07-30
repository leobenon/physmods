"""Display the rigid-Earth model in an interactive 3D viewer."""

from astrodynamics.simulation import simulate_rigid_earth
from astrodynamics.visualization.animation3d import (
    RigidEarthViewer,
)


def main() -> None:
    result = simulate_rigid_earth()

    if not result.success:
        raise RuntimeError(result.message)

    viewer = RigidEarthViewer(
        result,
        initial_index=0,
        rotation_axis_exaggeration=1.0e6,
    )
    viewer.set_view(elevation=25.0,azimuth=50.0)
    viewer.set_playback_speed(15.0)
    viewer.set_rotation_axis_trail_length(1500)
    viewer.set_rotation_axis_trail_enabled(True)
    #viewer.save_animation("rigid_earth.gif",fps=30, start_index=600 , end_index=1600,frame_step=5,dpi=130,close_figure=True)

    viewer.show()
    


if __name__ == "__main__":
    main()