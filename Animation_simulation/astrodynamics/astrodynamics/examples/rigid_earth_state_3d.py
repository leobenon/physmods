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

    viewer.set_playback_speed(15.0)
    viewer.set_rotation_axis_trail_length(1000)
    viewer.set_rotation_axis_trail_enabled(True)
    viewer.show()


if __name__ == "__main__":
    main()