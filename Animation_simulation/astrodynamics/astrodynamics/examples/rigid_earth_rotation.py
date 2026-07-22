"""Run and visualize the rigid-Earth rotation simulation."""

import matplotlib.pyplot as plt
from astrodynamics.visualization.plots import show_all_plots
from astrodynamics.visualization.animation3d import (
    plot_rigid_earth_state_3d,
)
from astrodynamics.simulation import (
    simulate_rigid_earth,
    save_simulation_csv,
)
from astrodynamics.visualization.plots import (
    plot_rigid_earth_summary,
)


def main() -> None:
    result = simulate_rigid_earth()

    save_simulation_csv(
        result,
        "astrodynamics/output/rigid_earth_python.csv",
    )
    if not result.success:
        raise RuntimeError(
            f"Integration failed: {result.message}"
        )

    print("Rigid-Earth integration completed.")
    print(f"Output samples: {result.time.size}")
    print(
        "Function evaluations: "
        f"{result.function_evaluations}"
    )
    print(
        "Final simulation time: "
        f"{result.time_days[-1]:.2f} days"
    )
    print("Final state:")
    print(result.state[-1])

 

    show_all_plots(result)



if __name__ == "__main__":
    main()