"""Run and visualize the rigid-Earth rotation simulation."""

import matplotlib.pyplot as plt

from astrodynamics.simulation import simulate_rigid_earth
from astrodynamics.visualization.plots import (
    plot_rigid_earth_summary,
)


def main() -> None:
    result = simulate_rigid_earth()

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

    plot_rigid_earth_summary(result)
    plt.show()


if __name__ == "__main__":
    main()