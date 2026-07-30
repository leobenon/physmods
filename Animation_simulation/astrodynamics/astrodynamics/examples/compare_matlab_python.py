"""Compare MATLAB and Python rigid-Earth simulation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = (
    "omega_1",
    "omega_2",
    "omega_3",
    "psi",
    "epsilon",
    "theta",
)


def load_result(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load time and six-state history from a CSV file."""
    data = np.loadtxt(
        filename,
        delimiter=",",
        skiprows=1,
    )

    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            f"{filename} must contain seven columns: "
            "time and six state components."
        )

    return data[:, 0], data[:, 1:]


def interpolate_state(
    source_time: np.ndarray,
    source_state: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Interpolate every state component onto target_time."""
    interpolated = np.empty(
        (target_time.size, source_state.shape[1]),
        dtype=float,
    )

    for index in range(source_state.shape[1]):
        interpolated[:, index] = np.interp(
            target_time,
            source_time,
            source_state[:, index],
        )

    return interpolated


def main() -> None:
    python_time, python_state = load_result(
        "/Users/rukan1/Desktop/physmods/Animation_simulation/astrodynamics/astrodynamics/outputs/rigid_earth_python.csv"
    )
    matlab_time, matlab_state = load_result(
        "/Users/rukan1/Desktop/AA/astrodynamik/Astrodynamik 2 /rigid_earth_matlab.csv"
    )

    matlab_on_python_grid = interpolate_state(
        source_time=matlab_time,
        source_state=matlab_state,
        target_time=python_time,
    )

    difference = python_state - matlab_on_python_grid

    print("Maximum absolute differences:")

    for index, name in enumerate(STATE_NAMES):
        maximum = np.max(np.abs(difference[:, index]))
        print(f"{name:>8s}: {maximum:.6e}")

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(12, 10),
        sharex=True,
    )

    for index, axis in enumerate(axes.flat):
        axis.plot(
            python_time / 86400.0,
            difference[:, index],
        )
        axis.set_title(STATE_NAMES[index])
        axis.set_ylabel("Python - MATLAB")
        axis.grid(True)

    axes[-1, 0].set_xlabel("Time [days]")
    axes[-1, 1].set_xlabel("Time [days]")

    figure.suptitle(
        "Rigid-Earth Model: Python–MATLAB Difference"
    )
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()