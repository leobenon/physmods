# tests/test_constants.py

import numpy as np

from astrodynamics.bodies.earth import EARTH
from astrodynamics.bodies.moon import MOON
from astrodynamics.constants import SECONDS_PER_DAY


def test_matlab_constants() -> None:
    assert SECONDS_PER_DAY == 86_400.0
    assert np.isclose(EARTH.rotation_rate, 7292115.1467e-11)
    assert np.isclose(MOON.gravitational_parameter, 398.6e12 / 81.3)
    assert np.isclose(MOON.orbital_radius, 3.8e8)
    assert np.isclose(MOON.orbital_angular_rate, 2.661707223e-6)