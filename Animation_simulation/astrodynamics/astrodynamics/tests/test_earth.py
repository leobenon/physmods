import numpy as np

from astrodynamics.bodies.earth import EARTH

def test_principal_moments_are_positive() -> None:
    assert np.all(EARTH.principal_moments > 0.0)

def test_principal_moments_match_gamma_coefficients() -> None:
    moment_a, moment_b, moment_c = EARTH.principal_moments

    gamma_1 = (moment_c - moment_b) / moment_a
    gamma_2 = (moment_a - moment_c) / moment_b
    gamma_3 = (moment_b - moment_a) / moment_c

    assert np.isclose(
        gamma_1,
        EARTH.gamma_1,
        rtol=1.0e-12,
    )

    assert np.isclose(
        gamma_2,
        EARTH.gamma_2,
        rtol=1.0e-12,
    )

    assert np.isclose(
        gamma_3,
        EARTH.gamma_3,
        atol=1.0e-15,
    )