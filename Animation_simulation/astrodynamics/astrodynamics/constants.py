# astrodynamics/constants.py

from __future__ import annotations

import numpy as np

# Time
SECONDS_PER_MINUTE: float = 60.0
SECONDS_PER_HOUR: float = 3600.0
SECONDS_PER_DAY: float = 86_400.0

# Angle conversion
DEG_TO_RAD: float = np.pi / 180.0
RAD_TO_DEG: float = 180.0 / np.pi
RAD_TO_ARCSEC: float = 206_264.80624709636
ARCSEC_TO_RAD: float = 1.0 / RAD_TO_ARCSEC

# Universal gravitation
GRAVITATIONAL_CONSTANT: float = 6.67430e-11  # m^3 kg^-1 s^-2