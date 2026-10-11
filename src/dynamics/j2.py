import numpy as np
from .two_body import MU, R_EARTH

J2 = 1.08262668e-3

def acceleration(position):
    position = np.asarray(position, dtype=float)
    r2 = np.sum(position**2, axis=-1, keepdims=True)
    z_ratio = position[..., 2:3]**2 / r2
    factors = np.concatenate((5*z_ratio-1, 5*z_ratio-1, 5*z_ratio-3), axis=-1)
    return 1.5 * J2 * MU * R_EARTH**2 * position * factors / r2**2.5
