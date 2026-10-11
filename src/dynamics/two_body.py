import numpy as np

MU = 398600.4418
R_EARTH = 6378.137

def acceleration(position):
    position = np.asarray(position, dtype=float)
    return -MU * position / np.linalg.norm(position, axis=-1, keepdims=True)**3
