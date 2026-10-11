import numpy as np
from scipy.integrate import solve_ivp
from .two_body import acceleration as gravity, R_EARTH
from .j2 import acceleration as oblateness

OMEGA = 7.292115e-5
DRAG_AREA_MASS = 0.015  # Cd*A/m in m^2/kg; fixed across this controlled population.

def drag_acceleration(position, velocity):
    altitude = np.linalg.norm(position) - R_EARTH
    density = 4e-12 * np.exp(-(altitude - 400.0) / 60.0)
    relative_m_s = 1000 * (velocity - np.cross([0, 0, OMEGA], position))
    return -0.5 * DRAG_AREA_MASS * density * np.linalg.norm(relative_m_s) * relative_m_s / 1000

def propagate(state, times, model="reference", rtol=1e-10, atol=1e-12, max_step=90.0):
    """Propagate an inertial state in km, km/s to nonnegative times in seconds."""
    state, times = np.asarray(state, float), np.asarray(times, float)
    if state.shape != (6,) or times.ndim != 1 or not len(times):
        raise ValueError("Expected state (6,) and a nonempty time vector")
    if not np.isfinite(state).all() or not np.isfinite(times).all():
        raise ValueError("Inputs must be finite")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing and nonnegative")
    if model not in ("two_body", "j2", "reference"):
        raise ValueError(f"Unknown dynamics: {model}")
    if times[-1] == 0:
        return state[None].copy()

    def rhs(_, y):
        a = gravity(y[:3])
        if model != "two_body":
            a = a + oblateness(y[:3])
        if model == "reference":
            a = a + drag_acceleration(y[:3], y[3:])
        return np.concatenate((y[3:], a))

    solution = solve_ivp(rhs, (0, times[-1]), state, t_eval=times, method="DOP853",
                         rtol=rtol, atol=atol, max_step=max_step)
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError(solution.message)
    return solution.y.T
