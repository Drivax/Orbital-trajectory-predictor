import numpy as np
from ..dynamics.two_body import MU, R_EARTH
from ..dynamics.propagator import propagate

def initial_state(rng, ood=False):
    altitude = rng.uniform(800, 1100) if ood else rng.uniform(450, 650)
    inclination = np.deg2rad(rng.uniform(90, 105) if ood else rng.uniform(35, 65))
    eccentricity = rng.uniform(0, 0.015 if ood else 0.008)
    raan, argument, anomaly = rng.uniform(0, 2*np.pi, 3)
    a = R_EARTH + altitude
    p = a * (1 - eccentricity**2)
    radius = p / (1 + eccentricity*np.cos(anomaly))
    r = radius * np.array([np.cos(anomaly), np.sin(anomaly), 0])
    v = np.sqrt(MU/p) * np.array([-np.sin(anomaly), eccentricity+np.cos(anomaly), 0])
    def rz(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    c, s = np.cos(inclination), np.sin(inclination)
    rotation = rz(raan) @ np.array([[1,0,0],[0,c,-s],[0,s,c]]) @ rz(argument)
    return np.r_[rotation @ r, rotation @ v], {
        "semimajor_altitude_km": altitude, "inclination_deg": float(np.rad2deg(inclination)),
        "eccentricity": eccentricity, "raan_rad": raan, "argument_rad": argument,
        "anomaly_rad": anomaly,
    }

def frames(states):
    radial = states[..., :3] / np.linalg.norm(states[..., :3], axis=-1, keepdims=True)
    normal = np.cross(states[..., :3], states[..., 3:])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    transverse = np.cross(normal, radial)
    return np.stack((radial, transverse, normal), axis=-1)

def features(states):
    basis = frames(states)
    radius = np.linalg.norm(states[:, :3], axis=1)
    local_v = np.einsum("nji,nj->ni", basis, states[:, 3:])
    # Axisymmetry removes longitude; these six features use only the current state.
    return np.column_stack((radius, local_v[:, :2], basis[:, 2, :]))

def generate(config, progress=True):
    if config.origin_spacing_s <= max(config.horizons_s):
        raise ValueError("Forecast windows must not overlap")
    rng = np.random.default_rng(config.data_seed)
    horizon = np.asarray(config.horizons_s)
    groups, manifest = {}, []
    for split, count in (("train",config.train_satellites), ("val",config.val_satellites),
                         ("test",config.test_satellites), ("ood",config.ood_satellites)):
        if progress:
            print(f"Simulating {split}: {count} satellites", flush=True)
        rows = {k: [] for k in ("state", "target", "two_body", "j2", "satellite", "origin_s")}
        for satellite in range(count):
            if progress and satellite and satellite % 12 == 0:
                print(f"  {split}: {satellite}/{count} satellites", flush=True)
            state0, elements = initial_state(rng, split == "ood")
            origins = np.arange(config.origins_per_satellite) * config.origin_spacing_s
            times = np.unique(np.r_[origins, (origins[:, None]+horizon).ravel()])
            truth = propagate(state0, times, rtol=config.rtol, atol=config.atol,
                              max_step=config.max_step_s)
            identity = f"{split}-{satellite:03d}"
            manifest.append(dict(satellite=identity, split=split, **elements))
            for origin in origins:
                state = truth[np.searchsorted(times, origin)]
                target = truth[np.searchsorted(times, origin+horizon)]
                rows["state"].append(state)
                rows["target"].append(target)
                for model in ("two_body", "j2"):
                    rows[model].append(propagate(state, horizon, model, config.rtol,
                                                 config.atol, config.max_step_s)[:, :3])
                rows["satellite"].append(identity)
                rows["origin_s"].append(origin)
        groups[split] = {k: np.asarray(v) for k,v in rows.items()}
    return groups, manifest
