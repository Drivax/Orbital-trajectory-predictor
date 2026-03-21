"""data_loader.py
TLE parsing, SGP4 propagation via skyfield, feature engineering,
sliding-window sequence construction, and train/test splitting.

Pipeline
--------
1. load_tle_file      – parse 2-line / 3-line TLE files
2. propagate_satellite – SGP4 propagation at 5-min cadence
3. build_dataset       – propagate a list of TLEs and concatenate
4. create_windows      – build (X, y) sliding-window arrays
5. split_and_normalize – chronological 80/20 split + Min-Max scaling

Feature vector (21 dimensions per timestep)
--------------------------------------------
 0  inclination         (deg)
 1  RAAN                (deg)
 2  eccentricity        (dimensionless)
 3  argument of perigee (deg)
 4  mean anomaly        (deg)
 5  mean motion         (rev/day)
 6  sin(2π t / T⊕)     (diurnal encoding)
 7  cos(2π t / T⊕)     (diurnal encoding)
 8  altitude            (km, from SGP4 position)
 9  orbital speed       (km/s, vis-viva)
10  BSTAR drag term     (1/Earth-radii, from TLE line 1)
11  argument of lat.    (deg, ω + M mod 360)
12  x                   (km, ECI frame – current position)
13  y                   (km, ECI frame – current position)
14  z                   (km, ECI frame – current position)
15  sin(arg_lat)        (smooth periodic encoding of orbital phase)
16  cos(arg_lat)        (smooth periodic encoding of orbital phase)
17  sin(inclination)    (smooth encoding of orbit plane orientation)
18  vx                  (km/s, finite-difference velocity)
19  vy                  (km/s, finite-difference velocity)
20  vz                  (km/s, finite-difference velocity)
"""

import math
import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skyfield.api import load, EarthSatellite

from src.physics import (
    R_EARTH_KM,
    extract_bstar,
    mean_motion_to_sma,
    orbital_altitude,
    orbital_speed_vis_viva,
)

# ── Constants (match README specification) ─────────────────────────────────
WINDOW_SIZE: int   = 12       # 12 steps × 5 min = 1 hour of history
STEP_MIN:    float = 5.0      # propagation cadence (minutes)
HOURS:       float = 48.0     # total propagation window per satellite
T_EARTH:     float = 86400.0  # sidereal day (seconds) for time encodings
N_FEATURES:  int   = 21       # feature-vector dimension (6 Keplerian + 2 diurnal + 4 physics + 3 ECI + 3 angular + 3 velocity)


# ── TLE utilities ───────────────────────────────────────────────────────────

def _tle_checksum(line: str) -> int:
    """Compute the modulo-10 checksum for a TLE line (positions 1-68)."""
    return (
        sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
        % 10
    )


def _fix_checksum(line: str) -> str:
    """Return *line* with the checksum digit at position 69 recomputed."""
    return line[:68] + str(_tle_checksum(line))


def load_tle_file(filepath: str) -> List[Tuple[str, str, str]]:
    """Parse a TLE file and return a list of ``(name, line1, line2)`` tuples.

    Accepts both **3-line** (name + two data lines) and **2-line** (data
    lines only) formats.  Checksums are automatically recomputed to tolerate
    minor formatting issues in downloaded files.
    """
    with open(filepath, "r") as fh:
        raw = [ln.rstrip() for ln in fh if ln.strip()]

    satellites: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if line.startswith("1 ") and len(line) >= 68:
            # 2-line format (no name preceding)
            if i + 1 < len(raw) and raw[i + 1].startswith("2 "):
                line1 = _fix_checksum(line.ljust(69))
                line2 = _fix_checksum(raw[i + 1].ljust(69))
                name  = f"SAT_{line1[2:7].strip()}"
                satellites.append((name, line1, line2))
                i += 2
            else:
                i += 1
        elif not (line.startswith("1 ") or line.startswith("2 ")):
            # 3-line format (name first)
            if (
                i + 2 < len(raw)
                and raw[i + 1].startswith("1 ")
                and raw[i + 2].startswith("2 ")
            ):
                name  = line.strip()
                line1 = _fix_checksum(raw[i + 1].ljust(69))
                line2 = _fix_checksum(raw[i + 2].ljust(69))
                satellites.append((name, line1, line2))
                i += 3
            else:
                i += 1
        else:
            i += 1

    return satellites


# ── Feature extraction ──────────────────────────────────────────────────────

def _extract_keplerian(line1: str, line2: str) -> Tuple[np.ndarray, float]:
    """Return the six Keplerian elements from TLE lines and the BSTAR term.

    Returns
    -------
    keplerian : ndarray, shape (6,) – [inclination, RAAN, eccentricity,
                                       arg_perigee, mean_anomaly, mean_motion]
    bstar     : float – BSTAR drag coefficient (1/Earth-radii)
    """
    inclination  = float(line2[8:16])
    raan         = float(line2[17:25])
    eccentricity = float("0." + line2[26:33].strip())
    arg_perigee  = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion  = float(line2[52:63])
    keplerian = np.array(
        [inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion],
        dtype=np.float64,
    )
    bstar = extract_bstar(line1)
    return keplerian, bstar


# ── SGP4 propagation ────────────────────────────────────────────────────────

def propagate_satellite(
    name: str,
    line1: str,
    line2: str,
    hours: float = HOURS,
    step_min: float = STEP_MIN,
) -> List[dict]:
    """Propagate a satellite with SGP4 and return a list of feature dicts.

    Each dict contains:
        ``features``  – ``np.ndarray`` shape ``(12,)``:
                        6 Keplerian elements + 2 sinusoidal time encodings
                        + altitude (km) + orbital speed (km/s)
                        + BSTAR drag term + argument of latitude (deg)
        ``x``, ``y``, ``z``  – float (km, ECI frame)

    The 12-dimensional feature vector is::

        e_k = [i, Ω, e, ω, M, n,
               sin(2π t_k / T⊕),
               cos(2π t_k / T⊕),
               altitude_k (km),
               speed_k (km/s),
               B* (1/Earth-radii),
               arg_latitude_k (deg)]

    where ``T⊕ = 86 400 s`` (sidereal day).

    The four new features (8–11) add physical context that helps the model
    distinguish altitude-dependent drag and the orbital phase angle.
    """
    ts        = load.timescale()
    satellite = EarthSatellite(line1, line2, name, ts)
    keplerian, bstar = _extract_keplerian(line1, line2)

    # Semi-major axis (km) – constant for a given TLE epoch
    mean_motion  = float(keplerian[5])
    a_km         = mean_motion_to_sma(mean_motion)
    arg_perigee  = float(keplerian[3])

    steps    = int(hours * 60 / step_min)
    epoch_tt = satellite.epoch.tt  # Julian date (Terrestrial Time)

    records: List[dict] = []
    for k in range(steps):
        t = ts.tt_jd(epoch_tt + k * step_min / 1440.0)
        try:
            pos = satellite.at(t).position.km  # shape (3,)
        except Exception:
            continue

        t_sec   = k * step_min * 60.0
        sin_enc = math.sin(2.0 * math.pi * t_sec / T_EARTH)
        cos_enc = math.cos(2.0 * math.pi * t_sec / T_EARTH)

        # Physics-derived features
        r_km    = float(np.linalg.norm(pos))
        alt_km  = r_km - R_EARTH_KM
        speed   = orbital_speed_vis_viva(r_km, a_km)
        # Argument of latitude ≈ ω + M (mod 360) for near-circular orbits
        mean_anom_k = float(keplerian[4]) + mean_motion * (k * step_min / 1440.0) * 360.0
        arg_lat     = (arg_perigee + mean_anom_k) % 360.0

        # Sinusoidal encodings for angular elements to avoid 360° discontinuity
        arg_lat_rad = math.radians(arg_lat)
        incl_rad    = math.radians(float(keplerian[0]))

        features = np.array(
            [
                keplerian[0],         # inclination (deg)
                keplerian[1],         # RAAN (deg)
                keplerian[2],         # eccentricity
                keplerian[3],         # argument of perigee (deg)
                keplerian[4],         # mean anomaly at epoch (deg)
                keplerian[5],         # mean motion (rev/day)
                sin_enc,              # diurnal sin(2π t/T⊕)
                cos_enc,              # diurnal cos(2π t/T⊕)
                alt_km,               # altitude (km)
                speed,                # orbital speed (km/s)
                bstar,                # BSTAR drag term
                arg_lat,              # argument of latitude (deg)
                float(pos[0]),        # x (km, ECI)
                float(pos[1]),        # y (km, ECI)
                float(pos[2]),        # z (km, ECI)
                math.sin(arg_lat_rad),# sin(arg_lat) – smooth phase encoding
                math.cos(arg_lat_rad),# cos(arg_lat) – smooth phase encoding
                math.sin(incl_rad),   # sin(incl) – orbit plane orientation
            ],
            dtype=np.float64,
        )
        records.append(
            {
                "features": features,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }
        )

    return records


def build_dataset(
    tle_list: List[Tuple[str, str, str]],
    hours: float = HOURS,
    step_min: float = STEP_MIN,
) -> List[List[dict]]:
    """Propagate every satellite in *tle_list* and return per-satellite record lists.

    Each inner list contains the time-series for one satellite.  Keeping
    satellites separate prevents :func:`create_windows` from generating
    windows that span satellite boundaries (which would be physically
    meaningless — mixing two unrelated orbit states).

    Parameters
    ----------
    tle_list : list of (name, line1, line2)

    Returns
    -------
    list of record lists, one per satellite
    """
    all_satellite_records: List[List[dict]] = []
    dt_sec = step_min * 60.0
    for name, l1, l2 in tle_list:
        try:
            recs = propagate_satellite(name, l1, l2, hours=hours, step_min=step_min)
            if recs:
                # Append finite-difference velocity (vx, vy, vz) in km/s
                positions = np.array(
                    [[r["x"], r["y"], r["z"]] for r in recs], dtype=np.float64
                )
                vel = np.diff(positions, axis=0) / dt_sec  # (N-1, 3)
                vel = np.vstack([vel[0:1], vel])            # backfill t=0
                for i, rec in enumerate(recs):
                    rec["features"] = np.concatenate(
                        [rec["features"], vel[i]]
                    )
                all_satellite_records.append(recs)
        except Exception as exc:
            print(f"[data_loader] Skipping {name}: {exc}")
    return all_satellite_records


# ── Window construction ─────────────────────────────────────────────────────

def create_windows(
    satellite_records: List[List[dict]],
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window input/target arrays **per satellite**.

    Windows are created independently for each satellite to avoid cross-
    satellite boundary artefacts.  For each satellite with records
    ``[r_0, …, r_{N-1}]`` and each index ``k`` in ``[W-1, N-2]``::

        X_k = [e_{k-W+1}, ..., e_k]   ∈ R^{W × N_FEATURES}
        y_k = r_{k+1}                  ∈ R^3

    Parameters
    ----------
    satellite_records : list of per-satellite record lists
                        (output of :func:`build_dataset`)
    window_size       : number of consecutive timesteps per input window

    Returns
    -------
    X : ndarray, shape (M, window_size, N_FEATURES)
    y : ndarray, shape (M, 3)
    """
    # Accept legacy flat list (backward compatibility with code that calls
    # build_dataset and passes the flat concatenated list directly).
    if satellite_records and isinstance(satellite_records[0], dict):
        satellite_records = [satellite_records]  # type: ignore[list-item]

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for records in satellite_records:
        if len(records) <= window_size:
            continue  # not enough data for even one window
        features  = np.array([r["features"] for r in records], dtype=np.float32)
        positions = np.array([[r["x"], r["y"], r["z"]] for r in records], dtype=np.float32)
        for k in range(window_size - 1, len(records) - 1):
            X_list.append(features[k - window_size + 1 : k + 1])  # (W, N_FEATURES)
            # Target: position DELTA (next - current) rather than absolute position.
            # Predicting the step-wise displacement is far easier to learn:
            # the 5-min displacement (~2 250 km) is much smaller and more
            # consistent across orbit types than the absolute ECI position
            # (which spans ±7 000 km depending on orbital geometry).
            y_list.append(positions[k + 1] - positions[k])        # (3,) delta

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


# ── Normalisation & splitting ───────────────────────────────────────────────

def split_and_normalize(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    satellite_lengths: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Per-satellite chronological train/test split + Min-Max normalisation.

    When *satellite_lengths* is supplied each satellite's window count is
    known, so the split is performed **independently per satellite**: the
    first ``train_ratio`` fraction of each satellite's windows go to the
    training set and the remainder to the test set.  This ensures:

    * **No temporal data leakage** – test windows are always in the future
      relative to training windows for the same satellite.
    * **Representative test set** – every orbit type present in the dataset
      appears in both train and test splits (no distribution shift).

    When *satellite_lengths* is omitted the function falls back to a single
    global chronological split (legacy behaviour, less accurate for diverse
    datasets).

    The Min-Max scalers are fitted **only on the training set**.

    Parameters
    ----------
    X                  : float32 ndarray, shape (N, W, F)
    y                  : float32 ndarray, shape (N, 3)
    train_ratio        : fraction of each satellite's windows used for training
    satellite_lengths  : list of per-satellite window counts
                         (output of :func:`compute_satellite_window_lengths`)

    Returns
    -------
    X_train, y_train, X_test, y_test : float32 ndarrays
    x_scaler, y_scaler               : fitted :class:`MinMaxScaler` objects
    """
    if satellite_lengths is not None:
        # Per-satellite split
        train_idx: List[int] = []
        test_idx:  List[int] = []
        offset = 0
        for length in satellite_lengths:
            split = max(1, int(length * train_ratio))
            train_idx.extend(range(offset, offset + split))
            test_idx.extend(range(offset + split, offset + length))
            offset += length
        X_train = X[train_idx]
        X_test  = X[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]
    else:
        # Legacy global split
        split   = int(len(X) * train_ratio)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

    # Flatten windows for scaler fitting
    N_tr, W, F  = X_train.shape
    X_tr_flat   = X_train.reshape(-1, F)
    X_te_flat   = X_test.reshape(-1, F)

    x_scaler = MinMaxScaler()
    x_scaler.fit(X_tr_flat)
    X_train_n = x_scaler.transform(X_tr_flat).reshape(N_tr, W, F).astype(np.float32)
    X_test_n  = x_scaler.transform(X_te_flat).reshape(X_test.shape[0], W, F).astype(np.float32)

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_n = y_scaler.transform(y_train).astype(np.float32)
    y_test_n  = y_scaler.transform(y_test).astype(np.float32)

    return X_train_n, y_train_n, X_test_n, y_test_n, x_scaler, y_scaler


def compute_satellite_window_lengths(
    satellite_records: List[List[dict]],
    window_size: int = WINDOW_SIZE,
) -> List[int]:
    """Return the number of windows each satellite contributes.

    Parameters
    ----------
    satellite_records : per-satellite record lists (output of :func:`build_dataset`)
    window_size       : sliding-window size

    Returns
    -------
    list of int – window counts, one per satellite (0 if too short)
    """
    lengths = []
    for records in satellite_records:
        n_windows = max(0, len(records) - window_size)
        lengths.append(n_windows)
    return lengths


# ── Optional: download live TLEs from Celestrak ────────────────────────────

def download_tle(
    url: str = "https://celestrak.org/SOCRATES/query.php?CODE=starlink&FORMAT=TLE",
    save_path: str = "data/downloaded_starlink_tle.txt",
) -> str:
    """Download a TLE file from *url* and save it to *save_path*.

    Returns the path to the saved file, or raises on network error.
    Requires the ``requests`` library.
    """
    import requests  # optional dependency

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, "w") as fh:
        fh.write(response.text)
    print(f"[data_loader] TLEs downloaded → {save_path}")
    return save_path
