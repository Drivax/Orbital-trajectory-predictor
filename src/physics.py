"""physics.py
Orbital mechanics utilities: coordinate transforms, perturbation models,
satellite-specific derived quantities, and conjunction analysis.

Physical Constants
------------------
R_EARTH_KM   = 6378.137 km       WGS84 equatorial radius
MU_KM3_S2    = 398600.4418 km³/s² Earth standard gravitational parameter
J2           = 1.08262668 × 10⁻³ Earth oblateness (C20 coefficient)
OMEGA_EARTH  = 7.2921150 × 10⁻⁵ rad/s  Earth rotation rate
F_SRP        = 4.56 × 10⁻⁶ N/m²  Solar radiation pressure at 1 AU
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
R_EARTH_KM:  float = 6378.137        # WGS84 equatorial radius (km)
MU_KM3_S2:   float = 398600.4418     # Earth gravitational parameter (km³/s²)
J2:          float = 1.08262668e-3   # Earth J2 oblateness coefficient
J3:          float = -2.53215306e-6  # Earth J3 coefficient
OMEGA_EARTH: float = 7.2921150e-5    # Earth rotation rate (rad/s)
FLAT:        float = 1.0 / 298.257223563  # WGS84 flattening
E2_WGS84:    float = 2 * FLAT - FLAT**2   # WGS84 first eccentricity squared
ATM_SCALE_HEIGHT_KM: float = 8.5     # Approximate atmospheric scale height (km)
RHO0_KG_M3:  float = 1.225          # Sea-level air density (kg/m³)


# ── TLE field parsing ────────────────────────────────────────────────────────

def extract_bstar(line1: str) -> float:
    """Parse the BSTAR drag term from TLE line 1.

    BSTAR occupies columns 54–61 (1-based) of line 1 and is stored in the
    *assumed decimal point* notation ``±NNNNN±E``, equivalent to
    ``±0.NNNNN × 10^±E`` (unit: 1/Earth-radii).

    Returns
    -------
    float – BSTAR value in 1/Earth-radii  (0.0 on parse failure)

    Examples
    --------
    >>> extract_bstar("1 44713U 19074A   25091.50000000  .00001234  00000-0  98765-4 0  9991")
    9.8765e-8
    """
    try:
        raw = line1[53:61].strip()
        if not raw or raw in ("00000-0", "00000+0", "+00000-0", "-00000-0"):
            return 0.0
        # Sign of mantissa
        if raw[0] in ("+", "-"):
            sign_m = -1.0 if raw[0] == "-" else 1.0
            raw = raw[1:]
        else:
            sign_m = 1.0
        # Last two chars: sign + single exponent digit
        exp_sign = -1 if raw[-2] == "-" else 1
        exp = int(raw[-1])
        mantissa = float("0." + raw[:-2].lstrip("0") or "0")
        return sign_m * mantissa * 10 ** (exp_sign * exp)
    except (ValueError, IndexError):
        return 0.0


# ── Semi-major axis & orbital period ────────────────────────────────────────

def mean_motion_to_sma(n_rev_per_day: float) -> float:
    """Convert mean motion (rev/day) to semi-major axis (km) via Kepler III.

    .. math::
        a = \\left(\\frac{\\mu}{n^2}\\right)^{1/3}, \\quad
        n = \\frac{2\\pi}{T}

    Parameters
    ----------
    n_rev_per_day : mean motion in revolutions per day

    Returns
    -------
    float – semi-major axis in km
    """
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    return (MU_KM3_S2 / n_rad_s**2) ** (1.0 / 3.0)


def orbital_period(a_km: float) -> float:
    """Orbital period in seconds from semi-major axis via Kepler's third law.

    .. math::
        T = 2\\pi \\sqrt{\\frac{a^3}{\\mu}}
    """
    return 2.0 * math.pi * math.sqrt(a_km**3 / MU_KM3_S2)


# ── Orbital state derived quantities ────────────────────────────────────────

def orbital_altitude(r_km: np.ndarray) -> float:
    """Altitude above the WGS84 equatorial sphere (km).

    .. math::
        h = \\|\\mathbf{r}\\| - R_{\\oplus}
    """
    return float(np.linalg.norm(r_km)) - R_EARTH_KM


def orbital_speed_vis_viva(r_km: float, a_km: float) -> float:
    """Orbital speed from the vis-viva equation (km/s).

    .. math::
        v = \\sqrt{\\mu\\left(\\frac{2}{r} - \\frac{1}{a}\\right)}

    Parameters
    ----------
    r_km : current geocentric distance (km)
    a_km : semi-major axis (km)
    """
    return math.sqrt(MU_KM3_S2 * (2.0 / r_km - 1.0 / a_km))


def specific_orbital_energy(r_km: float, v_km_s: float) -> float:
    """Specific orbital energy (km²/s²).

    .. math::
        \\varepsilon = \\frac{v^2}{2} - \\frac{\\mu}{r}

    Negative for bound (elliptic) orbits.
    """
    return 0.5 * v_km_s**2 - MU_KM3_S2 / r_km


# ── J2 perturbation ─────────────────────────────────────────────────────────

def j2_acceleration(r_km: np.ndarray) -> np.ndarray:
    """Perturbing acceleration due to Earth J2 oblateness (km/s²).

    The J2 term arises from the non-spherical mass distribution of Earth and
    is the dominant perturbation for LEO satellites.

    .. math::
        \\mathbf{a}_{J_2} = -\\frac{3\\mu J_2 R_\\oplus^2}{2 r^5}
        \\begin{pmatrix}
            x\\left(1 - 5\\frac{z^2}{r^2}\\right) \\\\
            y\\left(1 - 5\\frac{z^2}{r^2}\\right) \\\\
            z\\left(3 - 5\\frac{z^2}{r^2}\\right)
        \\end{pmatrix}

    Parameters
    ----------
    r_km : ndarray, shape (3,) – ECI position in km

    Returns
    -------
    ndarray, shape (3,) – J2 acceleration in km/s²
    """
    x, y, z = r_km
    r2 = float(x**2 + y**2 + z**2)
    r  = math.sqrt(r2)
    r5 = r2**2 * r
    factor = -1.5 * MU_KM3_S2 * J2 * R_EARTH_KM**2 / r5
    z2_r2  = z**2 / r2
    return np.array([
        factor * x * (1.0 - 5.0 * z2_r2),
        factor * y * (1.0 - 5.0 * z2_r2),
        factor * z * (3.0 - 5.0 * z2_r2),
    ], dtype=np.float64)


def j2_nodal_precession_rate(
    n_rev_per_day: float,
    e: float,
    i_deg: float,
) -> float:
    """RAAN secular precession rate (rad/s) due to Earth J2 oblateness.

    .. math::
        \\dot{\\Omega} = -\\frac{3}{2}\\,
        \\frac{n\\,J_2\\,R_\\oplus^2}{a^2(1-e^2)^2}\\cos i

    Parameters
    ----------
    n_rev_per_day : mean motion (rev/day)
    e             : orbital eccentricity (dimensionless)
    i_deg         : orbital inclination (degrees)

    Returns
    -------
    float – RAAN precession rate in rad/s  (negative for prograde orbits)
    """
    a_km    = mean_motion_to_sma(n_rev_per_day)
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    i_rad   = math.radians(i_deg)
    return (
        -1.5
        * n_rad_s
        * J2
        * (R_EARTH_KM / a_km) ** 2
        * math.cos(i_rad)
        / (1.0 - e**2) ** 2
    )


def j2_apsidal_precession_rate(
    n_rev_per_day: float,
    e: float,
    i_deg: float,
) -> float:
    """Argument-of-perigee secular precession rate (rad/s) due to J2.

    .. math::
        \\dot{\\omega} = \\frac{3}{4}\\,
        \\frac{n\\,J_2\\,R_\\oplus^2}{a^2(1-e^2)^2}
        (5\\cos^2 i - 1)
    """
    a_km    = mean_motion_to_sma(n_rev_per_day)
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    i_rad   = math.radians(i_deg)
    return (
        0.75
        * n_rad_s
        * J2
        * (R_EARTH_KM / a_km) ** 2
        * (5.0 * math.cos(i_rad) ** 2 - 1.0)
        / (1.0 - e**2) ** 2
    )


# ── Atmospheric drag ─────────────────────────────────────────────────────────

def atmospheric_density_exp(alt_km: float) -> float:
    """Simple exponential atmospheric density model (kg/m³).

    .. math::
        \\rho(h) = \\rho_0 \\exp\\!\\left(-\\frac{h}{H}\\right)

    where :math:`H = 8.5` km is the approximate scale height and
    :math:`\\rho_0 = 1.225` kg/m³ is sea-level density.

    Only valid for altitudes below ~1000 km; returns effectively zero above
    that range.

    Parameters
    ----------
    alt_km : geodetic altitude in km

    Returns
    -------
    float – density in kg/m³
    """
    if alt_km < 0.0:
        return RHO0_KG_M3
    return RHO0_KG_M3 * math.exp(-alt_km / ATM_SCALE_HEIGHT_KM)


def drag_deceleration(
    bstar: float,
    v_km_s: float,
    alt_km: float,
) -> float:
    """Approximate along-track deceleration magnitude due to atmospheric drag (km/s²).

    The TLE BSTAR term encapsulates the ballistic coefficient::

        B* = (C_D A) / (2 m)  in units of 1/Earth-radii

    The drag acceleration is approximated as::

        |a_drag| ≈ B* × ρ(h) × v²

    where ρ is the exponential atmospheric density in consistent units.

    Parameters
    ----------
    bstar  : BSTAR drag coefficient (1/Earth-radii)
    v_km_s : satellite speed (km/s)
    alt_km : geodetic altitude (km)

    Returns
    -------
    float – drag deceleration magnitude in km/s²
    """
    rho = atmospheric_density_exp(alt_km)       # kg/m³
    # Convert ρ to consistent units for BSTAR (1/Earth-radii in km⁻¹)
    bstar_km = bstar / R_EARTH_KM               # 1/km
    return bstar_km * rho * 1e9 * v_km_s**2    # km/s²


# ── Coordinate transformations ───────────────────────────────────────────────

def eci_to_ecef(r_eci: np.ndarray, t_sec: float) -> np.ndarray:
    """Rotate an ECI position vector to ECEF via Greenwich Sidereal Angle.

    .. math::
        \\mathbf{r}_{\\text{ECEF}} = R_z(\\theta_G)\\,\\mathbf{r}_{\\text{ECI}},
        \\quad \\theta_G = \\Omega_\\oplus \\cdot t

    Parameters
    ----------
    r_eci  : ndarray, shape (3,) – ECI position in km
    t_sec  : elapsed seconds since epoch

    Returns
    -------
    ndarray, shape (3,) – ECEF position in km
    """
    theta = OMEGA_EARTH * t_sec
    c, s  = math.cos(theta), math.sin(theta)
    R_z   = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ r_eci


def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """Convert ECEF position to geodetic (latitude, longitude, altitude).

    Uses the iterative Bowring method for sub-centimetre accuracy.

    Parameters
    ----------
    r_ecef : ndarray, shape (3,) – ECEF position in km

    Returns
    -------
    lat_deg : geodetic latitude  (degrees, +N)
    lon_deg : longitude          (degrees, +E)
    alt_km  : altitude above WGS84 ellipsoid (km)
    """
    x, y, z   = float(r_ecef[0]), float(r_ecef[1]), float(r_ecef[2])
    lon_rad   = math.atan2(y, x)
    p         = math.sqrt(x**2 + y**2)
    b_km      = R_EARTH_KM * (1.0 - FLAT)       # semi-minor axis
    ep2       = (R_EARTH_KM**2 - b_km**2) / b_km**2  # second eccentricity sq.

    # Initial latitude estimate (Bowring)
    theta = math.atan2(z * R_EARTH_KM, p * b_km)
    lat   = math.atan2(
        z + ep2 * b_km * math.sin(theta) ** 3,
        p - E2_WGS84 * R_EARTH_KM * math.cos(theta) ** 3,
    )

    for _ in range(10):
        N        = R_EARTH_KM / math.sqrt(1.0 - E2_WGS84 * math.sin(lat) ** 2)
        lat_new  = math.atan2(z + E2_WGS84 * N * math.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    N   = R_EARTH_KM / math.sqrt(1.0 - E2_WGS84 * math.sin(lat) ** 2)
    cos_lat = math.cos(lat)
    if abs(cos_lat) > 1e-10:
        alt_km = p / cos_lat - N
    else:
        alt_km = abs(z) / abs(math.sin(lat)) - N * (1.0 - E2_WGS84)

    return math.degrees(lat), math.degrees(lon_rad), alt_km


def eci_to_geodetic(
    r_eci: np.ndarray,
    t_sec: float,
) -> Tuple[float, float, float]:
    """Convert ECI position to geodetic coordinates via ECEF.

    Parameters
    ----------
    r_eci  : ndarray, shape (3,) – ECI position in km
    t_sec  : elapsed seconds since satellite epoch

    Returns
    -------
    lat_deg, lon_deg, alt_km : geodetic coordinates
    """
    r_ecef = eci_to_ecef(r_eci, t_sec)
    return ecef_to_geodetic(r_ecef)


# ── Conjunction analysis ─────────────────────────────────────────────────────

def conjunction_distance(r1_km: np.ndarray, r2_km: np.ndarray) -> float:
    """Euclidean distance between two satellites (km).

    .. math::
        d = \\|\\mathbf{r}_1 - \\mathbf{r}_2\\|_2
    """
    return float(np.linalg.norm(np.asarray(r1_km) - np.asarray(r2_km)))


def is_conjunction(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    threshold_km: float = 5.0,
) -> bool:
    """Return True if the two satellites are within *threshold_km* of each other.

    The default 5 km threshold is a commonly used proximity alert distance for
    LEO conjunction screening (e.g. used by 18th Space Control Squadron).
    """
    return conjunction_distance(r1_km, r2_km) < threshold_km


# ── Orbital energy and angular momentum ─────────────────────────────────────

def specific_angular_momentum(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
) -> np.ndarray:
    """Specific angular momentum vector **h** = **r** × **v** (km²/s).

    .. math::
        \\mathbf{h} = \\mathbf{r} \\times \\mathbf{v}
    """
    return np.cross(r_km, v_km_s)


def eccentricity_vector(
    r_km: np.ndarray,
    v_km_s: np.ndarray,
) -> np.ndarray:
    """Laplace–Runge–Lenz eccentricity vector (dimensionless).

    .. math::
        \\mathbf{e} = \\frac{\\mathbf{v} \\times \\mathbf{h}}{\\mu}
                    - \\frac{\\mathbf{r}}{r}
    """
    h   = specific_angular_momentum(r_km, v_km_s)
    r   = np.linalg.norm(r_km)
    return np.cross(v_km_s, h) / MU_KM3_S2 - r_km / r
