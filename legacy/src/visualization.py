"""visualization.py
3-D trajectory plots (Plotly), animated trajectory GIF (matplotlib + imageio),
per-component error plot, RMSE/MAE comparison bar chart, ground track
(lat/lon), altitude profile, and orbital velocity error plots (matplotlib).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in notebooks / server)
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.physics import (
    R_EARTH_KM,
    eci_to_geodetic,
    orbital_altitude,
    mean_motion_to_sma,
    orbital_speed_vis_viva,
)


# ── 3-D trajectory (Plotly) ────────────────────────────────────────────────

def plot_trajectory_3d(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    title:      str = "Orbital Trajectory – SGP4 (true) vs LSTM (predicted)",
    max_points: int = 576,
) -> go.Figure:
    """Return an interactive Plotly 3-D figure comparing *y_true* and *y_pred*.

    Parameters
    ----------
    y_true / y_pred : ndarray, shape (N, 3) – positions in km (ECI frame)
    title           : figure title
    max_points      : cap to avoid rendering too many points

    Returns
    -------
    :class:`plotly.graph_objects.Figure`
    """
    t = y_true[:max_points]
    p = y_pred[:max_points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=t[:, 0], y=t[:, 1], z=t[:, 2],
            mode="lines",
            name="SGP4 (true)",
            line=dict(color="royalblue", width=3),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="lines",
            name="LSTM (predicted)",
            line=dict(color="firebrick", width=3),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
        ),
        legend=dict(x=0.0, y=1.0),
    )
    return fig


# ── Animated GIF ───────────────────────────────────────────────────────────

def create_trajectory_gif(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    output_path: str = "results/trajectory_3d.gif",
    fps:         int = 10,
    max_frames:  int = 288,
) -> str:
    """Save an animated GIF of the growing trajectory over time.

    Each frame adds one more timestep so the viewer sees both paths being
    traced from the epoch forward.  Returns *output_path* on success.

    Requires ``imageio`` (installed via ``requirements.txt``).
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore[no-redef]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n      = len(y_true)
    step   = max(1, n // max_frames)
    frames = []

    for end in range(step, n, step):
        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot(
            y_true[:end, 0], y_true[:end, 1], y_true[:end, 2],
            "b-", linewidth=1.5, label="True (SGP4)",
        )
        ax.plot(
            y_pred[:end, 0], y_pred[:end, 1], y_pred[:end, 2],
            "r--", linewidth=1.5, label="Predicted (LSTM)",
        )
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title(f"t = {end * 5} min")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)
        plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps)
    print(f"[viz] GIF saved → {output_path}")
    return output_path


# ── RMSE / MAE bar chart ────────────────────────────────────────────────────

def plot_rmse_comparison(
    results:     Dict[str, Dict[str, float]],
    output_path: str = "results/rmse_comparison.png",
) -> str:
    """Save a grouped bar chart comparing RMSE and MAE across models.

    Parameters
    ----------
    results     : ``{"ModelName": {"RMSE_km": ..., "MAE_km": ...}, ...}``
    output_path : destination PNG file

    Returns
    -------
    str – *output_path*
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    models  = list(results.keys())
    rmse    = [results[m]["RMSE_km"] for m in models]
    mae     = [results[m].get("MAE_km", 0.0) for m in models]

    x     = np.arange(len(models))
    width = 0.35
    colors_rmse = ["#1565C0", "#BF360C"][:len(models)]
    colors_mae  = ["#42A5F5", "#FF7043"][:len(models)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rmse, width, label="RMSE (km)", color=colors_rmse)
    ax.bar(x + width / 2, mae,  width, label="MAE  (km)", color=colors_mae)

    # Value labels
    for xi, v in zip(x - width / 2, rmse):
        ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + width / 2, mae):
        ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Error (km)")
    ax.set_title("Position Prediction Error – Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] RMSE chart saved → {output_path}")
    return output_path


# ── Per-component error plot ───────────────────────────────────────────────

def plot_error_components(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    output_path: str = "results/error_components.png",
) -> str:
    """Save a three-panel plot showing Δx, Δy, Δz over time.

    Parameters
    ----------
    y_true / y_pred : ndarray, shape (N, 3) – positions in km
    output_path     : destination PNG file

    Returns
    -------
    str – *output_path*
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    diff   = y_true - y_pred
    t      = np.arange(len(diff))
    labels = ["Δx (km)", "Δy (km)", "Δz (km)"]
    colors = ["#1565C0", "#2E7D32", "#B71C1C"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, diff[:, i], color=col, linewidth=0.8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(lbl)
        ax.grid(linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Timestep (5 min each)")
    fig.suptitle("Per-Component Position Error over Time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Error-components chart saved → {output_path}")
    return output_path


# ── Ground track (lat/lon) ─────────────────────────────────────────────────

def plot_ground_track(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    step_min:    float = 5.0,
    output_path: str   = "results/ground_track.png",
) -> str:
    """Save a ground-track (latitude vs longitude) comparison plot.

    Converts ECI positions to geodetic using an Earth-rotation model.
    The ground track sweeps westward between successive orbits due to
    Earth's rotation (Δλ ≈ −360° × T_orbit / 86 400 s per revolution).

    Parameters
    ----------
    y_true / y_pred : ndarray, shape (N, 3) – ECI positions in km
    step_min        : propagation step in minutes (used for t_sec)
    output_path     : destination PNG file

    Returns
    -------
    str – *output_path*
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lats_t, lons_t = [], []
    lats_p, lons_p = [], []

    for k, (r_t, r_p) in enumerate(zip(y_true, y_pred)):
        t_sec = k * step_min * 60.0
        lat, lon, _ = eci_to_geodetic(r_t, t_sec)
        lats_t.append(lat)
        lons_t.append(lon)
        lat, lon, _ = eci_to_geodetic(r_p, t_sec)
        lats_p.append(lat)
        lons_p.append(lon)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lons_t, lats_t, "b-",  linewidth=0.8, label="SGP4 (true)",      alpha=0.85)
    ax.plot(lons_p, lats_p, "r--", linewidth=0.8, label="LSTM (predicted)", alpha=0.85)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Satellite Ground Track – SGP4 vs LSTM Prediction")
    ax.legend(loc="upper right")
    ax.grid(linestyle="--", alpha=0.4)
    # Equator and prime-meridian reference lines
    ax.axhline(0, color="k", linewidth=0.4)
    ax.axvline(0, color="k", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Ground track saved → {output_path}")
    return output_path


# ── Altitude profile ───────────────────────────────────────────────────────

def plot_altitude_profile(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    step_min:    float = 5.0,
    output_path: str   = "results/altitude_profile.png",
) -> str:
    """Save an altitude-vs-time comparison plot.

    Altitude oscillates between perigee and apogee with orbital period
    (about 95 min for Starlink at ~550 km).  Drag causes a long-term
    secular decrease that the LSTM should capture via the BSTAR feature.

    Parameters
    ----------
    y_true / y_pred : ndarray, shape (N, 3) – ECI positions in km
    step_min        : propagation step in minutes
    output_path     : destination PNG file

    Returns
    -------
    str – *output_path*
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t_min  = np.arange(len(y_true)) * step_min
    alt_t  = np.linalg.norm(y_true, axis=1) - R_EARTH_KM
    alt_p  = np.linalg.norm(y_pred, axis=1) - R_EARTH_KM

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # Top panel – altitude tracks
    axes[0].plot(t_min, alt_t, "b-",  linewidth=1.0, label="SGP4 (true)")
    axes[0].plot(t_min, alt_p, "r--", linewidth=1.0, label="LSTM (predicted)")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Orbital Altitude Profile")
    axes[0].legend()
    axes[0].grid(linestyle="--", alpha=0.4)

    # Bottom panel – altitude error
    axes[1].fill_between(t_min, alt_t - alt_p, color="#1565C0", alpha=0.5)
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_ylabel("Δh (km)")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_title("Altitude Prediction Error")
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Altitude profile saved → {output_path}")
    return output_path


# ── Conjunction distance ───────────────────────────────────────────────────

def plot_conjunction_analysis(
    y_positions:  List[np.ndarray],
    labels:       Optional[List[str]] = None,
    threshold_km: float = 5.0,
    step_min:     float = 5.0,
    output_path:  str   = "results/conjunction_analysis.png",
) -> str:
    """Plot pairwise inter-satellite distances over time for conjunction screening.

    Highlights time windows where any pair approaches within *threshold_km*
    (collision-avoidance alert zone).

    Parameters
    ----------
    y_positions  : list of ndarrays, each shape (N, 3) – ECI positions (km)
    labels       : satellite names (defaults to Sat-0, Sat-1, …)
    threshold_km : conjunction alert threshold in km  (default: 5 km)
    step_min     : propagation step in minutes
    output_path  : destination PNG file

    Returns
    -------
    str – *output_path*
    """
    from itertools import combinations

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_sats = len(y_positions)
    if n_sats < 2:
        print("[viz] Conjunction analysis requires ≥ 2 satellites – skipping.")
        return output_path

    labels = labels or [f"Sat-{i}" for i in range(n_sats)]
    # Trim all to shortest common length
    min_len = min(len(p) for p in y_positions)
    t_min   = np.arange(min_len) * step_min

    pairs  = list(combinations(range(n_sats), 2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(12, 4))
    for (i, j), col in zip(pairs, colors):
        dists = np.linalg.norm(
            y_positions[i][:min_len] - y_positions[j][:min_len], axis=1
        )
        ax.plot(t_min, dists, linewidth=0.9, color=col,
                label=f"{labels[i]} – {labels[j]}")
        # Shade conjunctions
        mask = dists < threshold_km
        if mask.any():
            ax.fill_between(t_min, 0, dists, where=mask,
                            color="red", alpha=0.35, label="_nolegend_")

    ax.axhline(threshold_km, color="red", linestyle="--", linewidth=1.2,
               label=f"Alert threshold ({threshold_km} km)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Inter-satellite distance (km)")
    ax.set_title("Conjunction Analysis – Pairwise Satellite Distances")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Conjunction analysis saved → {output_path}")
    return output_path


# ── Attention weight heatmap ──────────────────────────────────────────────

def plot_attention_weights(
    weights:     np.ndarray,
    step_min:    float = 5.0,
    output_path: str   = "results/attention_weights.png",
    n_samples:   int   = 20,
) -> str:
    """Visualise temporal attention weights as a heatmap.

    The heatmap shows which timesteps in the input window the model focuses
    on most strongly for a random subset of test predictions.  Brighter cells
    indicate higher attention weight.

    Parameters
    ----------
    weights     : ndarray, shape (N, seq_len, 1) – raw attention weights
    step_min    : propagation step in minutes (for axis labels)
    output_path : destination PNG file
    n_samples   : number of prediction samples to show (rows)

    Returns
    -------
    str – *output_path*
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    W = weights[:n_samples, :, 0]  # (n_samples, seq_len)
    seq_len = W.shape[1]
    ticks   = [f"t-{(seq_len-1-i)*int(step_min)}m" for i in range(seq_len)]

    fig, ax = plt.subplots(figsize=(10, max(3, n_samples * 0.35)))
    im = ax.imshow(W, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(ticks, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Input window timestep (relative to prediction point)")
    ax.set_ylabel("Sample index")
    ax.set_title("Temporal Attention Weights (OrbitalLSTMAttention)")
    plt.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[viz] Attention heatmap saved → {output_path}")
    return output_path

