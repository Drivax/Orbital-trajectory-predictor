#!/usr/bin/env python3
"""predict.py – CLI inference script for the Orbital Trajectory Predictor.

Usage examples
--------------
# Predict 24 h ahead using sample TLEs (trains a fresh model):
    python predict.py --tle data/starlink_tle.txt --hours 24

# Use the enhanced attention LSTM:
    python predict.py --tle data/starlink_tle.txt --hours 24 --attention

# Re-use previously saved weights:
    python predict.py --tle data/starlink_tle.txt --hours 24 --model-path lstm.pt

# Skip plot generation:
    python predict.py --tle data/starlink_tle.txt --no-plot
"""

import argparse
import os
import sys

import numpy as np
import torch

# Make the src package importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import (
    STEP_MIN,
    WINDOW_SIZE,
    build_dataset,
    compute_satellite_window_lengths,
    create_windows,
    load_tle_file,
    split_and_normalize,
)
from src.model import (
    OrbitalLSTM,
    OrbitalLSTMAttention,
    RandomForestPredictor,
    compute_metrics,
    compute_extended_metrics,
    predict_lstm,
    train_lstm,
)
from src.visualization import (
    plot_altitude_profile,
    plot_conjunction_analysis,
    plot_error_components,
    plot_ground_track,
    plot_rmse_comparison,
    plot_trajectory_3d,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_or_train(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    model_path: str,
    device:     str,
    use_attention: bool = False,
) -> torch.nn.Module:
    model = OrbitalLSTMAttention() if use_attention else OrbitalLSTM()
    model_label = "OrbitalLSTMAttention" if use_attention else "OrbitalLSTM"
    if model_path and os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[predict] Loaded {model_label} weights from '{model_path}'")
    else:
        print(f"[predict] No pre-trained model found – training {model_label} (200 epochs) …")
        train_lstm(
            model, X_train, y_train,
            epochs=200, lr=5e-4, batch_size=64, patience=25,
            device=device,
        )
        if model_path:
            torch.save(model.state_dict(), model_path)
            print(f"[predict] Model saved → '{model_path}'")
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict satellite trajectory from TLE data using a trained LSTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tle",
        required=True,
        help="Path to TLE file (3-line or 2-line format).",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Prediction horizon in hours.",
    )
    parser.add_argument(
        "--model-path",
        default="lstm.pt",
        help="Path to saved LSTM weights (.pt). Trains from scratch if not found.",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Use the OrbitalLSTMAttention model (enhanced, with temporal attention).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also train and evaluate the Random Forest baseline.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots and GIF.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[predict] Device : {device}")

    # ── 1. Load & propagate TLEs ────────────────────────────────────────────
    print(f"[predict] Loading TLEs from '{args.tle}' …")
    tle_list = load_tle_file(args.tle)
    if not tle_list:
        print("[predict] ERROR: No valid TLE entries found.")
        sys.exit(1)
    print(f"[predict] Found {len(tle_list)} satellite(s). Propagating …")

    # Propagate at least 48 h so the training window is meaningful
    prop_hours = max(48.0, args.hours + WINDOW_SIZE * STEP_MIN / 60.0)
    all_satellite_records = build_dataset(tle_list, hours=prop_hours, step_min=STEP_MIN)
    if not all_satellite_records:
        print("[predict] ERROR: Propagation produced no records.")
        sys.exit(1)
    total_records = sum(len(r) for r in all_satellite_records)
    print(f"[predict] Total propagation records : {total_records} across {len(all_satellite_records)} satellite(s)")

    # ── 2. Build windows & normalise ────────────────────────────────────────
    sat_lengths = compute_satellite_window_lengths(all_satellite_records, WINDOW_SIZE)
    X, y = create_windows(all_satellite_records, window_size=WINDOW_SIZE)
    print(f"[predict] Windows : X={X.shape}  y={y.shape}")

    X_train, y_train, X_test, y_test, x_sc, y_sc = split_and_normalize(
        X, y, satellite_lengths=sat_lengths
    )
    print(
        f"[predict] Train={len(X_train)}  Test={len(X_test)} "
        f"(80/20 per-satellite chronological split)"
    )

    # ── 3. LSTM (standard or attention) ─────────────────────────────────────
    model   = _load_or_train(X_train, y_train, args.model_path, device,
                             use_attention=args.attention)
    y_norm  = predict_lstm(model, X_test, device=device)
    y_pred  = y_sc.inverse_transform(y_norm)
    y_true  = y_sc.inverse_transform(y_test)

    lstm_metrics = compute_extended_metrics(y_true, y_pred)
    results      = {"LSTM": lstm_metrics}

    print("\n─── LSTM test-set metrics ─────────────────────────────────────")
    print(f"  RMSE      : {lstm_metrics['RMSE_km']:.3f} km")
    print(f"  MAE       : {lstm_metrics['MAE_km']:.3f} km")
    print(f"  P95 error : {lstm_metrics['P95_km']:.3f} km")
    print(f"  RMSE x/y/z: {lstm_metrics['RMSE_x_km']:.3f} / "
          f"{lstm_metrics['RMSE_y_km']:.3f} / {lstm_metrics['RMSE_z_km']:.3f} km")
    print("───────────────────────────────────────────────────────────────\n")

    # ── 4. Optional Random Forest baseline ──────────────────────────────────
    if args.baseline:
        print("[predict] Training Random Forest baseline …")
        rf = RandomForestPredictor()
        rf.fit(X_train, y_train)
        y_rf_norm    = rf.predict(X_test)
        y_rf_pred    = y_sc.inverse_transform(y_rf_norm)
        rf_metrics   = compute_extended_metrics(y_true, y_rf_pred)
        results["Random Forest"] = rf_metrics
        print(f"  RF RMSE : {rf_metrics['RMSE_km']:.3f} km")
        print(f"  RF MAE  : {rf_metrics['MAE_km']:.3f} km\n")

    # ── 5. Visualisations ───────────────────────────────────────────────────
    if not args.no_plot:
        os.makedirs("results", exist_ok=True)

        n_disp   = min(int(args.hours * 60.0 / STEP_MIN), len(y_true))

        # Interactive 3-D trajectory (HTML)
        fig      = plot_trajectory_3d(y_true[:n_disp], y_pred[:n_disp])
        html_out = "results/trajectory_3d.html"
        fig.write_html(html_out)
        print(f"[predict] 3-D trajectory saved → '{html_out}'")

        # Per-component error
        plot_error_components(y_true[:n_disp], y_pred[:n_disp])

        # Ground track
        plot_ground_track(y_true[:n_disp], y_pred[:n_disp], step_min=STEP_MIN)

        # Altitude profile
        plot_altitude_profile(y_true[:n_disp], y_pred[:n_disp], step_min=STEP_MIN)

        # Conjunction analysis (multi-satellite only)
        if len(all_satellite_records) >= 2:
            # Use SGP4 true positions from each satellite's test window
            sats_pos  = []
            sat_names = []
            for i, (sat_recs, (sat_name, _, _)) in enumerate(
                zip(all_satellite_records[:3], tle_list[:3])
            ):
                pos_arr = np.array(
                    [[r["x"], r["y"], r["z"]] for r in sat_recs], dtype=np.float32
                )
                sats_pos.append(pos_arr[:n_disp])
                sat_names.append(sat_name)
            if len(sats_pos) >= 2:
                plot_conjunction_analysis(
                    sats_pos, labels=sat_names, step_min=STEP_MIN
                )

        # RMSE comparison bar chart
        if len(results) > 1:
            plot_rmse_comparison(results)

    # ── 6. Print predicted positions table ──────────────────────────────────
    n_future = min(int(args.hours * 60.0 / STEP_MIN), len(y_pred))
    print(
        f"Predicted positions for the next {args.hours:.0f} h "
        f"({n_future} steps × {STEP_MIN:.0f} min):\n"
    )
    print(f"  {'Step':>5}  {'x (km)':>12}  {'y (km)':>12}  {'z (km)':>12}")
    print("  " + "-" * 46)
    for i in range(n_future):
        x_km, y_km, z_km = y_pred[i]
        print(f"  {i+1:>5}  {x_km:>12.3f}  {y_km:>12.3f}  {z_km:>12.3f}")


if __name__ == "__main__":
    main()

