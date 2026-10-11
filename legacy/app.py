"""app.py – Streamlit interactive dashboard for the Orbital Trajectory Predictor.

Run:
    streamlit run app.py
"""

import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch

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
    compute_extended_metrics,
    predict_lstm,
    train_lstm,
)
from src.visualization import (
    plot_altitude_profile,
    plot_conjunction_analysis,
    plot_ground_track,
    plot_trajectory_3d,
)

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orbital Trajectory Predictor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2a4a 60%, #0a3d62 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 2.2rem; }
    .hero p  { margin: 0; opacity: 0.8; font-size: 1rem; }

    /* Section cards */
    .section-card {
        background: #f8f9fb;
        border: 1px solid #e2e6ea;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.75rem 0 1.25rem 0;
    }

    /* Metric row spacing */
    div[data-testid="metric-container"] {
        background: #f0f4ff;
        border: 1px solid #d0dcff;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }

    /* Sidebar section labels */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin: 1rem 0 0.3rem 0;
    }

    /* Step badges */
    .step-badge {
        display: inline-block;
        background: #1b4f9c;
        color: white;
        border-radius: 50%;
        width: 1.7rem;
        height: 1.7rem;
        text-align: center;
        line-height: 1.7rem;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🛰️ Orbital Trajectory Predictor</h1>
        <p>
            Predict satellite positions <em>(x, y, z — ECI frame, km)</em> up to <strong>24 h ahead</strong>
            using Two-Line Element (TLE) data and a stacked <strong>LSTM</strong> neural network.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/200px-NASA_logo.svg.png",
        width=60,
    )
    st.markdown("## ⚙️ Configuration")

    st.markdown('<p class="sidebar-section">Prediction</p>', unsafe_allow_html=True)
    hours = st.slider("Horizon (hours)", 1, 24, 24, help="How far ahead to predict.")

    st.markdown('<p class="sidebar-section">Training</p>', unsafe_allow_html=True)
    epochs = st.slider("Epochs", 5, 100, 50, step=5)
    lr = st.select_slider(
        "Learning rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-3,
        format_func=lambda v: f"{v:.0e}",
    )
    batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)

    st.markdown('<p class="sidebar-section">Model</p>', unsafe_allow_html=True)
    use_attention = st.toggle("Attention LSTM", value=True, help="Use the enhanced model with self-attention.")
    run_rf        = st.toggle("Random Forest baseline", value=True, help="Train a RF model for comparison.")

    st.divider()
    device_label = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.caption(f"Compute: **{device_label}**")

# ── TLE input ─────────────────────────────────────────────────────────────────
st.markdown('<span class="step-badge">1</span> **Input TLE Data**', unsafe_allow_html=True)

DEFAULT_TLE = """\
STARLINK-1007
1 44713U 19074A   25091.50000000  .00001234  00000-0  98765-4 0  9991
2 44713  53.0543 249.3959 0001421  76.2878 283.8302 15.05692737 96510
STARLINK-1008
1 44714U 19074B   25091.50000000  .00001100  00000-0  87654-4 0  9999
2 44714  53.0521 123.5432 0001523  91.2345 271.0000 15.04900000 99820
STARLINK-1009
1 44715U 19074C   25091.50000000  .00001050  00000-0  83210-4 0  9998
2 44715  53.0498  10.1234 0001310  88.7654 274.5432 15.05100000 98732
"""

tab_paste, tab_upload = st.tabs(["✏️  Paste TLE", "📂  Upload file"])

with tab_paste:
    tle_text  = st.text_area(
        "TLE data (3-line format — name + line 1 + line 2):",
        value=DEFAULT_TLE,
        height=180,
        label_visibility="collapsed",
    )
    tle_bytes = tle_text.encode()

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a .txt TLE file",
        type=["txt"],
        help="Plain-text file with one or more 3-line TLE blocks.",
    )
    if uploaded:
        tle_bytes = uploaded.read()
        st.success(f"File **{uploaded.name}** loaded ({len(tle_bytes):,} bytes).")
    else:
        tle_bytes = DEFAULT_TLE.encode()
        st.caption("No file uploaded — using built-in Starlink sample data.")

st.divider()

# ── Run prediction ─────────────────────────────────────────────────────────────
col_btn, col_hint = st.columns([1, 4])
with col_btn:
    run = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
with col_hint:
    st.caption("Adjust the sidebar options, then click **Run Prediction** to start.")

if run:
    # ── Step 1 – Parse ────────────────────────────────────────────────────────
    with st.status("Parsing TLEs…", expanded=True) as status:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tf:
            tf.write(tle_bytes)
            tmp_path = tf.name
        try:
            tle_list = load_tle_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        if not tle_list:
            status.update(label="❌ No valid TLE entries found.", state="error")
            st.stop()

        st.write(f"Loaded **{len(tle_list)}** satellite(s).")

        # ── Step 2 – Propagate ────────────────────────────────────────────────
        status.update(label="Propagating orbits with SGP4…")
        prop_hours = max(48.0, hours + WINDOW_SIZE * STEP_MIN / 60.0)
        all_satellite_records = build_dataset(tle_list, hours=prop_hours, step_min=STEP_MIN)
        if not all_satellite_records:
            status.update(label="❌ Propagation produced no records.", state="error")
            st.stop()
        st.write(f"Propagated {len(all_satellite_records)} satellite(s) for {prop_hours:.0f} h.")

        # ── Step 3 – Windows & normalisation ──────────────────────────────────
        status.update(label="Building sequences and normalising…")
        sat_lengths = compute_satellite_window_lengths(all_satellite_records, WINDOW_SIZE)
        X, y = create_windows(all_satellite_records, window_size=WINDOW_SIZE)
        X_train, y_train, X_test, y_test, x_sc, y_sc = split_and_normalize(
            X, y, satellite_lengths=sat_lengths
        )
        st.write(f"Train windows: **{len(X_train)}** · Test windows: **{len(X_test)}**")

        # ── Step 4 – Train LSTM ───────────────────────────────────────────────
        status.update(label="Training LSTM…")
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        model   = OrbitalLSTMAttention() if use_attention else OrbitalLSTM()
        history = train_lstm(
            model, X_train, y_train,
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device,
        )
        st.write(f"Training complete — final loss: **{history['train_loss'][-1]:.6f}**")

        # ── Step 5 – Inference ────────────────────────────────────────────────
        status.update(label="Running inference…")
        y_pred_n  = predict_lstm(model, X_test, device=device)
        y_pred_km = y_sc.inverse_transform(y_pred_n)
        y_true_km = y_sc.inverse_transform(y_test)
        lstm_m    = compute_extended_metrics(y_true_km, y_pred_km)

        status.update(label="✅ All steps complete!", state="complete", expanded=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown('<span class="step-badge">2</span> **Performance Metrics**', unsafe_allow_html=True)

    rmse_ok = lstm_m["RMSE_km"] < 5.0
    if rmse_ok:
        st.success(f"✅  RMSE = **{lstm_m['RMSE_km']:.3f} km** — below the 5 km target.")
    else:
        st.warning(f"⚠️  RMSE = **{lstm_m['RMSE_km']:.3f} km** exceeds the 5 km target.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{lstm_m['RMSE_km']:.3f} km")
    c2.metric("MAE",  f"{lstm_m['MAE_km']:.3f} km")
    c3.metric("P95",  f"{lstm_m['P95_km']:.3f} km")
    c4.metric("Model", "Attention LSTM" if use_attention else "LSTM")

    col_ax, col_loss = st.columns(2)

    with col_ax:
        with st.expander("📐 Per-axis errors", expanded=True):
            axis_df = pd.DataFrame([
                {"Axis": "X", "RMSE (km)": lstm_m["RMSE_x_km"], "MAE (km)": lstm_m["MAE_x_km"]},
                {"Axis": "Y", "RMSE (km)": lstm_m["RMSE_y_km"], "MAE (km)": lstm_m["MAE_y_km"]},
                {"Axis": "Z", "RMSE (km)": lstm_m["RMSE_z_km"], "MAE (km)": lstm_m["MAE_z_km"]},
            ])
            st.dataframe(
                axis_df.style.format({"RMSE (km)": "{:.4f}", "MAE (km)": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

    with col_loss:
        with st.expander("📉 Training loss", expanded=True):
            st.line_chart(
                pd.DataFrame({"Train loss": history["train_loss"], "Val loss": history["val_loss"]}),
                color=["#1b4f9c", "#e05c2a"],
            )

    # ── Random Forest comparison ───────────────────────────────────────────────
    if run_rf:
        with st.spinner("Training Random Forest baseline…"):
            rf      = RandomForestPredictor()
            rf.fit(X_train, y_train)
            y_rf_km = y_sc.inverse_transform(rf.predict(X_test))
            rf_m    = compute_extended_metrics(y_true_km, y_rf_km)

        with st.expander("🌲 Random Forest vs LSTM", expanded=True):
            cmp_df = pd.DataFrame([
                {"Model": "LSTM" + (" + Attention" if use_attention else ""),
                 "RMSE (km)": lstm_m["RMSE_km"], "MAE (km)": lstm_m["MAE_km"], "P95 (km)": lstm_m["P95_km"]},
                {"Model": "Random Forest",
                 "RMSE (km)": rf_m["RMSE_km"],   "MAE (km)": rf_m["MAE_km"],   "P95 (km)": rf_m["P95_km"]},
            ])
            st.dataframe(
                cmp_df.style.format({"RMSE (km)": "{:.4f}", "MAE (km)": "{:.4f}", "P95 (km)": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

    n_disp = min(int(hours * 60.0 / STEP_MIN), len(y_true_km))

    # ── 3-D Trajectory ────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="step-badge">3</span> **3-D Trajectory — Real vs Predicted**', unsafe_allow_html=True)
    fig = plot_trajectory_3d(y_true_km[:n_disp], y_pred_km[:n_disp])
    st.plotly_chart(fig, use_container_width=True)

    # ── Ground track & Altitude side-by-side ─────────────────────────────────
    st.divider()
    col_gt, col_alt = st.columns(2)

    with col_gt:
        st.markdown('<span class="step-badge">4</span> **Ground Track**', unsafe_allow_html=True)
        gt_tmp = os.path.join(tempfile.gettempdir(), "ground_track.png")
        plot_ground_track(
            y_true_km[:n_disp], y_pred_km[:n_disp],
            step_min=STEP_MIN, output_path=gt_tmp,
        )
        st.image(gt_tmp, use_container_width=True)
        with open(gt_tmp, "rb") as fh:
            st.download_button("⬇️ Download ground track", fh, "ground_track.png", "image/png")

    with col_alt:
        st.markdown('<span class="step-badge">5</span> **Altitude Profile**', unsafe_allow_html=True)
        alt_tmp = os.path.join(tempfile.gettempdir(), "altitude_profile.png")
        plot_altitude_profile(
            y_true_km[:n_disp], y_pred_km[:n_disp],
            step_min=STEP_MIN, output_path=alt_tmp,
        )
        st.image(alt_tmp, use_container_width=True)
        with open(alt_tmp, "rb") as fh:
            st.download_button("⬇️ Download altitude profile", fh, "altitude_profile.png", "image/png")

    # ── Conjunction analysis ──────────────────────────────────────────────────
    if len(all_satellite_records) >= 2:
        st.divider()
        st.markdown('<span class="step-badge">6</span> **Conjunction Analysis**', unsafe_allow_html=True)
        sats_pos, sat_names = [], []
        for sat_recs, (sat_name, _, _) in zip(all_satellite_records[:3], tle_list[:3]):
            pos_arr = np.array(
                [[r["x"], r["y"], r["z"]] for r in sat_recs], dtype=np.float32
            )
            sats_pos.append(pos_arr[:n_disp])
            sat_names.append(sat_name)
        conj_tmp = os.path.join(tempfile.gettempdir(), "conjunction.png")
        plot_conjunction_analysis(
            sats_pos, labels=sat_names,
            step_min=STEP_MIN, output_path=conj_tmp,
        )
        st.image(conj_tmp, use_container_width=True)

    # ── Predicted positions table ─────────────────────────────────────────────
    st.divider()
    st.markdown('<span class="step-badge">7</span> **Predicted Positions**', unsafe_allow_html=True)

    rows = [
        {
            "Step": i + 1,
            "Time (min)": (i + 1) * int(STEP_MIN),
            "x (km)": round(float(y_pred_km[i, 0]), 3),
            "y (km)": round(float(y_pred_km[i, 1]), 3),
            "z (km)": round(float(y_pred_km[i, 2]), 3),
            "Error (km)": round(
                float(np.linalg.norm(y_pred_km[i] - y_true_km[i])), 3
            ),
        }
        for i in range(n_disp)
    ]
    pos_df = pd.DataFrame(rows)

    st.dataframe(
        pos_df.style.format(
            {"x (km)": "{:.3f}", "y (km)": "{:.3f}", "z (km)": "{:.3f}", "Error (km)": "{:.3f}"}
        ).background_gradient(subset=["Error (km)"], cmap="YlOrRd"),
        use_container_width=True,
        hide_index=True,
    )

    csv_buf = io.StringIO()
    pos_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download predictions (CSV)",
        csv_buf.getvalue().encode(),
        "predicted_positions.csv",
        "text/csv",
    )
