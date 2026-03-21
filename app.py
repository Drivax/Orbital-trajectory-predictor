"""app.py – Streamlit interactive dashboard for the Orbital Trajectory Predictor.

Run:
    streamlit run app.py
"""

import os
import sys
import tempfile

import numpy as np
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
)

st.title("🛰️ Orbital Trajectory Predictor")
st.markdown(
    "Predict satellite positions *(x, y, z in ECI frame, km)* up to **24 hours ahead** "
    "using Two-Line Element (TLE) data and a stacked **LSTM** model."
)

# ── Sidebar – configuration ──────────────────────────────────────────────────
st.sidebar.header("⚙️  Configuration")
hours      = st.sidebar.slider("Prediction horizon (hours)", 1, 24, 24)
epochs     = st.sidebar.slider("Training epochs", 5, 100, 50, step=5)
lr         = st.sidebar.select_slider(
    "Learning rate",
    options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    value=1e-3,
)
batch_size    = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
use_attention = st.sidebar.checkbox("Use Attention LSTM (enhanced model)", value=True)
run_rf        = st.sidebar.checkbox("Compare Random Forest baseline", value=True)

# ── TLE input ────────────────────────────────────────────────────────────────
st.header("1 · Input TLE Data")

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

input_method = st.radio("Input method", ["Paste TLE text", "Upload TLE file"], horizontal=True)

if input_method == "Paste TLE text":
    tle_text = st.text_area("Paste TLE data (3-line format):", value=DEFAULT_TLE, height=200)
    tle_bytes = tle_text.encode()
else:
    uploaded = st.file_uploader("Upload TLE file (.txt)", type=["txt"])
    tle_bytes = uploaded.read() if uploaded else DEFAULT_TLE.encode()

# ── Run prediction ───────────────────────────────────────────────────────────
if st.button("🚀  Run Prediction", type="primary"):

    # 1. Parse TLEs
    with st.spinner("Parsing TLEs …"):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tf:
            tf.write(tle_bytes)
            tmp_path = tf.name
        try:
            tle_list = load_tle_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        if not tle_list:
            st.error("❌  No valid TLE entries found. Please check the input.")
            st.stop()
        st.success(f"Loaded **{len(tle_list)}** satellite(s).")

    # 2. Propagate
    with st.spinner("Propagating orbits with SGP4 …"):
        prop_hours  = max(48.0, hours + WINDOW_SIZE * STEP_MIN / 60.0)
        all_satellite_records = build_dataset(tle_list, hours=prop_hours, step_min=STEP_MIN)
        if not all_satellite_records:
            st.error("❌  Propagation produced no records.")
            st.stop()

    # 3. Windows & normalisation
    with st.spinner("Building sequences and normalising …"):
        sat_lengths = compute_satellite_window_lengths(all_satellite_records, WINDOW_SIZE)
        X, y = create_windows(all_satellite_records, window_size=WINDOW_SIZE)
        X_train, y_train, X_test, y_test, x_sc, y_sc = split_and_normalize(
            X, y, satellite_lengths=sat_lengths
        )
        st.info(
            f"Training windows: **{len(X_train)}** &nbsp;·&nbsp; "
            f"Test windows: **{len(X_test)}**"
        )

    # 4. Train LSTM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress_bar = st.progress(0, text="Training LSTM …")

    model   = OrbitalLSTMAttention() if use_attention else OrbitalLSTM()
    history = train_lstm(
        model, X_train, y_train,
        epochs=epochs, lr=lr, batch_size=batch_size,
        device=device,
    )
    progress_bar.progress(100, text="Training complete ✅")

    # 5. Inference
    with st.spinner("Running inference …"):
        y_pred_n  = predict_lstm(model, X_test, device=device)
        y_pred_km = y_sc.inverse_transform(y_pred_n)
        y_true_km = y_sc.inverse_transform(y_test)
        lstm_m    = compute_extended_metrics(y_true_km, y_pred_km)

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.header("2 · Performance Metrics")
    col_r, col_m, col_p = st.columns(3)
    col_r.metric("RMSE (km)", f"{lstm_m['RMSE_km']:.3f}")
    col_m.metric("MAE  (km)", f"{lstm_m['MAE_km']:.3f}")
    col_p.metric("P95  (km)", f"{lstm_m['P95_km']:.3f}")

    # Per-axis errors
    with st.expander("📐  Per-axis RMSE"):
        import pandas as pd
        axis_df = pd.DataFrame([{
            "Axis": "X", "RMSE (km)": lstm_m["RMSE_x_km"], "MAE (km)": lstm_m["MAE_x_km"],
        }, {
            "Axis": "Y", "RMSE (km)": lstm_m["RMSE_y_km"], "MAE (km)": lstm_m["MAE_y_km"],
        }, {
            "Axis": "Z", "RMSE (km)": lstm_m["RMSE_z_km"], "MAE (km)": lstm_m["MAE_z_km"],
        }])
        st.dataframe(axis_df, use_container_width=True)

    # Target thresholds
    if lstm_m["RMSE_km"] < 5.0:
        st.success("✅  RMSE < 5 km target achieved.")
    else:
        st.warning(f"⚠️  RMSE = {lstm_m['RMSE_km']:.2f} km exceeds the 5 km target.")

    # Training loss curve
    with st.expander("📉  Training loss curve"):
        import pandas as pd
        st.line_chart(
            pd.DataFrame(
                {"train_loss": history["train_loss"], "val_loss": history["val_loss"]}
            )
        )

    # ── Optional Random Forest comparison ────────────────────────────────────
    if run_rf:
        with st.spinner("Training Random Forest baseline …"):
            rf = RandomForestPredictor()
            rf.fit(X_train, y_train)
            y_rf_km = y_sc.inverse_transform(rf.predict(X_test))
            rf_m    = compute_extended_metrics(y_true_km, y_rf_km)

        st.subheader("Model Comparison")
        cmp = st.columns(2)
        cmp[0].metric("RF RMSE (km)", f"{rf_m['RMSE_km']:.3f}",
                       delta=f"{lstm_m['RMSE_km'] - rf_m['RMSE_km']:.3f}",
                       delta_color="inverse")
        cmp[1].metric("RF MAE (km)",  f"{rf_m['MAE_km']:.3f}",
                       delta=f"{lstm_m['MAE_km'] - rf_m['MAE_km']:.3f}",
                       delta_color="inverse")

    # ── 3-D trajectory ────────────────────────────────────────────────────────
    st.header("3 · 3-D Trajectory (Real vs Predicted)")
    n_disp = min(int(hours * 60.0 / STEP_MIN), len(y_true_km))
    fig    = plot_trajectory_3d(y_true_km[:n_disp], y_pred_km[:n_disp])
    st.plotly_chart(fig, use_container_width=True)

    # ── Ground track ──────────────────────────────────────────────────────────
    st.header("4 · Ground Track")
    import tempfile as _tf, os as _os
    gt_tmp = _os.path.join(_tf.gettempdir(), "ground_track.png")
    plot_ground_track(y_true_km[:n_disp], y_pred_km[:n_disp],
                      step_min=STEP_MIN, output_path=gt_tmp)
    st.image(gt_tmp, use_container_width=True)

    # ── Altitude profile ──────────────────────────────────────────────────────
    st.header("5 · Altitude Profile")
    alt_tmp = _os.path.join(_tf.gettempdir(), "altitude_profile.png")
    plot_altitude_profile(y_true_km[:n_disp], y_pred_km[:n_disp],
                          step_min=STEP_MIN, output_path=alt_tmp)
    st.image(alt_tmp, use_container_width=True)

    # ── Conjunction analysis ──────────────────────────────────────────────────
    if len(all_satellite_records) >= 2:
        st.header("6 · Conjunction Analysis")
        sats_pos  = []
        sat_names = []
        for sat_recs, (sat_name, _, _) in zip(
            all_satellite_records[:3], tle_list[:3]
        ):
            pos_arr = np.array(
                [[r["x"], r["y"], r["z"]] for r in sat_recs], dtype=np.float32
            )
            sats_pos.append(pos_arr[:n_disp])
            sat_names.append(sat_name)
        conj_tmp  = _os.path.join(_tf.gettempdir(), "conjunction.png")
        plot_conjunction_analysis(sats_pos, labels=sat_names,
                                  step_min=STEP_MIN, output_path=conj_tmp)
        st.image(conj_tmp, use_container_width=True)

    # ── Predicted positions table ─────────────────────────────────────────────
    st.header("7 · Predicted Positions")
    import pandas as pd
    rows = [
        {
            "Step": i + 1,
            "Time (min)": (i + 1) * int(STEP_MIN),
            "x (km)": round(float(y_pred_km[i, 0]), 3),
            "y (km)": round(float(y_pred_km[i, 1]), 3),
            "z (km)": round(float(y_pred_km[i, 2]), 3),
        }
        for i in range(n_disp)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
