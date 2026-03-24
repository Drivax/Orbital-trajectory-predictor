# Orbital Trajectory Predictor

<p align="center">
  <img src="results/trajectory_3d_snapshot.png" alt="3D Trajectory Comparison" width="700"/>
</p>

---

## Project Objective

This project predicts the short-term positional displacement of low-Earth orbit satellites from Two-Line Element (TLE) data. Given one hour of recent orbital history, the model estimates where a satellite will be five minutes later — expressed as a 3D displacement vector $(\Delta x,\, \Delta y,\, \Delta z)$ in the Earth-Centered Inertial (ECI) frame.

The motivation is practical. Standard orbit propagators like SGP4 accumulate errors over time because they rely on simplified analytical equations. A learned model trained on real orbital data can capture perturbation patterns — atmospheric drag, geopotential harmonics, solar pressure — that the analytical model approximates but does not fully resolve. This project explores how far a sequence model can go using only TLE-derived features and no additional sensor data.

---

## Dataset

The dataset consists of **31 Starlink satellites** from the public TLE catalog maintained by [Celestrak](https://celestrak.org/). The satellites span three inclination shells (nominally 53°, 70°, and 51.6°), which provides some orbital diversity while keeping the dataset homogeneous in terms of satellite type and altitude regime.

Each TLE is propagated using the SGP4 standard propagator (via the `skyfield` library) at a **5-minute cadence** over a **48-hour window**. This produces 576 steps per satellite and **17,856 records** in total.

Each timestep is described by a **21-dimensional feature vector** built from:

| Group | Features |
|---|---|
| Keplerian elements | inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion |
| Diurnal time encoding | sin and cos of elapsed time over one sidereal day |
| Physics-derived | altitude (km), vis-viva orbital speed (km/s), BSTAR drag coefficient, argument of latitude |
| ECI position context | x, y, z in km |
| Smooth angular encodings | sin and cos of argument of latitude, sin of inclination |
| Finite-difference velocity | vx, vy, vz in km/s |

The regression target is the **position displacement** at the next step: $\Delta\mathbf{r}_{k+1} = \mathbf{r}_{k+1} - \mathbf{r}_k \in \mathbb{R}^3$.

---

## Methodology

### 1. SGP4 Propagation and Feature Engineering

Each TLE is parsed and fed to SGP4, which returns the satellite's ECI position at every 5-minute mark. Several derived quantities are then computed: altitude above the WGS84 reference sphere, orbital speed from the vis-viva equation, and the argument of latitude (argument of perigee plus mean anomaly, mod 360 degrees).

Angular features like inclination and argument of latitude are encoded as sine and cosine pairs. This matters because a raw angle of 359 degrees and 1 degree are numerically close but appear far apart to a network. The sinusoidal representation removes that discontinuity cleanly.

Finite-difference velocity is appended in a second pass over the propagated positions. Since the displacement $\Delta\mathbf{r} \approx \mathbf{v} \cdot \Delta t$, velocity is essentially a linearised prediction of the target. Including it directly in the input gives the model a strong physical prior from the beginning.

### 2. Sequence Construction and Normalization

Input windows of length $L = 12$ steps (one hour of history) are created as sliding windows across each satellite's time series independently. Windows are never created across satellite boundaries, as mixing two unrelated orbit states would produce physically meaningless sequences.

Input features are scaled to $[0, 1]$ with Min-Max normalization. Displacement targets are standardized to zero mean and unit variance, which suits the roughly zero-centered displacement distribution better than min-max scaling.

The train/test split is **80% / 20% per satellite**, applied in strict chronological order. This prevents future data from leaking into training and ensures every orbit family is represented in both sets.

### 3. LSTM with Temporal Attention

The main model (`OrbitalLSTMAttention`) is a two-layer stacked LSTM with 128 hidden units per layer, followed by a temporal self-attention mechanism, a dropout layer (0.3), and two fully-connected layers.

The attention layer computes a weighted context vector from all 12 LSTM hidden states. It learns to focus on the most informative timesteps in the window — for instance, the periapsis where drag is strongest, or a sharp velocity change. This is more expressive than reading off only the last hidden state.

A **residual skip connection** maps the last input timestep's feature vector directly to the output through a zero-initialized linear layer. It contributes nothing at the start of training and gradually activates as the model converges, acting as a physics-informed shortcut. The LSTM then only needs to learn the perturbation correction on top of this linear baseline.

Training uses the Adam optimizer with learning rate $5 \times 10^{-4}$, L2 weight decay ($10^{-4}$), gradient clipping (norm bound 1.0), and a ReduceLROnPlateau scheduler that halves the rate after 12 stagnant epochs. Maximum training is 200 epochs with early stopping at patience 25 and batch size 64.

A **Random Forest** (100 trees) is trained on the flattened input windows as a non-recurrent baseline.

---

## Key Equations

**Vis-viva equation** — gives orbital speed at any point along the trajectory:

$$v = \sqrt{\mu\left(\frac{2}{r} - \frac{1}{a}\right)}$$

$\mu = 3.986 \times 10^5\ \mathrm{km^3/s^2}$ is Earth's gravitational parameter, $r$ is the current geocentric distance, and $a$ is the semi-major axis. At 550 km altitude, this gives roughly 7.5 km/s. It is computed at every propagation step and included directly as a feature.

---

**LSTM gate equations** — at each timestep $t$ the cell state is updated through learned gates:

$$\mathbf{f}_t = \sigma\!\left(W_f [\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_f\right), \qquad \mathbf{i}_t = \sigma\!\left(W_i [\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_i\right)$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh\!\left(W_c [\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_c\right)$$

$$\mathbf{h}_t = \sigma\!\left(W_o [\mathbf{h}_{t-1};\, \mathbf{x}_t] + \mathbf{b}_o\right) \odot \tanh(\mathbf{c}_t)$$

The forget gate $\mathbf{f}_t$ controls how much of the previous cell state survives. When it is close to 1, gradients flow back without vanishing — which is what makes LSTMs effective on sequences where relevant context spans many steps. For orbital data, dependencies across several orbital periods matter (e.g., accumulated drag over multiple passes), so this property is directly useful.

---

**Temporal attention** — after the second LSTM layer produces the hidden sequence $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_L]$, a scalar energy is computed for each step and softmax-normalized:

$$e_t = \mathbf{w}_a^\top \mathbf{h}_t + b_a, \qquad \alpha_t = \frac{\exp(e_t)}{\displaystyle\sum_{j=1}^{L} \exp(e_j)}, \qquad \mathbf{c} = \sum_{t=1}^{L} \alpha_t\, \mathbf{h}_t$$

The weights $\alpha_t$ sum to one across the 12 input steps. The model can assign nearly all weight to a single step (sharp focus) or spread it evenly, whichever is more predictive for the current orbital configuration.

---

**J2 perturbing acceleration** — the dominant perturbation in LEO, caused by Earth's equatorial flattening:

$$\mathbf{a}_{J_2} = -\frac{3\mu J_2 R_\oplus^2}{2r^5}
\begin{pmatrix}
x\!\left(1 - 5z^2/r^2\right) \\
y\!\left(1 - 5z^2/r^2\right) \\
z\!\left(3 - 5z^2/r^2\right)
\end{pmatrix}$$

$J_2 = 1.08263 \times 10^{-3}$, $R_\oplus = 6378.137\ \mathrm{km}$. This acceleration is not a model input; it shapes the dynamics the LSTM must learn. It causes the orbital plane to precess in RAAN and the argument of perigee to drift, at rates that depend on inclination. SGP4 captures this analytically; the neural model learns the residuals on top of that approximation.

---

**Training loss** — mean squared error on normalized position displacement:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left\| \Delta\mathbf{r}_i - \Delta\hat{\mathbf{r}}_i \right\|_2^2$$

Predicting displacement rather than absolute position removes the baseline orbital radius ($\approx 7000\ \mathrm{km}$) from the regression target, reducing output variance by roughly a factor of nine and accelerating convergence.

---

## Evaluation

Metrics are computed on the held-out test set after inverse-transforming predictions back to kilometers. Per-sample error is the 3D Euclidean norm $e_i = \|\Delta\mathbf{r}_i - \Delta\hat{\mathbf{r}}_i\|_2$.

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum e_i^2}, \qquad \text{MAE} = \frac{1}{N}\sum e_i, \qquad P_{95} = \text{percentile}_{95}\!\left(e_1, \dots, e_N\right)$$

The 95th-percentile error $P_{95}$ captures tail behavior — the cases where the model is most wrong — which is the operationally relevant quantity for collision avoidance.

| Model | RMSE (km) | MAE (km) | P95 (km) |
|---|---|---|---|
| **LSTM + Attention** (2 layers, 128 units) | **41.9** | **37.7** | **69.7** |
| Random Forest (100 trees, baseline) | 63.3 | 53.3 | — |

**Per-axis RMSE for LSTM + Attention:** $x = 14.2\ \mathrm{km}$, $y = 17.1\ \mathrm{km}$, $z = 35.5\ \mathrm{km}$.

The z-axis error is the largest. The z-component of displacement is most sensitive to orbital inclination and J2-induced nodal precession, both of which vary across the 31 satellites in ways that are harder to generalize from kinematic features alone.

At LEO orbital speed, a satellite covers roughly 2 250 km in five minutes. A MAE of 37.7 km is a relative displacement error of about **1.7%**. The attention LSTM improves on the Random Forest by 34% in RMSE, which confirms that the sequential structure of the one-hour window has real predictive value beyond what a non-recurrent model can extract from a flattened feature vector.

---

## Repository Structure

```
orbital-trajectory-predictor/
├── src/
│   ├── data_loader.py         # TLE parsing, SGP4 propagation, feature engineering (21-D), windowing
│   ├── model.py               # OrbitalLSTM, OrbitalLSTMAttention, RandomForestPredictor, training
│   ├── physics.py             # Vis-viva, J2, drag, coordinate transforms, conjunction analysis
│   └── visualization.py      # 3-D Plotly, ground track, altitude profile, error plots, GIF export
├── data/
│   └── starlink_tle.txt       # 31-satellite TLE dataset (3 inclination shells)
├── notebooks/
│   ├── 01_data_loading.ipynb  # TLE loading and SGP4 propagation walkthrough
│   ├── 02_preprocessing.ipynb # Feature engineering, windowing, and normalization
│   ├── 03_lstm_model.ipynb    # Model definition, training, and evaluation
│   └── 04_results.ipynb       # Metrics, trajectory plots, and model comparison
├── results/
│   ├── trajectory_3d.html          # Interactive 3-D trajectory (Plotly)
│   └── linkedin/                   # Exported figures
├── predict.py                 # CLI inference script
├── app.py                     # Streamlit interactive dashboard
├── lstm.pt                    # Saved model weights
├── requirements.txt
└── README.md
```

---

## Installation and Execution

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**CLI — standard LSTM:**

```bash
python predict.py --tle data/starlink_tle.txt --hours 24
```

**CLI — LSTM with temporal attention:**

```bash
python predict.py --tle data/starlink_tle.txt --hours 24 --attention
```

**CLI — with Random Forest comparison:**

```bash
python predict.py --tle data/starlink_tle.txt --hours 24 --attention --baseline
```

**Re-use pre-trained weights:**

```bash
python predict.py --tle data/starlink_tle.txt --hours 24 --attention --model-path lstm.pt
```

**Interactive dashboard:**

```bash
streamlit run app.py
```

The dashboard accepts TLE input by paste or file upload, lets you choose between the two LSTM variants, and renders the 3-D trajectory comparison, ground track, altitude profile, conjunction analysis, and all evaluation metrics in real time.