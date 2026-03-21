"""model.py
OrbitalLSTM – stacked 2-layer LSTM with 256 hidden units per layer.
OrbitalLSTMAttention – LSTM with temporal self-attention for improved accuracy.
RandomForestPredictor – scikit-learn ensemble baseline.
Training, inference, and evaluation utilities.

Architecture (OrbitalLSTM)
---------------------------
Input  → [Batch, 12, 18]   (12 timesteps × 18 features)
   ↓
LSTM Layer 1  (256 hidden units, returns all timesteps)
   ↓
LSTM Layer 2  (256 hidden units, returns last timestep)
   ↓
Linear(256 → 3)
   ↓
Output → [Batch, 3]   # (x̂, ŷ, ẑ) in normalised km

Architecture (OrbitalLSTMAttention) – enhanced model
------------------------------------------------------
Input  → [Batch, 12, 18]
   ↓
LSTM Layer 1  (256 hidden units, return_sequences=True)
   ↓
LSTM Layer 2  (256 hidden units, return_sequences=True)
   ↓
Temporal Attention  → weighted context vector  [Batch, 256]
   ↓
Dropout(0.3)
   ↓
Linear(256 → 128) + ReLU
   ↓
Linear(128 → 3)
   ↓
Output → [Batch, 3]   # (x̂, ŷ, ẑ) in normalised km

Temporal Attention
------------------
Given LSTM output H ∈ R^{L×d}, attention scores are:

    α = softmax(W_a H^⊤)       ∈ R^L
    context = Σ_t α_t h_t      ∈ R^d

This allows the model to focus on the most informative timesteps in the
input window rather than relying solely on the last hidden state.

LSTM cell equations at each timestep t
---------------------------------------
f_t = σ(W_f [h_{t-1}; x_t] + b_f)          forget gate
i_t = σ(W_i [h_{t-1}; x_t] + b_i)          input  gate
c̃_t = tanh(W_c [h_{t-1}; x_t] + b_c)       candidate cell
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t           cell state
o_t = σ(W_o [h_{t-1}; x_t] + b_o)          output gate
h_t = o_t ⊙ tanh(c_t)                       hidden state
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import N_FEATURES


# ── LSTM model ──────────────────────────────────────────────────────────────

class OrbitalLSTM(nn.Module):
    """Stacked LSTM for satellite position prediction.

    Parameters
    ----------
    input_dim  : number of input features per timestep (default: N_FEATURES=18)
    hidden_dim : LSTM hidden units per layer (default: 128)
    num_layers : number of stacked LSTM layers (default: 2)
    output_dim : number of output coordinates (default: 3 → x, y, z)
    dropout    : dropout probability between LSTM layers (default: 0.3)
    """

    def __init__(
        self,
        input_dim:  int   = N_FEATURES,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        output_dim: int   = 3,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        Tensor, shape (batch, output_dim)
        """
        out, _ = self.lstm(x)           # (batch, seq_len, hidden_dim)
        return self.fc(out[:, -1, :])   # last timestep → (batch, output_dim)


# ── Temporal attention layer ─────────────────────────────────────────────────

class _TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) temporal self-attention over LSTM outputs.

    Given a sequence of hidden states H ∈ R^{L × d}, computes a weighted
    context vector:

    .. math::
        \\mathbf{e}_t = W_a \\mathbf{h}_t + b_a \\in \\mathbb{R}
        \\quad (\\text{scalar energy for each timestep})

        \\boldsymbol{\\alpha} = \\text{softmax}(\\mathbf{e}) \\in \\mathbb{R}^L

        \\mathbf{c} = \\sum_{t=1}^{L} \\alpha_t \\mathbf{h}_t \\in \\mathbb{R}^d
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : Tensor, shape (batch, seq_len, hidden_dim)

        Returns
        -------
        context : Tensor, shape (batch, hidden_dim) – attended context vector
        weights : Tensor, shape (batch, seq_len, 1) – attention weights
        """
        energies = self.score(lstm_out)                         # (batch, seq, 1)
        weights  = torch.softmax(energies, dim=1)               # (batch, seq, 1)
        context  = (weights * lstm_out).sum(dim=1)              # (batch, hidden)
        return context, weights


# ── Attention-enhanced LSTM ──────────────────────────────────────────────────

class OrbitalLSTMAttention(nn.Module):
    """Stacked LSTM with temporal attention for improved trajectory prediction.

    This model attends over all timesteps in the input window instead of
    relying on the last hidden state alone, allowing it to selectively weight
    the most physically informative observations (e.g. maneuver signatures,
    drag-induced acceleration spikes).

    Parameters
    ----------
    input_dim  : input feature dimension (default: N_FEATURES=18)
    hidden_dim : LSTM hidden units per layer (default: 128)
    num_layers : stacked LSTM depth (default: 2)
    output_dim : output positions (default: 3)
    dropout    : dropout on LSTM inter-layer and FC (default: 0.3)
    """

    def __init__(
        self,
        input_dim:  int   = N_FEATURES,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        output_dim: int   = 3,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(hidden_dim // 2, output_dim)
        # Residual skip connection: linear map from last timestep's input
        # features directly to the output.  This provides a physics-informed
        # shortcut (velocity features → displacement) so the LSTM only needs
        # to learn perturbation corrections (drag, J2, etc.).
        self.skip      = nn.Linear(input_dim, output_dim)
        # Zero-init so the skip path contributes nothing at the start;
        # the LSTM learns first, then the skip gradually activates.
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x                : Tensor, shape (batch, seq_len, input_dim)
        return_attention : if True, also return attention weight tensor

        Returns
        -------
        output  : Tensor, shape (batch, output_dim)
        weights : (only when return_attention=True) shape (batch, seq_len, 1)
        """
        lstm_out, _       = self.lstm(x)              # (batch, seq, hidden)
        context, weights  = self.attention(lstm_out)  # (batch, hidden)
        context           = self.dropout(context)
        h                 = self.relu(self.fc1(context))
        out               = self.fc2(h) + self.skip(x[:, -1, :])  # residual path
        if return_attention:
            return out, weights
        return out


# ── Training ────────────────────────────────────────────────────────────────

def train_lstm(
    model:         nn.Module,
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    epochs:        int   = 50,
    lr:            float = 1e-3,
    batch_size:    int   = 64,
    patience:      int   = 5,
    val_split:     float = 0.1,
    device:        str   = "cpu",
    grad_clip:     float = 1.0,
) -> Dict[str, List[float]]:
    """Train *model* with Adam optimiser, gradient clipping, and early stopping.

    Adam update rule::

        m_t = β₁ m_{t-1} + (1−β₁) g_t          (first  moment)
        v_t = β₂ v_{t-1} + (1−β₂) g_t²          (second moment)
        m̂_t = m_t / (1−β₁ᵗ)                    (bias-corrected)
        v̂_t = v_t / (1−β₂ᵗ)                    (bias-corrected)
        θ_t = θ_{t-1} − η / (√v̂_t + ε) · m̂_t

    Gradient clipping (L2 norm bound)::

        g_t ← g_t · min(1, max_norm / ‖g_t‖)

    This prevents gradient explosions during BPTT, which can occur when the
    forget gate values deviate from 1 over long sequences.

    ReduceLROnPlateau scheduler halves the learning rate after ``patience``
    epochs without validation improvement, allowing finer convergence.

    Training loss is MSE in normalised coordinates::

        L_MSE = (1/N) Σ ‖r_i − r̂_i‖₂²

    Parameters
    ----------
    model      : un-trained :class:`OrbitalLSTM` or :class:`OrbitalLSTMAttention`
    X_train    : float32 ndarray, shape (N, window_size, N_FEATURES)
    y_train    : float32 ndarray, shape (N, 3)
    epochs     : maximum number of training epochs
    lr         : Adam learning rate
    batch_size : mini-batch size
    patience   : early-stopping patience (epochs without val improvement)
    val_split  : fraction of X_train held out for validation
    device     : ``"cpu"`` or ``"cuda"``
    grad_clip  : max L2 norm for gradient clipping (0 = disabled)

    Returns
    -------
    dict with keys ``"train_loss"`` and ``"val_loss"``
    (per-epoch MSE on normalised coordinates)
    """
    model = model.to(device)

    # Random validation split – X_train already has per-satellite
    # chronological splitting applied, so a random holdout ensures all
    # orbit families are represented in both train and validation sets.
    rng = np.random.RandomState(42)
    n_total = len(X_train)
    indices = rng.permutation(n_total)
    split = int(n_total * (1.0 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).to(device)

    loader = DataLoader(
        TensorDataset(_t(X_tr), _t(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = _t(X_val)
    y_val_t = _t(y_val)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 2),
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val     = float("inf")
    patience_ctr = 0
    best_state: Optional[dict] = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            # OrbitalLSTMAttention returns (pred, weights) when not using
            # return_attention; calling model(xb) always returns pred only.
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= max(len(X_tr), 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val     = val_loss
            patience_ctr = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[LSTM] Early stopping at epoch {epoch + 1}/{epochs}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[LSTM] Epoch {epoch + 1:3d}/{epochs}  "
                f"train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}  "
                f"lr={current_lr:.2e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ── Inference ────────────────────────────────────────────────────────────────

def predict_lstm(
    model:  nn.Module,
    X:      np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference and return predictions as a NumPy array.

    Parameters
    ----------
    model  : trained :class:`OrbitalLSTM` or :class:`OrbitalLSTMAttention`
    X      : float32 ndarray, shape (N, window_size, N_FEATURES)
    device : inference device

    Returns
    -------
    ndarray, shape (N, 3) – normalised predictions
    """
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return preds


# ── Random Forest baseline ──────────────────────────────────────────────────

class RandomForestPredictor:
    """Scikit-learn Random Forest regression baseline.

    The input window X (shape W × F) is flattened into a 1-D feature vector
    of length W*F = 12*N_FEATURES = 12*18 = 216 before fitting.

    The ensemble prediction averages T trees::

        r̂ = (1/T) Σ f_t(x)

    Variance reduction relative to a single tree::

        Var[(1/T) Σ f_t] = σ²((1−ρ)/T + ρ)

    where ρ is the average pairwise tree correlation (reduced by random
    feature subsampling) and σ² is the individual tree variance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_jobs:       int = -1,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestPredictor":
        """Fit on (N, W, F) windows and (N, 3) targets."""
        self.model.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions, shape (N, 3)."""
        return self.model.predict(X.reshape(len(X), -1))


# ── Evaluation metrics ──────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE and MAE in kilometres.

    Both *y_true* and *y_pred* must be in the **original (km)** coordinate
    space (i.e. after inverse-transforming the scaler).

    RMSE = √( (1/N) Σ ‖r_i − r̂_i‖₂² )
    MAE  =    (1/N) Σ ‖r_i − r̂_i‖₂

    Parameters
    ----------
    y_true : ndarray, shape (N, 3) – ground-truth positions in km
    y_pred : ndarray, shape (N, 3) – predicted positions in km

    Returns
    -------
    dict with keys ``"RMSE_km"`` and ``"MAE_km"``
    """
    dist = np.linalg.norm(y_true - y_pred, axis=1)  # (N,) 3-D distances
    return {
        "RMSE_km": float(np.sqrt(np.mean(dist ** 2))),
        "MAE_km":  float(np.mean(dist)),
    }


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute extended position-error metrics in kilometres.

    Includes 3-D RMSE and MAE plus per-axis (Δx, Δy, Δz) statistics and
    the 95th-percentile position error, which is important for collision
    avoidance risk assessment.

    Parameters
    ----------
    y_true : ndarray, shape (N, 3) – ground-truth positions in km
    y_pred : ndarray, shape (N, 3) – predicted positions in km

    Returns
    -------
    dict with keys:
        ``"RMSE_km"``, ``"MAE_km"``,
        ``"RMSE_x_km"``, ``"RMSE_y_km"``, ``"RMSE_z_km"``,
        ``"MAE_x_km"``,  ``"MAE_y_km"``,  ``"MAE_z_km"``,
        ``"P95_km"``  (95th percentile 3-D error)
    """
    diff = y_true - y_pred                              # (N, 3)
    dist = np.linalg.norm(diff, axis=1)                 # (N,)

    metrics: Dict[str, float] = {
        "RMSE_km":   float(np.sqrt(np.mean(dist**2))),
        "MAE_km":    float(np.mean(dist)),
        "P95_km":    float(np.percentile(dist, 95)),
        "RMSE_x_km": float(np.sqrt(np.mean(diff[:, 0]**2))),
        "RMSE_y_km": float(np.sqrt(np.mean(diff[:, 1]**2))),
        "RMSE_z_km": float(np.sqrt(np.mean(diff[:, 2]**2))),
        "MAE_x_km":  float(np.mean(np.abs(diff[:, 0]))),
        "MAE_y_km":  float(np.mean(np.abs(diff[:, 1]))),
        "MAE_z_km":  float(np.mean(np.abs(diff[:, 2]))),
    }
    return metrics
