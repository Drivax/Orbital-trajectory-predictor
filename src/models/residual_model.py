import copy
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from ..data.synthetic import features, frames

def residual_targets(target, physics):
    if target.shape != physics.shape:
        raise ValueError("Target and physics shapes must match")
    return target - physics

def local_targets(group, kind):
    positions = group["target"][..., :3]
    if kind == "hybrid":
        positions = residual_targets(positions, group["two_body"])
    elif kind != "pure":
        raise ValueError(kind)
    return np.einsum("nji,nhj->nhi", frames(group["state"]), positions).reshape(len(positions), -1)

def train(train_group, val_group, kind, seed, config):
    x_scaler = StandardScaler().fit(features(train_group["state"]))
    y_scaler = StandardScaler().fit(local_targets(train_group, kind))
    x = x_scaler.transform(features(train_group["state"]))
    y = y_scaler.transform(local_targets(train_group, kind))
    xv = x_scaler.transform(features(val_group["state"]))
    yv = y_scaler.transform(local_targets(val_group, kind))
    network = MLPRegressor(hidden_layer_sizes=config.hidden, activation="tanh",
                           solver="adam", batch_size=min(128, len(x)), learning_rate_init=0.001,
                           alpha=1e-4, random_state=seed, shuffle=True)
    best_loss, best, best_epoch, stale, history = np.inf, None, 0, 0, []
    for epoch in range(config.epochs):
        network.partial_fit(x, y)
        loss = float(np.mean((network.predict(xv)-yv)**2))
        history.append({"epoch": epoch+1, "train_loss": float(network.loss_),
                        "validation_standardized_mse": loss})
        if loss < best_loss:
            best_loss, best, best_epoch, stale = loss, copy.deepcopy(network), epoch+1, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return dict(network=best, x_scaler=x_scaler, y_scaler=y_scaler, kind=kind,
                seed=seed, best_epoch=best_epoch, history=history)

def predict(model, group):
    x = model["x_scaler"].transform(features(group["state"]))
    local = model["y_scaler"].inverse_transform(model["network"].predict(x))
    local = local.reshape(len(x), -1, 3)
    output = np.einsum("nij,nhj->nhi", frames(group["state"]), local)
    if model["kind"] == "hybrid":
        output += group["two_body"]
    return output
