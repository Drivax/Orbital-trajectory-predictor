import numpy as np
from ..data.synthetic import frames

def position_metrics(predicted, target_states):
    error = predicted - target_states[..., :3]
    distance = np.linalg.norm(error, axis=-1)
    local = np.einsum("...ji,...j->...i", frames(target_states), error)
    return {
        "mae_3d_km": float(np.mean(distance)),
        "rmse_3d_km": float(np.sqrt(np.mean(distance**2))),
        "p95_km": float(np.percentile(distance, 95)),
        "radial_mae_km": float(np.mean(np.abs(local[..., 0]))),
        "along_track_mae_km": float(np.mean(np.abs(local[..., 1]))),
        "cross_track_mae_km": float(np.mean(np.abs(local[..., 2]))),
        "relative_error": float(np.mean(distance / np.linalg.norm(target_states[..., :3], axis=-1))),
    }

def summarize(records):
    summaries = []
    keys = sorted({(r["split"], r["model"], r["horizon_min"]) for r in records})
    for split, model, horizon in keys:
        rows = [r for r in records if (r["split"],r["model"],r["horizon_min"]) == (split,model,horizon)]
        item = dict(split=split, model=model, horizon_min=horizon, seed_count=len(rows))
        for metric in ("mae_3d_km","rmse_3d_km","p95_km","radial_mae_km",
                       "along_track_mae_km","cross_track_mae_km","relative_error"):
            values = [r[metric] for r in rows]
            item[metric+"_mean"] = float(np.mean(values))
            item[metric+"_std"] = float(np.std(values, ddof=1)) if len(values)>1 else 0.0
        summaries.append(item)
    return summaries
