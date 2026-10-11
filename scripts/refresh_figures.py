"""Regenerate figures from saved arrays without retraining."""
import json
from pathlib import Path
import numpy as np
from src.config import Config
from src.visualization.plots import make_figures


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results"
    config = Config(**json.loads((output / "run_metadata.json").read_text())["config"])
    with np.load(output / "dataset.npz") as saved:
        groups = {split: {key[len(split)+1:]: saved[key] for key in saved.files
                          if key.startswith(split + "_")}
                  for split in ("train", "val", "test", "ood")}
    with np.load(output / "predictions.npz") as saved:
        predictions = {}
        for key in saved.files:
            split, name, seed = key.split("__")
            predictions[(split, name.replace("_", " "), int(seed))] = saved[key]
    summary = json.loads((output / "metrics.json").read_text())["summary"]
    make_figures(output / "figures", groups, predictions, summary, config)


if __name__ == "__main__":
    main()
