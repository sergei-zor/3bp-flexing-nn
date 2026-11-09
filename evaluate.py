"""
python evaluate.py --model models/surrogate
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from threebp.data import FEATURES, TARGET, load_long, test_mask
from threebp.metrics import regression_metrics
from threebp.model import Surrogate


def parity_plot(y, y_pred, m, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y, y_pred, s=2, alpha=0.3, rasterized=True)
    lim = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.text(0.05, 0.95, f"R = {m['R']:.4f}\nR$^2$ = {m['R2']:.4f}\nMAE = {m['MAE_mm']:.2f} mm",
            transform=ax.transAxes, va="top")
    ax.set(xlabel="FE $v_e$ (mm)", ylabel="NN-predicted $v_e$ (mm)", title="Test set", aspect="equal")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/fe_dataset_training.csv")
    p.add_argument("--model", default="models/surrogate")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((Path(args.model) / "config.json").read_text())
    long = load_long(args.data)
    test = test_mask(len(long), cfg["test_frac"], cfg["split_seed"])
    y = long[TARGET].to_numpy()
    y_pred = Surrogate.load(args.model).predict(long[FEATURES].to_numpy())

    metrics = {"train": regression_metrics(y[~test], y_pred[~test]),
               "test": regression_metrics(y[test], y_pred[test])}
    print(json.dumps(metrics, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    long.assign(v_e_pred=y_pred, test=test).to_csv(out / "predictions.csv", index=False)
    parity_plot(y[test], y_pred[test], metrics["test"], out / "parity")


if __name__ == "__main__":
    main()
