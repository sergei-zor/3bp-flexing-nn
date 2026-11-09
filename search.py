"""
python search.py --model models/surrogate --budget 20000 --seeds 10
"""
import argparse
import time
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threebp.data import D_GRID
from threebp.geometry import LOWER, UPPER, is_valid
from threebp.model import Surrogate

COORDS = ["x1", "y3", "y4", "y7", "y8"]


def min_ve(sur, x, w, s):
    n, k = len(x), len(D_GRID)
    X = np.column_stack([np.repeat(x, k, axis=0), np.full(n * k, w), np.full(n * k, s), np.tile(D_GRID, n)])
    return sur.predict(X).reshape(n, k).min(axis=1)


def sample_valid(rng, n, w):
    x = np.empty((0, 5))
    while len(x) < n:
        cand = rng.uniform(LOWER, UPPER, size=(20 * n, 5))
        x = np.vstack([x, cand[is_valid(cand, w)]])
    return x[:n]


def random_screening(sur, budget, rng, w, s):
    x = sample_valid(rng, budget, w)
    return x, min_ve(sur, x, w, s)


def cmaes(sur, budget, rng, w, s):
    xs, fs = [], []

    def objective(u):
        x = LOWER + u * (UPPER - LOWER)
        if not is_valid(x, w)[0]:
            return np.nan
        xs.append(x)
        fs.append(min_ve(sur, x[None], w, s)[0])
        return fs[-1]

    x0 = lambda: (sample_valid(rng, 1, w)[0] - LOWER) / (UPPER - LOWER)
    opts = {"bounds": [0, 1], "seed": int(rng.integers(1, 2**31)), "tolfun": 1e-4, "tolx": 1e-5, "verbose": -9}
    cma.fmin2(objective, x0, 0.3, opts, restarts={"maxrestarts": 1000, "maxfevals": budget}, incpopsize=2)
    return np.array(xs[:budget]), np.array(fs[:budget])


def shortlist_for_fe(x, f, k, min_dist=0.1):
    u = (x - LOWER) / (UPPER - LOWER)
    chosen = []
    for i in np.argsort(f):
        if all(np.linalg.norm(u[i] - u[j]) >= min_dist for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return chosen


def plot_convergence(curves, out):
    fig, ax = plt.subplots(figsize=(5, 3.75))
    for method, label in [("random", "Random screening"), ("cmaes", "CMA-ES")]:
        c = curves[curves.method == method].pivot(index="n_evals", columns="seed", values="best_f")
        line, = ax.plot(c.index, c.median(axis=1), label=label)
        ax.fill_between(c.index, c.quantile(0.25, axis=1), c.quantile(0.75, axis=1),
                        color=line.get_color(), alpha=0.25, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Number of surrogate evaluations")
    ax.set_ylabel(r"Best predicted min$_D$ $v_e$ (mm)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"convergence.{ext}", dpi=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/fe_dataset_training.csv")
    p.add_argument("--model", default="models/surrogate")
    p.add_argument("--budget", type=int, default=20000)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--w", type=float, default=0.03)
    p.add_argument("--s", type=float, default=2.0)
    p.add_argument("--n_fe", type=int, default=20, help="number of candidates proposed for FE verification")
    p.add_argument("--outdir", default="results/search")
    p.add_argument("--plot_only", action="store_true", help="redraw the figure from the saved curves")
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_convergence(pd.read_csv(out / "convergence_curves.csv.gz"), out)
        return

    sur = Surrogate.load(args.model)
    curves, best, xs, fs = [], [], [], []
    for method, search in [("random", random_screening), ("cmaes", cmaes)]:
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            x, f = search(sur, args.budget, np.random.default_rng(seed), args.w, args.s)
            wall = time.perf_counter() - t0
            i = np.argmin(f)
            print(f"{method:6s} seed {seed}: best {f[i]:.3f} mm after {i + 1} evaluations ({wall:.0f} s)")
            curves.append(pd.DataFrame({"method": method, "seed": seed, "n_evals": np.arange(1, len(f) + 1),
                                        "best_f": np.minimum.accumulate(f)}))
            best.append({"method": method, "seed": seed, **dict(zip(COORDS, x[i])), "f_pred_mm": f[i],
                         "evals_to_best": i + 1, "s_per_design": wall / len(f)})
            xs.append(x)
            fs.append(f)

    curves = pd.concat(curves)
    curves.to_csv(out / "convergence_curves.csv.gz", index=False)
    pd.DataFrame(best).to_csv(out / "best_designs.csv", index=False)
    budgets = [10, 100, 1000, 10000, args.budget]
    summary = curves[curves.n_evals.isin(budgets)].groupby(["method", "n_evals"]).best_f.describe()
    summary[["min", "25%", "50%", "75%", "max"]].to_csv(out / "summary.csv")
    print(summary[["min", "25%", "50%", "75%", "max"]])
    plot_convergence(curves, out)

    # FE shortlist from all evaluated designs, with the distance to the nearest training design
    x, f = np.concatenate(xs), np.concatenate(fs)
    idx = shortlist_for_fe(x, f, args.n_fe)
    u_train = (pd.read_csv(args.data)[COORDS].to_numpy() - LOWER) / (UPPER - LOWER)
    u = (x[idx] - LOWER) / (UPPER - LOWER)
    dist = np.linalg.norm(u[:, None] - u_train[None], axis=2).min(axis=1)
    shortlist = pd.DataFrame(x[idx], columns=COORDS).assign(w=args.w, s=args.s, f_pred_mm=f[idx],
                                                            dist_to_training=dist)
    shortlist.to_csv(out / "candidates_for_fe.csv", index=False)


if __name__ == "__main__":
    main()
