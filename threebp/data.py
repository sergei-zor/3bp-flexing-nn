import numpy as np
import pandas as pd

GEOMETRY = ["x1", "y3", "y4", "y7", "y8", "w"]
FEATURES = GEOMETRY + ["s", "D"]
TARGET = "v_e"

D_GRID = np.round(np.arange(0.2, 10.0 + 1e-9, 0.2), 2)

def load_long(path):
    wide = pd.read_csv(path)
    v_cols = [c for c in wide.columns if c.startswith("v0(V=")]
    long = wide.melt(id_vars=GEOMETRY + ["s"], value_vars=v_cols, var_name="D", value_name=TARGET,
                     ignore_index=False).rename_axis("run_id").reset_index()
    long["D"] = long["D"].str.extract(r"V=([\d.]+)", expand=False).astype(float)
    long = long[(long["D"] > 0) & long[TARGET].notna()]  
    return long.sort_values(["run_id", "D"]).reset_index(drop=True)


def test_mask(n, test_frac=0.2, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    test = np.zeros(n, dtype=bool)
    test[perm[int(round((1 - test_frac) * n)):]] = True
    return test
