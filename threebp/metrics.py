import numpy as np


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "n": len(y_true),
        "R": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "MAE_mm": float(np.mean(np.abs(err))),
        "RMSE_mm": float(np.sqrt(np.mean(err ** 2))),
    }
