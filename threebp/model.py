"""Fully connected surrogate"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from .data import FEATURES


class SurrogateNet(nn.Module):
    def __init__(self, n_in=len(FEATURES), hidden=(512, 256, 128, 64), p_drop=0.2):
        super().__init__()
        layers, n = [], n_in
        for h in hidden:
            layers += [nn.Linear(n, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            n = h
        layers.append(nn.Linear(n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Surrogate:

    def __init__(self, net, scaler):
        self.net = net.eval()
        self.scaler = scaler

    @classmethod
    def load(cls, model_dir):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        net = SurrogateNet(hidden=tuple(cfg["hidden"]), p_drop=cfg["dropout"])
        net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
        return cls(net, joblib.load(model_dir / "scaler.joblib"))

    @torch.no_grad()
    def predict(self, X, batch_size=65536):
        X = self.scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        out = [self.net(torch.from_numpy(X[i:i + batch_size])).numpy().ravel()
               for i in range(0, len(X), batch_size)]
        return np.concatenate(out) if out else np.empty(0)
