"""
python train.py --outdir models/surrogate
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from threebp.data import FEATURES, TARGET, load_long, test_mask
from threebp.model import SurrogateNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/fe_dataset_training.csv")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="weight initialization and batch order")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=7.5e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128, 64])
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def train(net, X_train, y_train, X_test, y_test, args):
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        train_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb) / len(X_train)
        net.eval()
        with torch.no_grad():
            test_loss = loss_fn(net(X_test), y_test).item()
        history.append({"epoch": epoch, "train_mse": train_loss, "test_mse": test_loss})
        if epoch % 10 == 0:
            print(f"epoch {epoch:4d}  train MSE {train_loss:.3f}  test MSE {test_loss:.3f}", flush=True)
    return pd.DataFrame(history)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    long = load_long(args.data)
    test = test_mask(len(long), args.test_frac, args.split_seed)
    X, y = long[FEATURES].to_numpy(), long[[TARGET]].to_numpy()
    scaler = StandardScaler().fit(X[~test])
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    print(f"{len(long)} points, {(~test).sum()} for training and {test.sum()} for testing")

    net = SurrogateNet(hidden=args.hidden, p_drop=args.dropout)
    t0 = time.time()
    history = train(net, tensor(scaler.transform(X[~test])), tensor(y[~test]),
                    tensor(scaler.transform(X[test])), tensor(y[test]), args)
    train_time = time.time() - t0
    print(f"done in {train_time / 60:.1f} min")

    torch.save(net.state_dict(), out / "model.pt")
    joblib.dump(scaler, out / "scaler.joblib")
    history.to_csv(out / "history.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {"train_time_s": train_time}, indent=2))


if __name__ == "__main__":
    main()
