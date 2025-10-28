import sys, pathlib, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, str(pathlib.Path("src").resolve()))
from rjcalib.data.market import MarketSet
from rjcalib.calib.objective import vec_to_theta
from rjcalib.pricing.pricer import price_power_call_from_cf
from rjcalib.nn.surrogate import MLP, make_features

def generate_dataset(csv_path, I2, z0, n_samples=200, seed=42):
    rng = random.Random(seed)
    ms = MarketSet.from_csv(csv_path, I2_t0=I2, z_t0=z0)

    X_list, y_list = [], []
    for _ in range(n_samples):
        # sample theta in bounds similar to optimizer
        a = 10**rng.uniform(-3.5, 0.7)
        b = rng.uniform(0.1, 6.0)
        c = rng.uniform(0.1, 0.9)
        kappa = rng.uniform(0.02, 3.0)
        d = rng.uniform(0.56, 0.94)
        theta = vec_to_theta([a,b,c,kappa,d])

        for q in ms.quotes:
            y = float(price_power_call_from_cf(q.K, ms.I2_t0, ms.t0, q.T, q.Delta, ms.z_t0,
                                               dict(a=theta["a"], b=theta["b"], c=theta["c"], kappa=theta["kappa"], d=theta["d"])))
            X_list.append([float(q.K), float(q.T), float(q.Delta), float(I2), float(z0), a,b,c,kappa,d])
            y_list.append(y)

    X = make_features(*np.array(X_list).T)
    y = torch.tensor(np.array(y_list, dtype=np.float32))
    return X, y

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=".data/sample_vix.csv")
    p.add_argument("--I2", type=float, default=0.1)
    p.add_argument("--z0", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--out", default="surrogate.pt")
    args = p.parse_args()

    X, y = generate_dataset(args.csv, args.I2, args.z0, n_samples=args.samples)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=False)

    model = MLP(d_in=X.shape[1], d_hidden=256, d_out=1, n_layers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
        print(f"epoch {ep+1}/{args.epochs} - train MSE {tot/len(ds):.6g}")

    torch.save(model.state_dict(), args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
