from __future__ import annotations
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in=9, d_hidden=128, d_out=1, n_layers=3):
        super().__init__()
        layers = []
        last = d_in
        for i in range(n_layers):
            layers += [nn.Linear(last, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden)]
            last = d_hidden
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_features(K, T, Delta, I2, z0, a,b,c,kappa,d):
    import numpy as np
    x = np.stack([K, T, Delta, I2, z0, a,b,c,kappa,d], axis=-1)  # (N,10)
    # simple invariants: log/scale
    x[:,0] = np.log(np.maximum(x[:,0], 1e-8))
    x[:,1:3] = np.log(np.maximum(x[:,1:3], 1e-8))
    x[:,3:5] = x[:,3:5]  # leave as is
    return torch.tensor(x[:,[0,1,2,3,4,5,6,7,8,9]], dtype=torch.float32)
