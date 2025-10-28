from __future__ import annotations
from typing import Tuple
import numpy as np
from .kernels import H_delta
from ..utils.devices import get_xp

def precompute_H_grid(t_grid, s_grid, kappa, d, Delta) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build H_Δ(t-s) on a grid for reuse. Returns (T, S, H) as numpy arrays.
    If CuPy is available, compute on GPU and bring back to NumPy for caching.
    """
    xp = get_xp()
    t = xp.asarray(t_grid)[:, None]
    s = xp.asarray(s_grid)[None, :]
    ts = t - s
    H = xp.empty_like(ts, dtype=xp.float64)

    # Elementwise compute; you can later fuse with vectorized approximations
    it = np.nditer(np.empty((t.shape[0], s.shape[1]])), flags=['multi_index'])
    for i in range(t.shape[0]):
        for j in range(s.shape[1]):
            H[i, j] = H_delta(float(ts[i, j]), float(kappa), float(d), float(Delta))

    if xp.__name__.startswith("cupy"):
        import cupy as cp
        H = cp.asnumpy(H)
        t = cp.asnumpy(t); s = cp.asnumpy(s)

    return np.squeeze(t), np.squeeze(s), H

def save_grid(path: str, t, s, H):
    np.savez_compressed(path, t=np.asarray(t), s=np.asarray(s), H=np.asarray(H))

def load_grid(path: str):
    z = np.load(path)
    return z["t"], z["s"], z["H"]
