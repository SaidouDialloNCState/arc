from __future__ import annotations
import numpy as np
from ..utils.devices import get_xp

def _h_typeIII_elem(x, kappa, d, xp):
    thresh = (1.0 - d) / kappa
    if x < thresh:
        # Use double-precision floats; mpmath not used here for speed
        # h(x) ≈ x^{d-1} - ((1-d)/kappa)^{d-1}/Γ(d) - kappa^{1-d}/((1-d)^{2-d} Γ(d-1))
        from math import gamma, pow, e
        return pow(x, d - 1.0) - pow((1.0 - d) / kappa, d - 1.0) / gamma(d) \
               - pow(kappa, 1.0 - d) / (pow(1.0 - d, 2.0 - d) * gamma(d - 1.0))
    else:
        from math import gamma, pow, e
        # -(e^{kappa(1-d)}) * e^{-kappa x} / ((1-d)^{2-d} Γ(d-1))
        return - pow(e, kappa * (1.0 - d)) * xp.exp(-kappa * x).item() / (pow(1.0 - d, 2.0 - d) * gamma(d - 1.0))

def h_typeIII(x, kappa, d):
    """Vectorized CPU fallback; x: np.ndarray"""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i, xi in np.ndenumerate(x):
        out[i] = _h_typeIII_elem(float(xi), float(kappa), float(d), np)
    return out

def H_delta_numeric(x, kappa, d, Delta, n=64, xp=None):
    """
    H_Δ(x) ≈ (1/Δ) ∫_0^Δ h(x+u) du via trapezoid with n slices.
    Works on CPU (NumPy) or GPU (CuPy) depending on xp.
    """
    xp = xp or get_xp()
    x = xp.asarray(x, dtype=xp.float64)
    # u grid
    u = xp.linspace(0.0, float(Delta), int(n), dtype=xp.float64)
    # broadcast: (len(x), n)
    X = x[..., None] + u[None, ...]
    # apply h elementwise via Python loop over last axis chunks (keeps code simple)
    # For GPU, this still parallelizes over the large arrays internally.
    if xp.__name__.startswith("cupy"):
        import cupy as cp
        # vectorize via ElementwiseKernel for speed
        hker = cp.ElementwiseKernel(
            in_params='float64 x, float64 kappa, float64 d',
            out_params='float64 y',
            operation=r'''
            double thresh = (1.0 - d) / kappa;
            double gd1 = tgamma(d);
            double gd_1 = tgamma(d - 1.0);
            double one_md = 1.0 - d;
            if (x < thresh){
                y = pow(x, d - 1.0) - pow((1.0 - d)/kappa, d - 1.0)/gd1
                    - pow(kappa, 1.0 - d)/(pow(one_md, 2.0 - d) * gd_1);
            } else {
                y = - exp(kappa * (1.0 - d)) * exp(-kappa * x) / (pow(one_md, 2.0 - d) * gd_1);
            }
            ''',
            name='h_typeIII_ek'
        )
        H = hker(X, float(kappa), float(d))
    else:
        # CPU: call scalar helper via vectorize
        vec = np.vectorize(lambda xx: _h_typeIII_elem(float(xx), float(kappa), float(d), np), otypes=[np.float64])
        H = vec(X)

    # trapezoid along u-axis
    w = xp.ones((int(n),), dtype=xp.float64)
    w[0] = 0.5; w[-1] = 0.5
    integ = (H * w).sum(axis=-1) * (float(Delta) / (n - 1))
    return integ / float(Delta)
