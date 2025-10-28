from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional
import numpy as np
from mpmath import mp
from scipy.optimize import differential_evolution, minimize
from .objective import vec_to_theta
from .batchloss import loss_l2_batched as loss_l2
from ..data.market import MarketSet

@dataclass
class CalibResult:
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str

DEFAULT_BOUNDS = [
    (1e-4, 5.0),   # a
    (0.10, 10.0),  # b (tempering)
    (0.05, 0.95),  # c (stability index in (0,1))
    (0.01, 5.0),   # kappa (mean reversion)
    (0.55, 0.95),  # d in (0.5,1)
]

def _mp_loss(x: Sequence[float], ms: MarketSet) -> float:
    theta = vec_to_theta(x)
    val = loss_l2(ms, theta)
    return float(val)

def global_then_local(ms: MarketSet,
                      bounds = DEFAULT_BOUNDS,
                      seed: int = 42,
                      maxiter_global: int = 200,
                      popsize: int = 15,
                      maxiter_local: int = 200) -> CalibResult:
    # Global phase
    de = differential_evolution(lambda v: _mp_loss(v, ms),
                                bounds=bounds, seed=seed, maxiter=maxiter_global,
                                popsize=popsize, polish=False, updating="deferred")
    x0 = de.x

    # Local refine
    lbfgs = minimize(lambda v: _mp_loss(v, ms),
                     x0=x0, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=maxiter_local, ftol=1e-9))
    res = lbfgs

    return CalibResult(
        x = res.x,
        fun = float(res.fun),
        nit = res.nit,
        nfev = res.nfev,
        success = bool(res.success),
        message = str(res.message)
    )
