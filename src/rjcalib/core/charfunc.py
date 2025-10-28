
from __future__ import annotations
import os
from mpmath import mp, quad

# Precision: use helper if available
try:
    from ..utils.settings import set_precision_from_env
    set_precision_from_env(80)
except Exception:
    mp.dps = 80

# Base H_delta (reference)
from .kernels import H_delta

# Optional fast numeric H_delta (CPU/GPU)
try:
    from .hdelta_gpu import H_delta_numeric as H_delta_fast
except Exception:
    H_delta_fast = None

# Optional cached reference H_delta
try:
    from .hdelta_cache import H_delta_cached as H_delta_cached_ref
except Exception:
    H_delta_cached_ref = None

def phi_X1_exponent(l, a, b, c):
    """
    log phi_{X1}(l) = a * Gamma(-c) * ( (b - i l)^c - b^c )
    """
    return a * mp.gamma(-c) * (mp.power(b - 1j*l, c) - mp.power(b, c))

def xi1(a, b, c):
    """
    E[X1] = a * Gamma(1-c) / b^(1-c)
    """
    return a * mp.gamma(1 - c) / mp.power(b, 1 - c)

def Phi_I2_conditional(l, I2_t0, t0, t, Delta, z_t0, rv_params):
    """
    Structure-preserving conditional CF for I^2.
    Integrates H_delta over s in [t0, t]; keeps the original transform.

    rv_params: a, b, c, kappa, d
    toggles:
      - use_fast_H (bool) or env RJ_USE_FAST_HDELTA=1
      - cache_H   (bool) or env RJ_CACHE_HDELTA=1
    """
    a = rv_params["a"]; b = rv_params["b"]; c = rv_params["c"]
    kappa = rv_params["kappa"]; d = rv_params["d"]

    use_fast  = bool(rv_params.get("use_fast_H", False)) or os.environ.get("RJ_USE_FAST_HDELTA","0") in {"1","true","True"}
    use_cache = bool(rv_params.get("cache_H", False))    or os.environ.get("RJ_CACHE_HDELTA","0")    in {"1","true","True"}

    Hbase = (H_delta_fast if (use_fast and H_delta_fast is not None) else H_delta)
    Hfun  = (Hbase if not (use_cache and H_delta_cached_ref is not None) else H_delta_cached_ref)

    def H_of(delta_ts):
        # Ensure numeric scalar in/out for mpmath.quad
        x = float(delta_ts)  # mpf -> float for NumPy/CuPy paths
        val = Hfun(x, kappa, d, Delta)
        if hasattr(val, "item"):
            val = val.item()
        return mp.mpf(val)

    # Integral 1: int_{t0}^{t} H_delta(t-s) ds
    H_int = quad(lambda s: H_of(t - s), [t0, t])

    # Integral 2: int_{t0}^{t} log phi_{X1}( l * H_delta(t-s) ) ds
    log_phi_int = quad(lambda s: phi_X1_exponent(l * H_of(t - s), a, b, c), [t0, t])

    # Drift/remainder (simple, numerically stable form)
    J = I2_t0 - xi1(a, b, c) * H_int

    return mp.e**(1j*l*J + log_phi_int)
