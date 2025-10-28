from __future__ import annotations
from functools import lru_cache
from .kernels import H_delta as H_ref

def _key(x, kappa, d, Delta, ndp=6):
    r = lambda z: round(float(z), ndp)
    return (r(x), r(kappa), r(d), r(Delta), ndp)

@lru_cache(maxsize=200_000)
def H_delta_cached_key(key):
    x,kappa,d,Delta,_ = key
    return float(H_ref(x, kappa, d, Delta))

def H_delta_cached(x, kappa, d, Delta, ndp=6):
    return H_delta_cached_key(_key(x,kappa,d,Delta,ndp))
