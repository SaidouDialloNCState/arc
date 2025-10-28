from __future__ import annotations
from typing import Dict, List
from mpmath import mp
from .pricer import price_power_call_from_cf
from ..data.market import MarketSet

def prices_for_market(ms: MarketSet, theta: Dict[str, mp.mpf], use_fast_H: bool = False) -> List[mp.mpf]:
    out: List[mp.mpf] = []
    for q in ms.quotes:
        rv_params = dict(
            a=theta["a"], b=theta["b"], c=theta["c"], kappa=theta["kappa"], d=theta["d"],
            use_fast_H=use_fast_H
        )
        p = price_power_call_from_cf(q.K, ms.I2_t0, ms.t0, q.T, q.Delta, ms.z_t0, rv_params)
        out.append(+p)
    return out
