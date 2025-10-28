from __future__ import annotations
from typing import Sequence, List
from mpmath import mp
from .pricer import price_power_call_from_cf
from ..data.market import MarketSet, Quote

def price_all(ms: MarketSet, theta: dict) -> List[mp.mpf]:
    out: List[mp.mpf] = []
    for q in ms.quotes:
        if ms.payoff == "power_call":
            rv_params = dict(
                a=theta["a"], b=theta["b"], c=theta["c"],
                kappa=theta["kappa"], d=theta["d"]
            )
            out.append(+price_power_call_from_cf(q.K, ms.I2_t0, ms.t0, q.T, q.Delta, ms.z_t0, rv_params))
        else:
            raise NotImplementedError(ms.payoff)
    return out
