from __future__ import annotations
from typing import Dict, Sequence, Tuple
from mpmath import mp
from ..pricing.pricer import price_power_call_from_cf
from ..data.market import MarketSet, Quote

ParamVec = Tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf]  # (a,b,c,kappa,d)

def price_quote(q: Quote, ms: MarketSet, theta: Dict[str, mp.mpf]) -> mp.mpf:
    if ms.payoff == "power_call":
        rv_params = dict(
            a=theta["a"], b=theta["b"], c=theta["c"],
            kappa=theta["kappa"], d=theta["d"]
        )
        p = price_power_call_from_cf(q.K, ms.I2_t0, ms.t0, q.T, q.Delta, ms.z_t0, rv_params)
        # If CSV marks puts, you can add put-call parity for the specific payoff here later
        return +p
    raise NotImplementedError(ms.payoff)

def quote_weight(q: Quote) -> mp.mpf:
    # Inverse-width weight: tighter markets weigh more, floor to avoid blowups
    if q.bid is not None and q.ask is not None and q.ask > q.bid:
        w = 1 / mp.max(mp.mpf("0.05"), q.ask - q.bid)
    else:
        w = mp.mpf("1.0")
    # Optional: emphasize at-the-money region
    return w

def loss_l2(ms: MarketSet, theta: Dict[str, mp.mpf]) -> mp.mpf:
    err = mp.mpf("0.0")
    for q in ms.quotes:
        model = price_quote(q, ms, theta)
        w = quote_weight(q)
        e = (model - q.mid)
        err += w * (e * e)
    return err

def vec_to_theta(x: Sequence[float]) -> Dict[str, mp.mpf]:
    a,b,c,kappa,d = x
    return dict(a=mp.mpf(a), b=mp.mpf(b), c=mp.mpf(c), kappa=mp.mpf(kappa), d=mp.mpf(d))
