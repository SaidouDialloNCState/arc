from __future__ import annotations
from mpmath import mp
from tqdm import tqdm
from .objective import price_quote
from ..data.market import MarketSet

def loss_l2_batched(ms: MarketSet, theta: dict, show_tqdm: bool = True) -> mp.mpf:
    err = mp.mpf("0.0")
    it = tqdm(ms.quotes, desc="pricing", leave=False) if show_tqdm else ms.quotes
    for q in it:
        model = price_quote(q, ms, theta)
        if q.bid is not None and q.ask is not None and q.ask > q.bid:
            w = 1 / mp.max(mp.mpf("0.05"), q.ask - q.bid)
        else:
            w = mp.mpf("1.0")
        e = (model - q.mid)
        err += w * (e * e)
    return err
