from __future__ import annotations
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from mpmath import mp
from ..data.market import MarketSet, Quote
from .pricer import price_power_call_from_cf

def _worker_price_one(args: Tuple[Quote, Dict[str, mp.mpf], mp.mpf, mp.mpf, mp.mpf, bool, int]) -> float:
    q, theta, I2_t0, t0, z0, use_fast_H, dps = args
    mp.mp.dps = dps
    rv_params = dict(a=theta["a"], b=theta["b"], c=theta["c"], kappa=theta["kappa"], d=theta["d"], use_fast_H=use_fast_H)
    p = price_power_call_from_cf(q.K, I2_t0, t0, q.T, q.Delta, z0, rv_params)
    return float(p)

def prices_for_market_parallel(ms: MarketSet, theta: Dict[str, mp.mpf], use_fast_H: bool = False,
                               nprocs: int = 0, dps: int = 80) -> List[float]:
    """
    Parallel price all quotes in a MarketSet using separate processes.
    nprocs=0/None -> uses ProcessPool default (#CPU cores).
    """
    tasks = [(q, theta, ms.I2_t0, ms.t0, ms.z_t0, use_fast_H, dps) for q in ms.quotes]
    results: List[float] = [0.0] * len(tasks)
    with ProcessPoolExecutor(max_workers=(None if not nprocs else nprocs)) as ex:
        futs = {ex.submit(_worker_price_one, (tasks[i])): i for i in range(len(tasks))}
        for f in as_completed(futs):
            idx = futs[f]
            results[idx] = f.result()
    return results
