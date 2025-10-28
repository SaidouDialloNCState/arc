from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Iterable, Optional
import csv
from mpmath import mp

Payoff = Literal["power_call"]  # extend later if needed

@dataclass
class Quote:
    K: mp.mpf
    T: mp.mpf         # time to maturity (same units as model t)
    Delta: mp.mpf     # window length for I^2 aggregation
    mid: mp.mpf
    bid: Optional[mp.mpf] = None
    ask: Optional[mp.mpf] = None
    typ: Literal["C","P"] = "C"     # VIX options are quoted as calls/puts

@dataclass
class MarketSet:
    quotes: List[Quote]
    I2_t0: mp.mpf     # initial integrated variance proxy
    z_t0: mp.mpf      # state variable from paper (can set 0 if unknown)
    t0: mp.mpf = mp.mpf("0.0")
    payoff: Payoff = "power_call"

    @classmethod
    def from_csv(cls, path: str, I2_t0: float, z_t0: float = 0.0, payoff: Payoff = "power_call") -> "MarketSet":
        out: List[Quote] = []
        with open(path, newline="") as f:
            rdr = csv.DictReader(f)
            needed = {"K","T","Delta","mid"}
            missing = needed - set(rdr.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing columns: {missing}. Required: {needed}.")
            for r in rdr:
                out.append(Quote(
                    K=mp.mpf(r["K"]),
                    T=mp.mpf(r["T"]),
                    Delta=mp.mpf(r["Delta"]),
                    mid=mp.mpf(r["mid"]),
                    bid=mp.mpf(r["bid"]) if r.get("bid") else None,
                    ask=mp.mpf(r["ask"]) if r.get("ask") else None,
                    typ=(r.get("type") or r.get("typ") or "C").strip().upper()[0]
                ))
        return cls(out, mp.mpf(I2_t0), mp.mpf(z_t0), mp.mpf("0.0"), payoff)
