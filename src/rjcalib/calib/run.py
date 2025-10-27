from __future__ import annotations
import argparse, json, pathlib
from mpmath import mp
from .optimize import global_then_local, DEFAULT_BOUNDS
from ..data.market import MarketSet

def main():
    p = argparse.ArgumentParser(description="Rough-jump VIX calibration (structure-preserving CF).")
    p.add_argument("--csv", required=True, help="Path to market CSV with K,T,Delta,mid[,bid,ask,type]")
    p.add_argument("--I2_t0", type=float, required=True, help="Initial integrated variance proxy (I2_t0)")
    p.add_argument("--z_t0", type=float, default=0.0, help="z_t0 state (default 0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maxiter_global", type=int, default=200)
    p.add_argument("--maxiter_local", type=int, default=200)
    p.add_argument("--out", default="calib_result.json")
    p.add_argument("--dump_prices", default=None, help="Optional path to write model prices for the CSV")
    args = p.parse_args()

    mp.mp.dps = 80

    ms = MarketSet.from_csv(args.csv, I2_t0=args.I2_t0, z_t0=args.z_t0, payoff="power_call")

    res = global_then_local(ms, bounds=DEFAULT_BOUNDS,
                            seed=args.seed,
                            maxiter_global=args.maxiter_global,
                            maxiter_local=args.maxiter_local)

    out = dict(
        x = res.x.tolist(),
        fun = res.fun,
        nit = res.nit,
        nfev = res.nfev,
        success = res.success,
        message = res.message,
        bounds = DEFAULT_BOUNDS
    )
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print("Saved", args.out)
    print(json.dumps(out, indent=2))
    if args.dump_prices:
        from ..calib.objective import vec_to_theta
        from ..pricing.batch import price_all
        theta = vec_to_theta(out["x"])
        ms2 = MarketSet.from_csv(args.csv, I2_t0=args.I2_t0, z_t0=args.z_t0, payoff="power_call")
        prices = [float(p) for p in price_all(ms2, theta)]
        import csv
        rows = []
        import pandas as pd
        with open(args.csv, newline="") as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        for r, pv in zip(rows, prices):
            r["model"] = f"{pv:.10g}"
            r["abs_err"] = f"{abs(pv - float(r["mid"])):.10g}"
        import json as _json
        import pathlib as _pl
        # write CSV with model column
        outp = _pl.Path(args.dump_prices)
        with outp.open("w", newline="") as wf:
            w = csv.DictWriter(wf, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("Wrote model prices to", str(outp))

if __name__ == "__main__":
    main()
