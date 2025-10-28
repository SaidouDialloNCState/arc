import csv, math, statistics as stats
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def load_rows(path):
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    # ensure model col exists
    if not rows or "model" not in rows[0]:
        raise ValueError("Input CSV must include a 'model' column (use --dump_prices).")
    return rows

def summarize(rows):
    errs = []
    ks = []
    Ts = []
    for r in rows:
        m = float(r["model"])
        y = float(r["mid"])
        k = float(r["K"]); t = float(r["T"])
        ks.append(k); Ts.append(t)
        errs.append(m - y)
    abs_err = [abs(e) for e in errs]
    mae = sum(abs_err)/len(abs_err)
    rmse = math.sqrt(sum(e*e for e in errs)/len(errs))
    mape = 100.0 * sum(abs(e/(float(r["mid"])+1e-12)) for e,r in zip(errs, rows))/len(rows)
    summary = dict(
        n=len(rows),
        mae=mae, rmse=rmse, mape=mape,
        mean_abs=stats.mean(abs_err), median_abs=stats.median(abs_err),
        max_abs=max(abs_err)
    )
    return summary, ks, Ts, errs, abs_err

def write_summary_csv(out_dir, summary):
    outp = Path(out_dir)/"calib_summary.csv"
    with outp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        for k,v in summary.items():
            w.writerow([k, v])
    return outp

def write_report_md(out_dir, summary, input_csv):
    p = Path(out_dir)/"calib_report.md"
    p.write_text(f"""# Calibration Report

**Input CSV:** `{input_csv}`

## Summary
- n = {summary['n']}
- MAE = {summary['mae']:.6g}
- RMSE = {summary['rmse']:.6g}
- MAPE (%) = {summary['mape']:.4f}
- Mean |err| = {summary['mean_abs']:.6g}
- Median |err| = {summary['median_abs']:.6g}
- Max |err| = {summary['max_abs']:.6g}

See plots in `plots/`.
""")
    return p

def make_plots(rows, ks, Ts, errs, abs_err, out_dir):
    out_plots = Path(out_dir)/"plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    # abs error histogram
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(abs_err, bins=20)
    plt.xlabel("|error|"); plt.ylabel("count")
    plt.title("Absolute Error Histogram")
    plt.tight_layout()
    p1 = out_plots/"abs_error_hist.png"
    plt.savefig(p1); print("Saved", p1)

    # error vs K
    plt.figure()
    plt.scatter(ks, abs_err)
    plt.xlabel("K"); plt.ylabel("|error|")
    plt.title("Absolute Error vs Strike")
    plt.tight_layout()
    p2 = out_plots/"abs_error_vs_K.png"
    plt.savefig(p2); print("Saved", p2)

    # error vs T
    plt.figure()
    plt.scatter(Ts, abs_err)
    plt.xlabel("T"); plt.ylabel("|error|")
    plt.title("Absolute Error vs Maturity")
    plt.tight_layout()
    p3 = out_plots/"abs_error_vs_T.png"
    plt.savefig(p3); print("Saved", p3)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with market + model cols (from --dump_prices)")
    ap.add_argument("--outdir", default="report")
    args = ap.parse_args()

    rows = load_rows(args.input)
    summary, ks, Ts, errs, abs_err = summarize(rows)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    scsv = write_summary_csv(args.outdir, summary)
    rmd = write_report_md(args.outdir, summary, args.input)
    make_plots(rows, ks, Ts, errs, abs_err, args.outdir)
    print("Summary CSV:", scsv)
    print("Report MD:", rmd)

if __name__ == "__main__":
    main()
