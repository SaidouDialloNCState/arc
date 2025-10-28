import csv
import argparse
import pathlib
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".data/sample_vix_modeled.csv")
    ap.add_argument("--out", default="plots/fit.png")
    args = ap.parse_args()

    rows = []
    with open(args.input, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    Ks   = [float(r["K"])   for r in rows]
    mids = [float(r["mid"]) for r in rows]
    mods = [float(r.get("model", "nan")) for r in rows]
    errs = [abs(m - float(r["mid"])) for m, r in zip(mods, rows)]

    pathlib.Path("plots").mkdir(exist_ok=True)

    plt.figure()
    plt.scatter(Ks, mids, label="market mid")
    plt.scatter(Ks, mods, marker="x", label="model")
    plt.xlabel("K"); plt.ylabel("price")
    plt.title("Market vs Model (power-type)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print("Saved", args.out)

    plt.figure()
    plt.plot(Ks, errs)
    plt.xlabel("K"); plt.ylabel("|error|")
    plt.title("Absolute error by strike")
    plt.tight_layout()
    out2 = args.out.replace(".png", "_abs_err.png")
    plt.savefig(out2)
    print("Saved", out2)

if __name__ == "__main__":
    main()
