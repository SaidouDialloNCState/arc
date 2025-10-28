param(
  [string]$Input = ".data/sample_vix_modeled.csv",
  [string]$Out = "plots/fit.png"
)
python scripts/plot_fit.py --input $Input --out $Out
