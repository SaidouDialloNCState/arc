param(
  [string]$Config = ".configs/toy.json"
)
$env:RJ_USE_FAST_HDELTA="1"
python -m rjcalib.calib.run --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/plot_fit.ps1
