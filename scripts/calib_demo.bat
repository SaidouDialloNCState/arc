@echo off
setlocal
set RJ_USE_FAST_HDELTA=1
python -m rjcalib.calib.run --csv .data\sample_vix.csv --I2_t0 0.1 --z_t0 0.0 --maxiter_global 30 --maxiter_local 80 --use_fast_H --out calib_result.json --dump_prices .data\sample_vix_modeled.csv
python scripts\plot_fit.py
endlocal
