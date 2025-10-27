# Rough Volatility with Tempered-Stable Jumps — Fast Calibration

Structure-preserving CF pricer + GPU precompute + tiny NN surrogate.


### Fast vs Reference
- **Reference (default):** mpmath + closed-form \(H_\Delta\) for maximal fidelity.
- **Fast path:** numeric \(H_\Delta\) with CuPy/NumPy via `--use_fast_H` (or env `RJ_USE_FAST_HDELTA=1`).
  Use fast for calibration sweeps; verify final params with the reference path.

