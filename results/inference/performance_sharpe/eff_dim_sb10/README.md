# Ledoit–Wolf Sharpe-difference tests, effective-dimension returns (Table 4.1)

Produced by `drivers/run_lw_sharpe_inference('eff_dim', cells)`.

## Procedure

- **HAC standard error:** prewhitened quadratic-spectral (QS) kernel.
- **Final bootstrap:** M = 4999, seed 2026081502.
- **Block-length calibration:** Algorithm 3.1, with K = 5000 and Minner = 499. The pseudo data come from a stationary bootstrap of the paired returns with mean block 10. Seed 2026081501.
- **Block grid:** {1, 2, 4, 6, 8, 10}, restricted to b ≤ T/10.
- **Calibration check:** see `../validate_cv_sb10/`.

## Inputs

| Cells | Strategy returns | Variance-swap returns |
|---|---|---|
| `full_is_<h>` | alpha'h_star of `results/empirical/eff_dim/<h>/<h>_model_state.mat` | `data/return_pairs/` (Python pipeline) |
| `noncrisis_is_<h>` | `results/empirical_noncrisis/eff_dim/<h>/<h>_noncrisis_returns.csv`, from `drivers/run_noncrisis_effdim_fits.m` | same source |
| `oos_<h>` | `results/empirical_oos/eff_dim/primary/<h>/<h>_oos_components.csv` (total) | same source |

The variance-swap series do not depend on the estimator.

## Files

- **Full-sample and OOS cells:** `summary_20261121_165112.csv`.
- **Non-crisis cells:** `summary_20261121_171411.csv`.
- **Block-length sensitivity:** `block_sensitivity_p_one_sided*.csv`, the one-sided p at every candidate block, using the same final-bootstrap seed.
- **Comparison with the CV-based estimates:** `comparison_effdim_vs_cv.csv` (full-sample and OOS cells).
