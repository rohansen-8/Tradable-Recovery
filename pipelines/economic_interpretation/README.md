# Out-of-sample attribution and economic-interpretation experiments

These analyses use the analysis panels `data/<h>/df.csv` and the effective-dimension (`eff_dim`) fits.
All analyses need base MATLAB only.

## Code

| Analysis | Code | Entry point |
|---|---|---|
| OOS payoff attribution | `pipelines/empirical_oos_attribution/` | `oos_setup(); run_empirical_oos_attribution(horizons, opts)`, then `make_oos_tables` |
| Ledoit–Wolf Sharpe-difference tests (Table 4.1) | `testing/lw_sharpe_*.m`, `drivers/run_lw_sharpe_inference.m`, `drivers/run_noncrisis_effdim_fits.m` | `run_lw_sharpe_inference('eff_dim', cells)` |
| Experiment 1: forward timing vs. no-look-ahead constant exposure | `pipelines/economic_interpretation/exp1_stageA.m`, `exp1_stageB.m` | `exp1_stageA(h)`, then `exp1_stageB(h)` (no `exp1_stageB` for quarterly) |
| Experiment 2: economic role of the option strip | `pipelines/economic_interpretation/exp2_option_role.m` | `exp2_option_role(h)` |
| Experiment 3: weekly vs. daily nonlinear option exposure on a common calendar | `pipelines/economic_interpretation/exp3_cross_maturity.m` | `exp3_cross_maturity()` |

`h` is one of `'quarterly'`, `'monthly'`, `'weekly'` or `'daily'`.
Add `pipelines/economic_interpretation` to the path, or `cd` into it, before calling the experiment drivers. Each driver locates the repository root from its own file location.
The optional argument `outRoot` redirects outputs, for example to a scratch folder for verification. Its default is `results/economic_interpretation/eff_dim`.
Every writer refuses to overwrite an existing output. OOS windows are checkpointed and fingerprint-checked, and a rerun resumes.

## Run order and dependencies

1. **Saved fits.** The OOS attribution and the Ledoit–Wolf full-sample cells need the `eff_dim` fits in `results/empirical/eff_dim/<h>/<h>_model_state.mat` (`drivers/run_empirical_pipeline`).
2. **OOS attribution** produces `results/empirical_oos/eff_dim/primary/<h>/<h>_oos_components.csv` and `windows/w*.mat`, and the tables in `results/empirical_oos/eff_dim/tables/`. It needs about 900 window fits in total; full runs require a clean Git state. Window fingerprints include a hash of the pipeline code, so the windows are rebuilt whenever that code changes.
3. **Ledoit–Wolf** (`eff_dim` mode) writes `results/inference/performance_sharpe/eff_dim_sb10/`.
   - The non-crisis cells first need `run_noncrisis_effdim_fits`.
4. **Experiment 1**
   - `exp1_stageA` uses the OOS windows, `data/<h>/df.csv` and the OOS schedule functions. It needs no fits.
   - `exp1_stageB` runs the Ledoit–Wolf test with M = 4999, K = 5000, Minner = 499, the SB10 calibration, block grid {1, 2, 4, 6, 8, 10} with b <= T/10, and seeds 2026081501/2026081502.
5. **Experiment 2** uses the OOS components and the OOS variance-swap series `data/return_pairs/oos_<h>_return_pairs.csv`. It needs no fits.
   - Bootstrap: paired circular blocks, B = 9999, seed 2026112201, blocks {1, L_T, 2L_T}.
   - Regressions: Bartlett HAC with lag `bartlett_hac_bandwidth(T)`.
6. **Experiment 3** uses the weekly and daily OOS components and saved window fits. It needs no new fits.
   - Regressions: Bartlett HAC with lag `bartlett_hac_bandwidth(n)`.
   - Inference on c_daily − c_weekly: paired calendar-month block bootstrap, seed 2026112203, blocks of 1 (primary), 2 and 3 months.

## Local inputs (not distributed)

Keep these files at the paths shown. The SHA-256 hashes identify the versions that produced the saved results.

| Path | SHA-256 |
|---|---|
| `data/quarterly/df.csv` | `e06521d9e406dffc0238db2c699271cbaf40c0d0f9823a8e7e8e21d86013fde5` |
| `data/monthly/df.csv` | `3dd5204afc0e4fec35db19abcead9e7e5b25ec59713f1078ccbe9a51fb5c6e8a` |
| `data/weekly/df.csv` | `83585e163218262f8e560515bcda87ded737b6e36fae08559ca909f3d11581c9` |
| `data/daily/df.csv` | `2870bdb07a2f8c5e074cd5fe4d1be2ee08a2d479fe2dbfc36fa09d49e109399e` |

**Variance-swap benchmark series.** These are built with the authors' Python pipeline and are kept locally in `data/return_pairs/` (licensed-derived; not distributed). `run_lw_sharpe_inference`, `run_noncrisis_effdim_fits` and `exp2_option_role` read them.

| File | SHA-256 |
|---|---|
| `full_is_daily_return_pairs.csv` | `8d3080e30b040db925a37ba06adcd26e4e2c58da58cb7e4021ced355c1ce9baf` |
| `full_is_monthly_return_pairs.csv` | `0c14e5b50a41d9d658c16e86596a3a41a5ab87f16589e6c79c345d6f5616140e` |
| `full_is_quarterly_return_pairs.csv` | `3cfbcdf807c9e3ffa30236a08484eda1e8cc33ac4e9167c2cb95fbed55afa1a0` |
| `full_is_weekly_return_pairs.csv` | `528457ec5ac0e04e49b2bb1263a93116105a5dbd838b598d6a20e7347cfbf4e7` |
| `noncrisis_is_daily_return_pairs.csv` | `08f0d393915b33f6b17f53c6e1fce53f5903e879d72d8b41aaeb6e397cdf5076` |
| `noncrisis_is_monthly_return_pairs.csv` | `896c1d5b40b8f8162f5a046e8bc7f92d1682dfc042d787e114e00bc5f9acccd7` |
| `noncrisis_is_quarterly_return_pairs.csv` | `d03a72d3d1e76d99763ab5b2c384e70653a1fd26980fcd6fa509a07ff0f9af26` |
| `noncrisis_is_weekly_return_pairs.csv` | `9401d5684762f08b09c768fca1d72b2717601658954c2aabbe67417b84b2f73e` |
| `oos_daily_return_pairs.csv` | `e439fa240abfb52d5c634de19ef8a79a5566ba44b78bdc48f82b461e2c8261cf` |
| `oos_monthly_return_pairs.csv` | `f79a89eae039897aca61f6236594d180f163ffcb2f6a8f43e7d5d166b8e9f9c3` |
| `oos_weekly_return_pairs.csv` | `21e14c15e22d0547479a0af30edc8d36e057f0729111c8f5eb63e94ca2cf84f7` |

**Reference Ledoit–Wolf bootstrap realizations.** These are in `tests/fixtures/ledoit_wolf_cv/`: 11 `.mat` files plus a README, produced by the authors' Python Ledoit–Wolf implementation (not distributed). Only `run_lw_sharpe_inference('validate_cv')` needs them; it checks the MATLAB implementation against them.

| File | SHA-256 |
|---|---|
| `full_is_daily.mat` | `af23690c614ed00885defda6295774a160e932a498b263b82f3e0480400e4aa8` |
| `full_is_monthly.mat` | `b4284bec15fede36d65e0ec434a1e2bea0d5ba2359619bd13de70ab7aad0aae5` |
| `full_is_quarterly.mat` | `096fff4ac55c9eca6909444787043a24c89b9fd6057278f7e168a19c0617866c` |
| `full_is_weekly.mat` | `7116a1d6ec38b58f4d1f1c65125673edcb1b3dea408e0e2c25d42700dd23d529` |
| `noncrisis_is_daily.mat` | `04700906dc846de4aaa6752e13c52e399c4668a12043656bee5d715d3e9f5a4d` |
| `noncrisis_is_monthly.mat` | `e221e2deef85b42e9f07593effa6422dd13c231b3977c6474a2e5076e70fb1ec` |
| `noncrisis_is_quarterly.mat` | `38a4bce53c7963bac0d52f76a0504e7ed32ab6c2868f487d3d7973b5879b2dc4` |
| `noncrisis_is_weekly.mat` | `7756b7dcdf92fc2fcd3a378037702323e48482fb13d9ca02cb6b22497b189e5c` |
| `oos_daily.mat` | `f5cb6334ac076f5ba5494462d0b3a54b135b6b7826d7ae8e869414a55a30f99c` |
| `oos_monthly.mat` | `a8bc78820e6c9506bea0b365c22fc4b79970d9410c8d51352e7ef32da4e8485b` |
| `oos_weekly.mat` | `e330ab0e5b3dafaa22e9a23253eaab290d61338bd237532809cb5e9a37fcf7ee` |

## Verification

```matlab
cd pipelines/economic_interpretation
exp1_stageA('monthly', tempdir); exp2_option_role('monthly', tempdir)   % compare CSVs with results/economic_interpretation/eff_dim
```

`pipelines/empirical_oos_attribution/tests/test_oos_smoke(scratchDir)` checks the OOS attribution pipeline with about 10 fits.
