# Tradable Recovery: replication code

MATLAB code for *Tradable Recovery*
the tensor-product RKHS projection estimator of the S&P 500 SDF (Matern 7/2 in forward
moneyness x Laplace in VIX), the effective-dimension penalty rule, regional shape tests
(monotonicity, convexity, concavity) with HAC Monte Carlo inference, out-of-sample portfolio
attribution, the economic-interpretation experiments, and the Black-Scholes (BS) and CHJ
simulation studies.

The design parameters are defined in two configuration files, which
`tests/verify_frozen_design.m` checks against this document:
`configs/get_frozen_design.m` (global design parameters) and
`configs/get_empirical_maturity_config.m` (per-maturity domains, regions and figure ranges).

## Requirements

- MATLAB R2026b with the Statistics and Machine Learning Toolbox (simulations and
  `verify_simulation_design` use `normcdf`). The Parallel Computing Toolbox is optional
  (12 process workers for simulations); results do not depend on it. The empirical, OOS,
  Ledoit-Wolf and economic-interpretation code needs base MATLAB only.
- Python 3 with `wrds` and `pandas`, only for the optional programmatic data download
  (`pipelines/data_construction/wrds_acquire.py`).

## Repository layout

| Directory | Purpose |
| --- | --- |
| `configs/` | Design parameters and per-experiment configurations. |
| `drivers/` | Runners: empirical pipeline, simulations, Ledoit-Wolf inference, SDF recovery. |
| `training/`, `testing/`, `evaluation/`, `kernels/` | Estimator, shape tests and HAC inference, evaluation, kernels. |
| `simulators/` | Black-Scholes (BS) and CHJ option-panel simulators. |
| `plotting/` | Figure and table renderers that only read saved results. |
| `pipelines/data_construction/` | WRDS download (optional) and reconstruction of forwards and option panels. |
| `pipelines/empirical_oos_attribution/` | Expanding-window out-of-sample fits and payoff attribution. |
| `pipelines/economic_interpretation/` | Experiments 1-3 (forward timing, option-strip role, cross-maturity); see its README. |
| `tests/` | Design verification and unit tests of the estimator, inference, CV solvers and simulation designs. |
| `run_replication.m` | Staged replication entry point. |

`data/` and `results/` hold local inputs and generated outputs. Only aggregate result
tables (CSV/TeX), publication figures and READMEs under `results/` are tracked; option panels,
derived returns, `.mat` checkpoints and per-replication files are not (see `.gitignore`,
`results/README.md`).

## Data

The option data are OptionMetrics IvyDB US files licensed through WRDS and cannot be
redistributed; neither raw files nor the derived panels in `data/` are included.
`data/SHA256SUMS` identifies the analysis panels behind the reported results.

Raw inputs (SPX, secid 108105) are downloaded from WRDS web queries as ZIP files into
`data_reconstructed/raw/` under the names in `pipelines/data_construction/recon_config.m`
(`c.zip`). Alternatively, `wrds_acquire.py` downloads the same tables as CSV files into
`data_reconstructed/raw/wrds/` (set `WRDS_USER`; the password is read by libpq from the user's
own `~/.pgpass`):

| Dataset | Columns used |
| --- | --- |
| Option prices (`opprcd`) | secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, volume, open_interest, impl_volatility, delta, gamma, vega, theta, optionid, am_settlement, contract_size, forward_price, index_flag, exercise_style |
| Forward prices (`fwdprd`) | secid, date, expiration, AMSettlement, ForwardPrice |
| Zero curve (`zerocd`) | date, days, rate |
| Security prices (`secprd`) | secid, date, close, return |
| Cboe VIX | DATE, OPEN, HIGH, LOW, CLOSE |

`run_forward_reconstruction` and then `run_stage1_reconstruction` rebuild the forwards and the
four maturity panels into `data_reconstructed/` (about 3 minutes; outputs are never
overwritten). The analysis panels `data/<h>/df.csv` (columns `date`, `frwd_moneyness`, `VIX`,
`OPT_NET_R_PAYOFF`, `weight`, `SP500`) and the variance-swap benchmark series in
`data/return_pairs/` were built with the authors' Python data pipeline.

**Reproducibility notes.**
1. *Panel reconstruction.* Rebuilding the panels from the raw WRDS files reproduces the
   analysis panels exactly through 2022. For 2023-2025 the rebuilt forward prices, and hence the
   panels, differ. All results use the analysis panels identified by `data/SHA256SUMS`.
2. *Weekly SDF validation sample.* The manuscript reports N = 600 for the weekly SDF
   validation regressions (Python, category P below); the current validation output has N = 634.

## Replication

```matlab
V = run_replication();                         % regenerate the empirical results and verify them
V = run_replication({'empirical','verify'});   % selected stages
V = run_replication({'cv','lw'});              % optional stages (CV robustness, Ledoit-Wolf)
```

`run_replication` recomputes every empirical stage from `data/` into `results_replication/`
(untracked; redirected through `SDF_RESULTS_ROOT`, so each stage reads what the preceding stages
produced) and then compares every regenerated CSV with its counterpart in `results/`
(numeric columns, relative tolerance 1e-8; report in `results_replication/verification.csv`).
The `results/` folder is only read. An interrupted run resumes: fits, shape-test
checkpoints and OOS windows are fingerprint-checked and reused within `results_replication/`.
External inputs are checked against `data/SHA256SUMS` first.

**Verification (R2026b, 12 thread workers).** All 75 regenerated CSVs agree
with `results/`: fits, penalty parameters, OOS returns and tables, non-crisis fits, the effective-dimension
target table and Experiments 1-3 to 1e-11 or better. Shape-test p-values and critical values
reproduce up to Monte Carlo error: the HAC score covariance is singular, so the draws use an
eigen-factor whose vectors in near-degenerate directions are not bitwise stable across runs
(multithreaded LAPACK); the seeds fix the Gaussian draws but not that factor. Over the 216 tests
the largest p-value difference is 0.018 (2.7 Monte Carlo standard errors of a difference), W_T
agrees to 4e-6, no decision changes at 10% or 5%, and one changes at 1% (quarterly, VIX = 15,
OTM call, concavity: reported p = 0.0102, regenerated 0.0097). The optional `cv` and `lw` stages and the
simulations are not part of this verification; the simulation tables are built from the saved
simulation outputs. Exhibits marked **P** below are produced by the authors' Python code and are
not reproduced by this repository.

| Stage | Runner(s) | Default | Runtime |
| --- | --- | --- | --- |
| `data` | `run_forward_reconstruction`, `run_stage1_reconstruction` (raw WRDS files) | no | ~3 min |
| `empirical` | `run_empirical_pipeline(H, true, 'eff_dim')`, `plot_conditional_sdf_figures` | yes | ~15 min |
| `cv` | `run_empirical_pipeline(H, true, 'cv')` | no | ~30 min |
| `noncrisis` | `run_noncrisis_effdim_fits` | yes | ~1 min |
| `oos` | `run_empirical_oos_attribution` (882 windows; primary payoff convention and the alternative payoff protocol), `run_oos_sensitivity_exclude_dates`, `make_oos_tables` | yes | ~45 min |
| `effdim` | `run_effdim_target_sensitivity('monthly')` | yes | ~10 min |
| `lw` | `run_lw_sharpe_inference('eff_dim')` | no | ~1 h |
| `econ` | `exp1_stageA/B`, `exp2_option_role`, `exp3_cross_maturity` | yes | ~20 min |
| `verify` | comparison with `results/` | yes | seconds |

Runs into `results/` call the same runners directly (the OOS production mode also
requires a clean Git state). Seeds: shape-test Monte Carlo p-values use per-draw seeds
(`rng(seed + b, 'twister')`); Ledoit-Wolf 2026081501
(calibration) and 2026081502 (final bootstrap); Experiment 2 2026112201; Experiment 3 2026112203
(B = 9999).

**Tests.** `run_bs_tests` and `run_chj_tests` run the unit tests of the estimator, shape inference,
CV solvers and the BS/CHJ simulation designs (seconds to minutes). Tests that build the empirical
configuration read the option panels in `data/`: `test_empirical_maturity_domain_identity` (in both
suites), `test_empirical_lambda_switch` and `verify_frozen_design`, which checks the design
parameters against this README. The remaining `tests/test_*`
files run individually; `pipelines/empirical_oos_attribution/tests/test_oos_smoke(scratchDir)` checks
the OOS stage with about 10 fits and requires the monthly fit produced by the `empirical` stage.

## Simulations (optional)

Simulations are never started by `run_replication`. Their saved outputs and the
tables and figures built from them are in `results/simulations/<eff_dim,fixed_lambda,cv>/` and
`results/figures/`. To reproduce them (Statistics Toolbox; 12 process workers if the Parallel
Computing Toolbox is installed; every runner resumes from its block checkpoints):

```matlab
run('drivers/setpath.m');
run_simulation_production('bs_increasing_t', true);                     % ~1 h
run_simulation_production('chj_increasing_t', true);                    % ~5.7 h
run_simulation_production('bs_empirical_t', true, 'eff_dim', 'monthly'); % one maturity per call (daily not run)
run_bs_hj_recovery_refit('eff_dim');  run_chj_hj_recovery_refit('eff_dim');
run_fixed_lambda_simulation('bs_increasing_t', true);                   % ~7.8 h
run_fixed_lambda_simulation('chj_increasing_t', true);                  % ~10 h
render_hac_simulated_shape_rejection_figures(true, 'eff_dim');          % Figures IA.A.3-IA.A.4
render_simulation_recovery_boxplots('eff_dim');                         % Figures IA.A.1-IA.A.2
render_bs_empirical_t_rejection_table('eff_dim');                       % IA BS empirical-T table
build_fixed_lambda_tables('bs_increasing_t'); build_fixed_lambda_tables('chj_increasing_t');  % Tables IA.A.1-IA.A.2
population_lambda_scan('scan_out');  % choice of the fixed-lambda grid (saved: results/simulations/fixed_lambda/grid_selection/)
```

Seeds: `seed0` is fixed in the study configurations; replication b at the i-th sample size uses
`seed0 + 1e6*(i-1) + b - 1`. The renderers and table builders read saved outputs only and
rewrite their output files.

## Manuscript exhibits

Categories: **M** MATLAB code in this repository; **P** the authors' Python code; **E** MATLAB
code with an input prepared by the authors' Python pipeline (documented, not reconstructed here).

| Exhibit | Cat. | Source |
| --- | --- | --- |
| Shape-test tables (monthly; IA quarterly, weekly, daily) | M | `run_empirical_pipeline` -> `results/empirical/eff_dim/empirical_all.csv` (LaTeX assembled by hand) |
| Table 4.1 trading performance and Ledoit-Wolf tests | E | `run_lw_sharpe_inference`, `run_noncrisis_effdim_fits`; variance-swap benchmark series from `data/return_pairs/` |
| OOS performance and attribution | M | `pipelines/empirical_oos_attribution` -> `results/empirical_oos/eff_dim/tables/` |
| Experiments 1-3 (economic interpretation) | M (Exp. 2: E) | `pipelines/economic_interpretation` -> `results/economic_interpretation/eff_dim/` |
| Table 4.2 transaction costs | E | Mid column and OOS cells MATLAB; cost columns and variance-swap rows from the Python pipeline |
| IA CV robustness (in-sample lambda and Sharpe ratios) | M | `results/empirical/cv/`; OOS Sharpe ratios at the CV lambda: P |
| IA effective-dimension target (0.5x, 2x) | M | `run_effdim_target_sensitivity` -> `results/empirical_effdim_target/monthly/` |
| Tables IA.A.1-IA.A.2 (fixed lambda) | M | `run_fixed_lambda_simulation` -> `build_fixed_lambda_tables` |
| Figures IA.A.3-IA.A.4 (simulation rejection rates) | M | `render_hac_simulated_shape_rejection_figures('eff_dim')` |
| IA BS empirical-T table | M | `render_bs_empirical_t_rejection_table('eff_dim')`; daily column not run |
| SDF, h', h'' figures, ablations, data summary, BS/CHJ recovery figures, Jackwerth and Dalderop-Linton comparisons, VRP/NCC/forecast-error/ECDF figures, SDF validation tables | P | authors' Python repository |

**Cross-validation.** The manuscript's cross-validation (robustness) searches lambda in
[1e-5, 1e-2] with a TPE optimizer. This repository minimizes the same one-step-ahead projection
loss over the grid `logspace(-5,-2,31)`, with the same initial training windows (20, 60, 260,
1260 periods). Both select lambda = 0.010 at all four maturities, and the fits reproduce the
reported CV in-sample Sharpe ratios (1.04, 0.96, 0.97, 1.50).

**Inputs from the Python pipeline.** `data/<h>/df.csv` (option panels) and
`data/return_pairs/` (variance-swap benchmark series, read by `run_lw_sharpe_inference`,
`run_noncrisis_effdim_fits` and `exp2_option_role`); the transaction-cost columns of Table 4.2.

**SDF exports for the Python validation.** The SDF validation tables and the VRP/NCC/PIT
exhibits (P) are computed in Python from the MATLAB fits. `export_validation_sdf_at_nodes(stageADir, outDir)`
evaluates the saved fitted `h` at the quadrature nodes supplied by the Python validation code and
writes the HJ-normalized SDF `M = 1 - h + qbar`; `export_pit_sdf_at_grid(inDir, outDir)` does the
same on the 500-point PIT grid, with `qbar` taken from the node export. Both read saved fits only
and are not part of `run_replication`.

# Design reference

## Empirical estimator

### Penalty selection

One switch, `get_frozen_design().empirical_lambda_method`, read by
`configs/config_empirical.m` as `cfg.lambda_method`; drivers accept an
optional override. Both methods share the estimator, payoff construction,
kernels, saved-state format and shape tests; only the lambda passed to the
final solve differs.

- `'eff_dim'` (default; manuscript baseline, eq. effective_dimension_rule):
  lambda solves `N_T(lambda) = sum_j mu_j/(mu_j + lambda) = T^(1/3)`, where
  mu_j are the eigenvalues of `G_T/T`. They are computed as the squared
  singular values of `F / R_K`, where `F` is the triangular factor of the
  scaled payoff panel (`training/build_payoff_factor.m`, `F'F = Q_alpha`) and
  `K_h = R_K'R_K`; neither `Q_alpha` nor a T-by-T Gram matrix is
  diagonalized, so eigenvalues far below `eps * cond(K_h)` are resolved. At
  most `min(Nh, T)` are kept. The fit is the equivalent least-squares QR solve
  (`training/solve_projection_qr.m`), whose triangular factor also gives the
  score/HAC inverse actions. The root is solved continuously in log lambda
  (no grid, no floor) to a relative residual of 1e-10; an unattainable target
  raises an error (`training/select_lambda_effective_dimension.m`).
- `'cv'` (robustness): one-step-ahead expanding-window CV over the grid
  `logspace(-5,-2,31)` with the mean projection-loss rule.

Outputs: `results/empirical/<lambda_method>/<horizon>/`. Fingerprints are
method-specific, so a CV checkpoint can never be resumed as an eff_dim one.
The simulation studies use the same rule through their own switch,
`get_frozen_design().simulation_lambda_method` (see Simulation studies).

- endogenous kernel: Matern 7/2; exogenous kernel: Laplace
- lengthscales: median pairwise distance over the option-level estimation
  rows (computed once and passed into the alpha and K_h construction)
- grid mode: fixed_grid; loss: projection_loss (uncentered Q_alpha)
- reference point: m = 1 (separate basis element)
- estimator grid size: 30 basis points per coordinate
- jitter: `K_M = K_M + 1e-10*I`, `K_V = K_V + 1e-10*I` applied separately to
  the two kernel factors before the Kronecker product `K_h = kron(K_V, K_M)`
- fit: `(Q_alpha + lambda*K_h) h = alpha_bar`
- panels: `data/<maturity>/df.csv`; quadrature: trapezoidal weights on each
  date's option strip; date-level sorting by (date, frwd_moneyness)

Estimator forward-moneyness basis domains (fixed, not data-dependent):

| maturity | basis domain | VIX basis grid |
| --- | --- | --- |
| quarterly | [0.7, 1.4] | linspace(5, 65, 30) |
| monthly | [0.7, 1.3] | linspace(5, 65, 30) |
| weekly | [0.7, 1.3] | linspace(min VIX, max VIX, 30) over the estimation panel |
| daily | [0.8, 1.2] | linspace(5, 65, 30) |

## Shape inference

Three forward-moneyness objects are always distinct: observed option
locations, the 30-point estimator basis and the 60-point shape-test grid.

- G = 60 equally spaced midpoint test points in raw forward moneyness
- test domain = figure domain INTERSECT empirical 20%-date-coverage support.
  The support is derived from the `data/<maturity>/df.csv` panel with one deterministic, interpolation-free
  rule: with k = ceil(0.2*T), support_lo is the k-th smallest per-date
  minimum and support_hi the k-th largest per-date maximum of
  frwd_moneyness. Verification requires exact equality with this
  derivation, and every test-grid point must retain at least 20% date
  coverage:

| maturity | figure log-range | 20% support | test domain |
| --- | --- | --- | --- |
| quarterly | [-0.25, 0.25] | [0.56147167769263, 1.18831169511829] | [exp(-0.25), 1.18831169511829] |
| monthly | [-0.30, 0.15] | [0.744456334784572, 1.09089309666474] | = support |
| weekly | [-0.20, 0.10] | [0.917587749971634, 1.03688696897526] | = support |
| daily | [-0.10, 0.10] | [0.967264263145474, 1.0208603191431] | = support |

- region cutoffs (OTM put / near-ATM / OTM call boundaries), all strictly
  inside the test domains: quarterly 0.880, 1.030; monthly 0.920, 1.030;
  weekly 0.945, 1.000; daily 0.970, 1.010
- shape restrictions, defined directly on the fitted h with derivatives in
  raw forward moneyness: mono `h_m >= 0`; conv `-h_mm >= 0`; conc
  `h_mm >= 0`
- statistic: one-sided cone Wald projection `W_T = T * min_{w>=0}
  (theta-w)' Omega^+ (theta-w)`; Omega is the Bartlett-HAC long-run score
  covariance with bandwidth `floor(4*(T/100)^(2/9))`; p-values from 10000
  Monte Carlo draws with the +1 correction and deterministic seeds
- lambda_inference = lambda_fit always (the selected eff_dim or CV lambda)
- conditional VIX states: five states chosen by greedy furthest-point
  selection (`evaluation/select_vix_levels.m`: 50 candidates over the fitted
  exogenous range, 200 evaluation points on the figure log-range); integer
  legend labels are for display only
- equal-date state: one observed VIX per date, clipped to the fitted
  exogenous support, averaged with weight 1/T

## Figures

Conditional figures display log forward moneyness on the x-axis; all kernel
evaluation is in raw forward moneyness (`m_raw = exp(m_log)`, 300 points).
The five conditional curves are scaled by one common mean
(`plotting/plot_conditional_sdf_figures.m`); this normalization affects the
figures only, not the shape tests.

## Simulation studies

All studies share: the fitting lambda (below), lambda_inference =
lambda_fit, HAC rule `nw_4_t100_2over9`, nMC = 10000, G = 60 midpoints for
bs_empirical_t and G = 52 for the increasing-T studies, and OTM contract
selection (puts below m = 1, calls above). Each study aligns its three forward-moneyness supports: generated
option observations, the 30-point estimator basis (plus the separate
reference point m = 1), and the 60-midpoint shape-test grid all use one
common domain. This alignment is simulation-only; the empirical application
keeps the three distinct domains tabulated above.
- simulation estimator kernel factors use the same `1e-10` pre-Kronecker
  jitter as the empirical estimator
- `nMC = 10000` is the Gaussian Monte Carlo null calibration inside one
  shape test; `B` is the number of independent simulated datasets

| | bs_increasing_t | bs_empirical_t | chj_increasing_t |
| --- | --- | --- | --- |
| maturity match | monthly | each maturity | daily |
| T | 500...10000 | empirical T | 500...10000 |
| B | 1000 | 2000 | 1000 |
| common domain | [0.95, 1.05] | exp(figure log range) | [0.95, 1.05] |
| observations | J = 30 linspace on domain | J = round(median date strike count inside the domain); linspace on domain | J = 30 linspace on domain |
| region cutoffs | 0.985, 1.015 | maturity empirical cutoffs | 0.985, 1.015 |
| exogenous grid | none | none | 30 points, replication q05--q95 VIX support |

The bs_empirical_t domains are `[exp(-0.25), exp(0.25)]` (quarterly),
`[exp(-0.30), exp(0.15)]` (monthly), `[exp(-0.20), exp(0.10)]` (weekly), and
`[exp(-0.10), exp(0.10)]` (daily), with J = 46, 77, 27, 28 respectively. The
30 increasing-T strikes on `[0.95, 1.05]` straddle m = 1 (15 puts and 15
calls), so no simulated contract sits on the at-the-money point.

### Penalty selection (simulations)

`get_frozen_design().simulation_lambda_method` (default `'eff_dim'`) selects
the fitting lambda in every simulation replication by the empirical
rule, `N_T(lambda) = T^(1/3)` (`effdim_c = 1`), with the same eigenvalue
convention, rank tolerance, root solve, SVD selector and QR solve. Results go to `results/simulations/eff_dim/<experiment>/` and figures to
`results/figures/eff_dim/`. `'cv'` gives the CV robustness results under
`results/simulations/cv/` (fitting lambda by CV over `logspace(-5,-2,31)`). Fixed-lambda studies (`run_fixed_lambda_simulation`) write to
`results/simulations/fixed_lambda/<experiment>/lambda_<tag>/`.

Finite-sample feasibility. The rule has a solution only if the numerical rank
of `G_T` exceeds `T^(1/3)`. In the BS simulator a contract that never finishes
in the money in a sample contributes a constant payoff column, so the payoff
rank is the number of paying contracts plus two, capped by the 31-dimensional
moneyness basis. The increasing-T studies, CHJ, and bs_empirical_t quarterly,
monthly and weekly clear the target comfortably (ranks 17--30
against targets 4.6--8.9 at the empirical T). bs_empirical_t daily does not:
its 28 uniform strikes span about +/-8 daily return standard deviations, so
only 11--14 ever pay, the rank is 11--14 against the target 12.52, and the
target is unattainable in about 5% of samples (marginal in about 40%). This is
a property of that controlled design, not of the empirical daily estimate
(rank 260). An unattainable replication stops the run with
`select_lambda_effective_dimension:Unattainable`, naming the seed. Under eff_dim, bs_empirical_t therefore runs one explicitly named
maturity per call, and `render_bs_empirical_t_rejection_table` labels any
table without all four maturities as partial.

### Lengthscales in the BS empirical-T design

The lengthscale rule is `option_level_median`: the median pairwise distance
over option-level rows, i.e. every strike repeated on every date. The BS
simulator takes the median over the J option locations
(`build_bs_alpha_precompute.m`), while CHJ computes the option-level median
directly (`run_chj_simulated_shape_rejection_study.m`). Including the zero
distances between repeated strikes can move the selected order statistic to a
neighboring pair-distance class, one grid step away, depending on the strike
count. For the increasing-T studies, J = 30 keeps both medians in the same
class (identical lengthscales for BS and CHJ), which `verify_simulation_design`
asserts. For `bs_empirical_t` the two medians differ at the quarterly and
daily maturities:

| maturity | J | class | grid-only | option_level_median | relative gap |
| --- | --- | --- | --- | --- | --- |
| quarterly | 46 | 14 vs 13 | 0.15718099712508238 | 0.14595378304471962 | 7.69e-02 |
| monthly | 77 | 23 vs 23 | 0.12741274351409204 | 0.12741274351409193 | 8.71e-16 |
| weekly | 27 | 8 vs 8 | 0.088135435383897232 | 0.088135435383897121 | 1.26e-15 |
| daily | 28 | 9 vs 8 | 0.066777833346562621 | 0.059358074085833601 | 1.25e-01 |

Simulation replay is deterministic: seed plus saved per-replication
lambda_fit (`training/resolve_job_lambda.m`).
