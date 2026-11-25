# Generated results

Tracked here: aggregate tables (`.csv`, `.tex`), publication figures (`.png`, `.pdf`) and READMEs.
Not tracked: `.mat` fits and checkpoints, per-replication and per-date files, and derived return
series (licensed-data derivatives). Large checkpoints are stored separately from the repository.

| Folder | Contents | Generator |
| --- | --- | --- |
| `empirical/<eff_dim,cv>/<h>/` | fitted state, 54 shape tests per maturity, figures; `empirical_all.csv` | `run_empirical_pipeline` |
| `empirical_noncrisis/eff_dim/` | non-crisis fits (Table 4.1 Panel B) | `run_noncrisis_effdim_fits` |
| `empirical_oos/eff_dim/` | OOS runs (`primary`; `reconciliation_python_protocol`, the alternative payoff protocol; `sensitivity_excl_20231019`), `tables/` | `pipelines/empirical_oos_attribution` |
| `inference/performance_sharpe/eff_dim_sb10/` | Ledoit-Wolf Sharpe-difference tests (Table 4.1); `validate_cv*` = checks of the implementation against reference bootstrap realizations | `run_lw_sharpe_inference` |
| `economic_interpretation/eff_dim/exp{1,2,3}/` | Experiments 1-3 | `pipelines/economic_interpretation` |
| `empirical_effdim_target/monthly/` | effective-dimension target 0.5x/1x/2x (IA table): fits, OOS windows, `effdim_target_monthly.csv` | `run_effdim_target_sensitivity` |
| `simulations/<eff_dim,cv,fixed_lambda>/<bs,chj>_*` | BS and CHJ simulation studies per lambda rule | `run_simulation_production`, `run_fixed_lambda_simulation` |
| `figures/<eff_dim,cv>/` | simulation figures and tables | `plotting/render_*` |

`empirical/cv/` (lambda = 0.01 at all maturities) and `simulations/cv/` hold the
cross-validation robustness outputs. The CHJ population checkpoint used by
`build_fixed_lambda_tables` is in `simulations/cv/chj_increasing_t/diagnostics/`.
