# Data

The empirical option panels are not distributed with this repository. They are
derived from OptionMetrics IvyDB data licensed through WRDS, which does not
permit redistribution. `SHA256SUMS` lists the panels behind the reported results
(`shasum -a 256 -c SHA256SUMS` from this folder).

Place one CSV per maturity here before running the empirical pipeline:

```
data/quarterly/df.csv
data/monthly/df.csv
data/weekly/df.csv
data/daily/df.csv
```

Each row is one listed option contract on one date. The required columns are
`date`, `frwd_moneyness`, `VIX`, `OPT_NET_R_PAYOFF`, `weight` and `SP500`; see
the README section "Data". `return_pairs/` holds the variance-swap benchmark
series, built with the authors' Python pipeline and used by `run_lw_sharpe_inference` and `run_noncrisis_effdim_fits`.

Empirical support constants are derived from whichever panels are present, by a
deterministic 20%-date-coverage rule, and `verify_frozen_design` requires exact
equality between the derived supports and the design parameters. Substituting a
different panel therefore requires updating
`configs/get_empirical_maturity_config.m` to match.
