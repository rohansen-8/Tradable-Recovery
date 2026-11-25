function report = verify_supported_testing_domain(horizon)
%% Re-derive the 20%-date-coverage support and test domain from the panel.
%
% Support definition (pointwise): a forward-moneyness point m is covered on
% date t when min_t <= m <= max_t of that date's observed option strip; the
% support is the interval of points covered on at least 20% of dates.
% Deterministic, interpolation-free computation with k = ceil(0.2*T):
%   support_lo = k-th smallest per-date minimum of frwd_moneyness
%   support_hi = k-th largest  per-date maximum of frwd_moneyness
% The constants in get_empirical_maturity_config ARE this computation
% on the data panels, so the comparison requires exact equality - any
% deviation means the panel or the constants changed. The test domain is the
% intersection of exp(figure_log_range) with the support, and every point of
% the fixed 60-point test grid must retain at least 20% date coverage.

horizon = validatestring(char(string(horizon)), ...
    {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon');
maturity = get_empirical_maturity_config(horizon);
cfg = config_empirical(horizon);

df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
[g,~] = findgroups(df.date);
loByDate = splitapply(@min, df.frwd_moneyness, g);
hiByDate = splitapply(@max, df.frwd_moneyness, g);
T = numel(loByDate);
k = ceil(0.2 * T);
minsSorted = sort(loByDate(:));
maxsSorted = sort(hiByDate(:), 'descend');

report = struct();
report.horizon = horizon;
report.T = T;
report.k_order_statistic = k;
report.frozen_support = maturity.support_domain;
report.recomputed_support = [minsSorted(k), maxsSorted(k)];
report.support_deviation = ...
    abs(report.recomputed_support - report.frozen_support);
report.frozen_test_domain = maturity.test_domain;
report.recomputed_test_domain = [ ...
    max(exp(maturity.figure_log_range(1)), report.recomputed_support(1)), ...
    min(exp(maturity.figure_log_range(2)), report.recomputed_support(2))];
report.test_domain_deviation = ...
    abs(report.recomputed_test_domain - report.frozen_test_domain);
report.exact_match = all(report.support_deviation == 0) && ...
    all(report.test_domain_deviation == 0);

% Coverage of the fixed test grid on the data panel.
m = build_global_shape_grid(cfg);
cover = mean(loByDate(:).' <= m(:) & m(:) <= hiByDate(:).', 2);
report.min_point_coverage = min(cover);
report.n_points_below_20pct = sum(cover < 0.20);
report.all_test_points_covered_20pct = all(cover >= 0.20);
report.region_cutoffs = maturity.region_cutoffs;
report.cutoffs_inside = ...
    maturity.region_cutoffs(1) > maturity.test_domain(1) && ...
    maturity.region_cutoffs(2) < maturity.test_domain(2);

end
