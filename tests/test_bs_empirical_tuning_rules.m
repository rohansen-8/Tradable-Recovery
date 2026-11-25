function test_bs_empirical_tuning_rules()
%% Verify BS kernel and HAC tuning against the empirical conventions.
%
% The empirical lengthscale rule is median(pdist(training_moneyness)). The BS
% increasing-T panel repeats the same 34 moneyness points every date, and at
% that strike count the full-panel median is identical to the fixed-grid
% median used by the simulator. The equality is a property of the strike
% count, not a general one: verify_simulation_design asserts it so the design
% cannot drift onto a J where it fails.

setpath();

cfg = config_bs_simulation_design('monthly','bs_increasing_t');
option_grid = cfg.sim.option_m_grid;
mon_grid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2), cfg.training.n_m_grid).';

ellGrid = median_pairwise_lengthscale(option_grid);
assert(ellGrid > 0 && isfinite(ellGrid));

% Direct analogue of scipy pdist on a repeated training panel.
training_moneyness = repmat(option_grid, 3, 1);
D = abs(training_moneyness - training_moneyness.');
d = D(triu(true(numel(training_moneyness)), 1));
ellPanel = median(d);
assert(abs(ellPanel - ellGrid) < 1e-12);

pre = build_bs_alpha_precompute(cfg, mon_grid);
cfg.kernel.endo.lengthscale = pre.ell_m;
[~, ~, ~, basis] = build_kernel_norm_matrix(cfg, mon_grid, []);
assert(abs(pre.ell_m - ellGrid) < 1e-12);
assert(abs(basis.ls_endo - ellGrid) < 1e-12);
assert(isempty(basis.ls_exo));

Tvals = [97; 273; 694; 1964; 500; 1000; 2000; 5000; 10000];
expected = [3; 5; 6; 7; 5; 6; 7; 9; 11];
actual = arrayfun(@bartlett_hac_bandwidth, Tvals);
assert(isequal(actual, expected));

fprintf('test_bs_empirical_tuning_rules passed.\n');

end
