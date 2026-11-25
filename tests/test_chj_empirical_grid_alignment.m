function test_chj_empirical_grid_alignment()
%% Verify the 30-by-30 CHJ grid on the fixed increasing-T domain.

cfg = config_chj_simulation_design();
cfg.sim.T = 100;
path = simulate_chj_numeric_path(cfg, cfg.sim.seed);
exo_grid = build_chj_exo_grid(cfg, path.VIX);
mon_grid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
option_grid = chj_option_moneyness_grid(cfg);

bounds = quantile(path.VIX, cfg.training.exo_support_quantiles);
expectedExo = linspace(bounds(1), bounds(2), cfg.training.n_exo_grid).';
% Generation, estimator, and test supports coincide on the fixed domain.
domain = [0.95 1.05];
regionBounds = [0.95 0.985; 0.985 1.015; 1.015 1.05];

assert(cfg.training.n_m_grid == 30);
assert(cfg.training.n_exo_grid == 30);
assert(strcmp(cfg.training.exo_grid_type, 'replication_q05_q95_linspace'));
assert(isequal(cfg.training.exo_support_quantiles, [0.05, 0.95]));
assert(numel(mon_grid) == 30);
assert(numel(option_grid) == 30);
assert(numel(exo_grid) == 30);
assert(max(abs(exo_grid - expectedExo)) <= 1e-12);
assert((numel(mon_grid) + 1) * numel(exo_grid) == 930);
assert(abs(mon_grid(1) - domain(1)) <= 1e-14);
assert(abs(mon_grid(end) - domain(2)) <= 1e-14);
assert(max(abs(option_grid - linspace(domain(1), domain(2), 30).')) <= 1e-14);
assert(~any(abs(option_grid - 1) <= 1e-12));
assert(sum(option_grid < 1) == 15 && sum(option_grid > 1) == 15);
assert(abs(path.VIX(1) - 0.20) <= 1e-12);

cfg.grids = struct();
grids = build_region_grids(cfg);
allM = vertcat(grids.m);
globalM = build_global_shape_grid(cfg);
counts = arrayfun(@(g) numel(g.m), grids).';

assert(isequal(allM, globalM), ...
    'CHJ regional grids must exactly reconstruct the common 52-point midpoint grid.');
assert(numel(unique(allM)) == 52);
assert(all(allM > cfg.testing.grid.domain(1) & allM < cfg.testing.grid.domain(2)));
expectedCounts = arrayfun(@(j)sum(allM >= regionBounds(j,1) & ...
    (allM < regionBounds(j,2) | j == 3)), (1:3).');
assert(isequal(counts(:), expectedCounts(:)));
assert(abs(cfg.reference_point - 1) <= 1e-14, ...
    'The separate first-derivative representer reference point must remain m=1.');

fprintf('test_chj_empirical_grid_alignment passed\n');

end
