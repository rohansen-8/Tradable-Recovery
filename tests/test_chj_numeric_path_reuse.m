function test_chj_numeric_path_reuse()
%% Verify that CHJ alpha built from a supplied path matches the same-seed
% construction and the simulated-panel route.

cfg = config_chj_simulation_design();
cfg.sim.T = 20;
cfg.sim.seed = 78123;

mon_grid = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
path = simulate_chj_numeric_path(cfg, cfg.sim.seed);

option_m_grid = chj_option_moneyness_grid(cfg);
cfg.kernel.endo.lengthscale = median_pairwise_lengthscale_weighted( ...
    option_m_grid, repmat(cfg.sim.T, numel(option_m_grid), 1));
cfg.kernel.exo.lengthscale = median_pairwise_lengthscale_weighted( ...
    path.VIX, repmat(cfg.sim.J, cfg.sim.T, 1));
cfg.training.exo_grid = build_chj_exo_grid(cfg, path.VIX);

pre = build_chj_alpha_precompute(cfg, mon_grid, cfg.training.exo_grid);
[alphaPath, infoPath] = build_chj_alpha_numeric(cfg, mon_grid, cfg.sim.seed, pre, path);
[alphaSeed, infoSeed] = build_chj_alpha_numeric(cfg, mon_grid, cfg.sim.seed, pre);

assert(isequal(size(alphaPath), size(alphaSeed)));
assert(max(abs(alphaPath(:) - alphaSeed(:))) <= 1e-12);
assert(max(abs(infoPath.VIX - infoSeed.VIX)) <= 1e-12);
assert(max(abs(infoPath.Delta - infoSeed.Delta)) <= 1e-12);

cfgPanel = cfg;
cfgPanel.sim.seed = cfg.sim.seed;
[dfPanel, truthPanel] = simulate_chj_panel(cfgPanel);
assert(max(abs(path.R - truthPanel.R)) <= 1e-12);
assert(max(abs(path.Delta - truthPanel.Delta)) <= 1e-12);
assert(max(abs(path.VIX - truthPanel.VIX)) <= 1e-12);

alphaPanel = build_alpha_panel(cfg, dfPanel, mon_grid, cfg.training.exo_grid);
assert(isequal(size(alphaPath), size(alphaPanel.alpha)));
assert(max(abs(alphaPath(:) - alphaPanel.alpha(:))) <= 1e-10 * ...
    max(1, max(abs(alphaPanel.alpha(:)))));

bounds = quantile(path.VIX, cfg.training.exo_support_quantiles);
expectedGrid = linspace(bounds(1), bounds(2), cfg.training.n_exo_grid).';
assert(max(abs(cfg.training.exo_grid - expectedGrid)) <= 1e-12);
assert(abs(path.VIX(1) - 0.20) <= 1e-12);

fprintf('test_chj_numeric_path_reuse passed\n');

end
