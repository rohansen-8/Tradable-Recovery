function test_build_bs_alpha_from_returns()
%% Verify the shared BS alpha builder reproduces the seeded simulator path.

setpath();

cfg = config_bs_simulation_design('monthly');
cfg.sim.T = 25;
cfg.sim.seed = 12345;
mon_grid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.sim.J).';
pre = build_bs_alpha_precompute(cfg, mon_grid);

alphaNumeric = build_bs_alpha_numeric(cfg, mon_grid, cfg.sim.seed, pre);

rng(cfg.sim.seed, 'twister');
epsP = randn(cfg.sim.T, 1);
logR = (cfg.bs.mu - 0.5 * cfg.bs.sigma^2) * cfg.bs.dt ...
    + cfg.bs.sigma * sqrt(cfg.bs.dt) * epsP;
R = exp(logR);
alphaDirect = build_bs_alpha_from_returns(R, pre);

err = max(abs(alphaNumeric(:) - alphaDirect(:)));
assert(err < 1e-12, 'Shared BS alpha builder changed the simulated alpha panel.');

fprintf('test_build_bs_alpha_from_returns passed.\n');
end
