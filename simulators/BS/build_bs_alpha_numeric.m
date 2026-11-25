function alpha = build_bs_alpha_numeric(cfg, mon_grid, seed, pre)
%% Build the Black--Scholes alpha panel from simulated gross returns.

if nargin < 2 || isempty(mon_grid)
    mon_grid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.training.n_m_grid).';
end
if nargin < 3 || isempty(seed)
    seed = cfg.sim.seed;
end
if nargin < 4 || isempty(pre)
    pre = build_bs_alpha_precompute(cfg, mon_grid);
end

T = cfg.sim.T;
mu = cfg.bs.mu;
sigma = cfg.bs.sigma;
dt = cfg.bs.dt;

rng(seed, 'twister');
epsP = randn(T, 1);
logR = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * epsP;
R = exp(logR);

alpha = build_bs_alpha_from_returns(R, pre);

end
