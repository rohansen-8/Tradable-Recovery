function [alpha, info] = build_chj_alpha_numeric(cfg, mon_grid, seed, pre, path)
%% Build the CHJ alpha panel from one already-simulated state/return path.
%
% Passing path avoids a second stochastic simulation after replication-specific
% VIX lengthscale and exogenous-grid objects have been computed. If path is
% omitted, the same path is regenerated from seed.

if nargin < 2 || isempty(mon_grid)
    mon_grid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.training.n_m_grid).';
end
if nargin < 3 || isempty(seed)
    seed = cfg.sim.seed;
end
if nargin < 4 || isempty(pre)
    pre = build_chj_alpha_precompute(cfg, mon_grid, cfg.training.exo_grid);
end
if nargin < 5 || isempty(path)
    path = simulate_chj_numeric_path(cfg, seed);
end

T = cfg.sim.T;
assert(path.T == T, 'CHJ path length must equal cfg.sim.T.');
assert(path.seed == seed, 'CHJ path seed must equal the requested replication seed.');

chj = pre.chj;
params = pre.params;

mGrid = pre.option_m_grid(:);
wGrid = pre.option_weights(:);
J = numel(mGrid);

gh_z = pre.gh_z(:);
gh_w = pre.gh_w(:);

mon_grid = pre.mon_grid(:);
n = numel(mon_grid);

v_grid = pre.fixed_exo_grid(:);
assert(~isempty(v_grid), ...
    'build_chj_alpha_numeric requires the replication-specific exogenous grid.');

ell_v = pre.ls_exo;
assert(isscalar(ell_v) && isfinite(ell_v) && ell_v > 0, ...
    'build_chj_alpha_numeric requires a positive replication-specific VIX lengthscale.');

p = numel(v_grid);
Nh = (n + 1) * p;
alpha = zeros(Nh, T);
zeta = zeros(n + 1, T);
kV = zeros(p, T);

for t = 1:T
    h_t = path.h(t);
    R_t = path.R(t);

    Rq = exp(chj.r + (chj.mu - 0.5) * h_t + sqrt(h_t) * gh_z);
    h_next_q = chj.omega + chj.beta * h_t ...
        + chj.alpha * (gh_z - chj.gamma * sqrt(h_t)).^2;

    Mq = Rq.^params.phi .* exp(params.delta + params.eta * h_t ...
        + chj.xi * (h_next_q - h_t));
    EM = sum(gh_w .* Mq);

    callPayoffQ = max(Rq - mGrid.', 0);
    putPayoffQ = max(mGrid.' - Rq, 0);
    payoffQ = callPayoffQ;
    payoffQ(:, mGrid < 1) = putPayoffQ(:, mGrid < 1);

    forwardPrice = (gh_w .* Mq).' * payoffQ / EM;
    forwardPrice = forwardPrice(:);

    rawPayoff = max(R_t - mGrid, 0);
    rawPayoff(mGrid < 1) = max(mGrid(mGrid < 1) - R_t, 0);

    payoff_t = wGrid .* (rawPayoff - forwardPrice);

    zeta_t = zeros(n + 1, 1);
    zeta_t(1) = path.Delta(t) * pre.k00 + pre.k0m * payoff_t;
    zeta_t(2:end) = path.Delta(t) * pre.kg0 + pre.Kgm * payoff_t;

    kV_t = laplace_k(v_grid, path.VIX(t), ell_v);

    zeta(:, t) = zeta_t;
    kV(:, t) = kV_t;
    alpha(:, t) = kron(kV_t, zeta_t);
end

info = struct();
info.seed = seed;
info.T = T;
info.J = J;
info.mon_grid = mon_grid;
info.option_m_grid = mGrid;
info.exo_grid = v_grid;
info.ls_endo = pre.ls_endo;
info.ls_exo = ell_v;
info.VIX = path.VIX;
info.Delta = path.Delta;
info.zeta = zeta;
info.kV = kV;

end
