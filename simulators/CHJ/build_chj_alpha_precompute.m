function pre = build_chj_alpha_precompute(cfg, mon_grid, fixed_exo_grid)
%% Precompute deterministic CHJ option-pricing and moneyness-kernel objects.
%
% The exogenous grid and VIX lengthscale may be filled replication by
% replication. All expensive deterministic moneyness and Gauss--Hermite
% objects are reusable because the CHJ economy and moneyness design are fixed.

if nargin < 2 || isempty(mon_grid)
    mon_grid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.training.n_m_grid).';
end
if nargin < 3
    fixed_exo_grid = [];
end

mon_grid = mon_grid(:);

option_m_grid = chj_option_moneyness_grid(cfg);
option_weights = local_trapz_weights(option_m_grid);

ell_m = cfg.kernel.endo.lengthscale;
if isempty(ell_m)
    ell_m = median_pairwise_lengthscale(mon_grid);
end

fixed_exo_grid = fixed_exo_grid(:);
ell_v = cfg.kernel.exo.lengthscale;
if isempty(ell_v) && ~isempty(fixed_exo_grid)
    ell_v = median_pairwise_lengthscale(fixed_exo_grid);
end

zbar = cfg.reference_point;

% Gauss--Hermite nodes for standard-normal expectations.
nGH = 81;
ii = (1:nGH-1).';
C = diag(sqrt(ii/2), 1) + diag(sqrt(ii/2), -1);
[V,D] = eig(C);
gh_x = diag(D);
gh_w = V(1,:).^2;
[gh_x, idx] = sort(gh_x);
gh_w = gh_w(idx).';

pre = struct();
pre.mon_grid = mon_grid;
pre.option_m_grid = option_m_grid;
pre.option_weights = option_weights;
pre.fixed_exo_grid = fixed_exo_grid;
pre.ls_endo = ell_m;
pre.ls_exo = ell_v;
pre.reference_point = zbar;
pre.gh_z = sqrt(2) * gh_x;
pre.gh_w = gh_w;

pre.k00 = matern72_dxy(zbar, zbar, ell_m);
pre.k0m = matern72_dxyy(zbar, option_m_grid.', ell_m);
pre.kg0 = matern72_dxyy(zbar, mon_grid, ell_m);
pre.Kgm = matern72_dxxyy(mon_grid, option_m_grid.', ell_m);

assert(isfield(cfg, 'chj'), ...
    'build_chj_alpha_precompute requires cfg.chj.');
pre.chj = cfg.chj;
pre.params = calculate_chj_parameters(cfg.chj.mu, cfg.chj.xi, cfg.chj.omega, ...
    cfg.chj.beta, cfg.chj.alpha, cfg.chj.gamma, cfg.chj.r);

end

function w = local_trapz_weights(x)
%% Return composite trapezoidal weights on a strictly increasing grid.

x = x(:);
n = numel(x);
if n < 2 || any(diff(x) <= 0)
    error('build_chj_alpha_precompute:BadGrid', ...
        'The option moneyness grid must be strictly increasing.');
end

w = zeros(n, 1);
w(1) = 0.5 * (x(2) - x(1));
w(n) = 0.5 * (x(n) - x(n-1));
if n > 2
    w(2:n-1) = 0.5 * (x(3:n) - x(1:n-2));
end

end
