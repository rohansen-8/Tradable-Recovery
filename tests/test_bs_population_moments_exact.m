function test_bs_population_moments_exact()
%% Verify exact BS population moments and the projected-SDF shape convention.

setpath();

cfg = config_bs_simulation_design('monthly');
mon_grid = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
cfg.training.mon_grid = mon_grid;
pre = build_bs_alpha_precompute(cfg, mon_grid);
cfg.kernel.endo.lengthscale = pre.ell_m;
[K_h, ~, ~, basis] = build_kernel_norm_matrix(cfg, mon_grid, []);
pop = build_bs_population_moments_exact(cfg, mon_grid, pre);

% The two factorization identities are checked relative to the norm of the
% operator itself. Absolute thresholds are not usable here: ||Q_alpha||_F is
% O(1e5) on the simulation domain, so an absolute 1e-10 would demand a
% relative accuracy finer than double precision can deliver.
assert(norm(pop.alpha_bar - pop.B * pop.mean_x) < 1e-12);
assert(norm(pop.Q_alpha - pop.B * pop.Q_x * pop.B.', 'fro') <= ...
    1e-12 * norm(pop.Q_alpha, 'fro'));
assert(norm(pop.Q_alpha - pop.Q_alpha.', 'fro') <= ...
    1e-12 * norm(pop.Q_alpha, 'fro'));

% The exact BS HJ residual is c*dQ/dP. Its allocation representative is
% h = -c*dQ/dP up to an additive constant, so h_m > 0 and -h_mm > 0.
a = (cfg.bs.mu - cfg.bs.r) / cfg.bs.sigma^2;
thetaMarket = (cfg.bs.mu - cfg.bs.r) / cfg.bs.sigma;
c = exp(-(thetaMarket^2) * cfg.bs.dt);
m = build_global_shape_grid(cfg);
M = bs_true_sdf(m, cfg.bs.mu, cfg.bs.sigma, cfg.bs.dt, cfg.bs.r);
h_m = c * a * M ./ m;
minus_h_mm = c * a * (a + 1) * M ./ (m .^ 2);
assert(all(h_m > 0));
assert(all(minus_h_mm > 0));

% The 30-point working basis can interpolate the exact BS derivative data.
target = [c * a * bs_true_sdf(1, cfg.bs.mu, cfg.bs.sigma, cfg.bs.dt, cfg.bs.r); ...
    -c * a * (a + 1) * bs_true_sdf(mon_grid, cfg.bs.mu, cfg.bs.sigma, cfg.bs.dt, cfg.bs.r) ./ (mon_grid .^ 2)];
coef = K_h \ target;
model = struct();
model.h_star = coef;
model.mon_grid = basis.mon_grid;
model.exo_grid = basis.exo_grid;
model.ls_endo = basis.ls_endo;
model.ls_exo = basis.ls_exo;
model.reference_point = basis.reference_point;
exo = build_exo_representer(cfg, model, struct());
thetaConv = build_theta_hat(cfg, model, m, 'conv', exo);
assert(min(thetaConv) > 0, ...
    'The working basis should preserve BS convex-SDF curvature on the test grid.');

fprintf('test_bs_population_moments_exact passed.\n');

end
