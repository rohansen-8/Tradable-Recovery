function test_bs_conv_conc_calibration_identity()
%% Verify that BS convexity and concavity share the same null calibration.
%
% Convexity and concavity use opposite second-derivative loadings. Negating
% all score coordinates leaves their covariance matrix unchanged, so the
% one-sided Wald null distribution is identical within a fixed test region.

cfg = config_bs_simulation_design('monthly');

mon_grid = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
cfg.training.mon_grid = mon_grid;
pre = build_bs_alpha_precompute(cfg, mon_grid);
cfg.kernel.endo.lengthscale = pre.ell_m;
[~, ~, ~, basis] = build_kernel_norm_matrix(cfg, mon_grid, []);

model_state = struct();
model_state.h_star = zeros((basis.n + 1) * basis.p, 1);
model_state.mon_grid = basis.mon_grid;
model_state.exo_grid = basis.exo_grid;
model_state.ls_endo = basis.ls_endo;
model_state.ls_exo = basis.ls_exo;
model_state.reference_point = basis.reference_point;

exo = build_exo_representer(cfg, model_state, struct());

bounds = cfg.testing.regionSpecs(2).bounds;
mAll = build_global_shape_grid(cfg);
grid = struct();
grid.m = mAll(mAll >= bounds(1) & mAll < bounds(2));

[~, L_conv] = build_theta_hat(cfg, model_state, grid, 'conv', exo);
[~, L_conc] = build_theta_hat(cfg, model_state, grid, 'conc', exo);

assert(norm(L_conv + L_conc, 'fro') <= 1e-12 * max(1, norm(L_conv, 'fro')));

rng(48211, 'twister');
T = 200;
S_conv = randn(T, size(L_conv, 2));
S_conc = -S_conv;

cov_opts = struct();
cov_opts.type = 'hac_bartlett';
cov_opts.bandwidth = 4;
cov_opts.bandwidth_rule = '';
cov_opts.winsor_margin = 0;
cov_opts.recenter = false;

[Omega_conv, ~] = build_score_covariance(S_conv, (1:T).', cov_opts);
[Omega_conc, ~] = build_score_covariance(S_conc, (1:T).', cov_opts);

assert(norm(Omega_conv - Omega_conc, 'fro') <= ...
    1e-12 * max(1, norm(Omega_conv, 'fro')));

opts = struct();
opts.parallel = false;
opts.rng_seed = 77123;
[~, ~, W_conv] = build_mc_pvalue(0, Omega_conv, [], T, 0.05, 200, opts);
[~, ~, W_conc] = build_mc_pvalue(0, Omega_conc, [], T, 0.05, 200, opts);

assert(isequaln(W_conv, W_conc));

fprintf('test_bs_conv_conc_calibration_identity passed.\n');

end
