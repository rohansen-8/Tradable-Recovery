function test_chj_hj_recovery()
%% Synthetic checks for the CHJ HJ-normalized recovery (seconds, no saved results).
%
% 1. build_chj_zeta_from_returns at R = path.R(t) reproduces the moneyness
%    block zeta(:, t) of build_chj_alpha_numeric on every date.
% 2. The per-date HJ SDF at R = path.R(t) equals the estimator's payoff SDF
%    1 - alpha_t'h built from the estimation panel.
% 3. build_chj_sdf_recovery_curve runs on the fit and returns a mean-one,
%    zero-mean-difference curve.

setpath();
cfg = config_chj_simulation_design('eff_dim');
Tval = 300; seed = 4242;
cfg.sim.T = Tval; cfg.sim.seed = seed;
mon_grid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
optGrid = chj_option_moneyness_grid(cfg);
cfg.kernel.endo.lengthscale = median_pairwise_lengthscale_weighted( ...
    optGrid, repmat(Tval, numel(optGrid), 1));
cfg.training.mon_grid = mon_grid;
pre = build_chj_alpha_precompute(cfg, mon_grid, []);
path = simulate_chj_numeric_path(cfg, seed);
theta_v = median_pairwise_lengthscale_weighted(path.VIX, repmat(cfg.sim.J, Tval, 1));
exo_grid = build_chj_exo_grid(cfg, path.VIX);
cfg.kernel.exo.lengthscale = theta_v;
cfg.training.exo_grid = exo_grid;
cfg.training.n_exo_grid = numel(exo_grid);
pre.fixed_exo_grid = exo_grid; pre.ls_exo = theta_v;
[alpha, info] = build_chj_alpha_numeric(cfg, mon_grid, seed, pre, path);

errZ = 0;
for t = 1:Tval
    z = build_chj_zeta_from_returns(path.R(t), path.h(t), pre);
    errZ = max(errZ, max(abs(z - info.zeta(:, t))) / max(1, max(abs(info.zeta(:, t)))));
end
assert(errZ < 1e-12, 'zeta mismatch %.3g', errZ);

[K_h, ~, ~, basis] = build_kernel_norm_matrix(cfg, mon_grid, exo_grid);
payoff = build_payoff_factor(alpha);
h = solve_projection_qr(payoff, K_h, 1e-3);
H = reshape(h, basis.n + 1, basis.p);
KV = laplace_k(exo_grid(:), path.VIX(:).', theta_v);
hjAtR = zeros(Tval, 1);
for t = 1:Tval
    hjAtR(t) = 1 - build_chj_zeta_from_returns(path.R(t), path.h(t), pre).' * (H * KV(:, t));
end
errM = max(abs(hjAtR - (1 - alpha.' * h)));
assert(errM < 1e-10, 'HJ SDF vs payoff SDF mismatch %.3g', errM);

ms = struct('H_star', H, 'lambda', 1e-3, 'exo_grid', exo_grid(:), 'ls_exo', theta_v);
r1 = build_chj_sdf_recovery_curve(cfg, ms, path, Tval, 1, seed, pre);
assert(all(isfinite(r1.difference)) && abs(mean(r1.difference)) < 1e-10);
fprintf('test_chj_hj_recovery passed: zeta err %.2e, payoff-SDF err %.2e, max |diff| %.3f\n', ...
    errZ, errM, max(abs(r1.difference)));
end
