function population_lambda_scan(outDir)
%% Deterministic population scan for choosing fixed-lambda grids (no simulation, no MC).
% BS : exact analytic population map at half-decade lambdas in [1e-12, 1e-6].
% CHJ: saved 50,000-date population checkpoint (Q_alpha, K_h, alpha_bar, loadings), solved at
%      half-decade lambdas in [1e-6, 1]. Anchored to the saved production map at the fixed floor.
% Per lambda: population effective dimension N(lam) = sum mu/(mu+lam), mu = gen. eigs of (Q, K_h);
% whitened condition (mu_max+lam)/(mu_min+lam); rcond of Q + lam K_h; per-cell min/max theta and
% sign classification with the fixed tau. Reads saved inputs only; writes CSVs to outDir.

if ~isfolder(outDir), mkdir(outDir); end
setpath();

%% BS
lamBS = 10.^(-12:0.5:-6).';
cfg = config_bs_simulation_design('monthly', 'bs_increasing_t');
mon = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
cfg.training.mon_grid = mon;
pre = build_bs_alpha_precompute(cfg, mon); cfg.kernel.endo.lengthscale = pre.ell_m;
Kh = build_kernel_norm_matrix(cfg, mon, []);
pop = build_bs_population_moments_exact(cfg, mon, pre);
[muBS, rkBS] = gen_eigs(pop.Q_alpha, Kh);
req = table(repmat("bs_increasing_t", numel(lamBS), 1), repmat("monthly", numel(lamBS), 1), lamBS, ...
    repmat("left", numel(lamBS), 1), repmat("mono", numel(lamBS), 1), ...
    'VariableNames', {'experiment', 'horizon', 'lambda_target', 'region', 'testType'});
map = build_bs_two_target_population_map(req);
cellsBS = table(repmat("BS", height(map), 1), map.lambda_target, string(map.region), string(map.testType), ...
    double(map.economic_benchmark_null_holds), map.economic_min_theta, double(map.regularized_target_null_holds), ...
    map.regularized_min_theta, map.regularized_max_theta, ...
    'VariableNames', {'dgp', 'lambda', 'region', 'restriction', 'economic_holds', 'economic_min_theta', ...
    'population_holds', 'population_min_theta', 'population_max_theta'});
sumBS = spectral_summary("BS", lamBS, muBS, rkBS, pop.Q_alpha, Kh);

%% CHJ
lamCHJ = 10.^(-6:0.5:0).';
dd = fullfile(simulation_output_dir('simulations', 'cv'), 'chj_increasing_t', 'diagnostics', 'chj_population_T50000_burn5000_grid52');
P = load(fullfile(dd, 'chj_population_checkpoint.mat'));
tau = config_chj_simulation_design().testing.tau;
[muCHJ, rkCHJ] = gen_eigs(P.Q_alpha, P.K_h);
floorLam = get_frozen_design().lambda_grid(1);
S = load(fullfile(dd, 'chj_population_lambda_solutions.mat'), 'populationMap'); M0 = S.populationMap;
M0 = M0(abs(M0.lambda - floorLam) <= 10 * eps, :);
A0 = chj_cells(P, floorLam, tau);
gap = 0; sameClass = true;
for j = 1:height(M0)
    k = A0.region == string(M0.region(j)) & A0.restriction == string(M0.testType(j));
    gap = max(gap, abs(A0.population_min_theta(k) - M0.regularized_min_theta(j)) / max(1, abs(M0.regularized_min_theta(j))));
    sameClass = sameClass && A0.population_holds(k) == double(M0.regularized_null_holds(j));
end
% R2026b logspace(-5,-2,31) differs from R2025b in the last bit of entry 1 (known); relative 1e-8 bound.
assert(gap <= 1e-8 && sameClass, 'CHJ anchor relative gap %.3e', gap);
ts = P.trueShape; parts = cell(numel(lamCHJ), 1);
for il = 1:numel(lamCHJ), parts{il} = chj_cells(P, lamCHJ(il), tau); end
cellsCHJ = vertcat(parts{:});
cellsCHJ.economic_min_theta = nan(height(cellsCHJ), 1); cellsCHJ.economic_holds = nan(height(cellsCHJ), 1);
for j = 1:height(ts)
    k = cellsCHJ.region == string(ts.region(j)) & cellsCHJ.restriction == string(ts.testType(j));
    cellsCHJ.economic_min_theta(k) = ts.min_shape_value(j);
    cellsCHJ.economic_holds(k) = double(ts.min_shape_value(j) >= -tau);
end
cellsCHJ = cellsCHJ(:, cellsBS.Properties.VariableNames);
sumCHJ = spectral_summary("CHJ", lamCHJ, muCHJ, rkCHJ, P.Q_alpha, P.K_h);

writetable([cellsBS; cellsCHJ], fullfile(outDir, 'population_cells_by_lambda.csv'));
writetable([sumBS; sumCHJ], fullfile(outDir, 'population_spectral_by_lambda.csv'));
fprintf('CHJ anchor relative gap at frozen floor: %.3e (classifications identical)\n', gap);
fprintf('BS rank %d, mu1 %.4g; CHJ rank %d, mu1 %.4g\n', rkBS, muBS(1), rkCHJ, muCHJ(1));
end

function [mu, rk] = gen_eigs(Q, K)
%% Generalized eigenvalues of (Q, K): eig(L^{-1} Q L^{-T}), K = L L'. Rank at relative 1e-12.
Q = (Q + Q.') / 2; K = (K + K.') / 2; L = chol(K, 'lower');
B = L \ Q / L.'; mu = sort(eig((B + B.') / 2), 'descend'); mu = max(mu, 0);
rk = sum(mu > 1e-12 * mu(1));
end

function T = spectral_summary(dgp, lam, mu, rk, Q, K)
%% Population effective dimension and conditioning per lambda.
m = mu(1:rk); n = numel(lam); N = zeros(n, 1); wc = N; rc = N;
for i = 1:n
    N(i) = sum(m ./ (m + lam(i)));
    wc(i) = (m(1) + lam(i)) / (m(end) + lam(i));
    A = Q + lam(i) * K; rc(i) = rcond((A + A.') / 2);
end
T = table(repmat(dgp, n, 1), lam, N, wc, rc, repmat(rk, n, 1), repmat(m(1), n, 1), repmat(m(end), n, 1), ...
    'VariableNames', {'dgp', 'lambda', 'population_eff_dim', 'whitened_condition', 'rcond_system', ...
    'operator_rank', 'mu_max', 'mu_min_in_rank'});
end

function C = chj_cells(P, lambda, tau)
%% Cholesky solve of the symmetrized population system; theta = loadings' * h.
A = 0.5 * ((P.Q_alpha + lambda * P.K_h) + (P.Q_alpha + lambda * P.K_h).');
[L, f] = chol(A, 'lower'); assert(f == 0);
h = L.' \ (L \ P.alpha_bar); th = P.L_all.' * h; nb = height(P.loadingBlocks); r = cell(nb, 1);
for ib = 1:nb
    b = P.loadingBlocks(ib, :); t = th(b.first_index:b.last_index);
    r{ib} = table("CHJ", lambda, string(b.region), string(b.testType), double(all(t >= -tau)), min(t), max(t), ...
        'VariableNames', {'dgp', 'lambda', 'region', 'restriction', 'population_holds', ...
        'population_min_theta', 'population_max_theta'});
end
C = vertcat(r{:});
end
