function [compact, population] = build_bs_power_diagnostics(raw, nQuadrature)
%% Build BS finite-sample size/power diagnostics from known population truth.
%
% mono and conv are true population nulls in the maintained BS economy; conc
% is a known false null. For false-null cells, the population cone distance is
% the Mahalanobis distance from the true derivative vector to the nonnegative
% orthant under the population score covariance corresponding to the selected
% lambda. The compact table reports the median of that population distance
% across the lambda values selected in independent simulated datasets.

if nargin < 2 || isempty(nQuadrature)
    nQuadrature = 120;
end

required = {'design_id', 'design_group', 'experiment', 'horizon', 'Tval', 'lambda', ...
    'region', 'testType', 'p_value', 'reject_1pct', 'reject_5pct', 'reject_10pct'};
for j = 1:numel(required)
    assert(ismember(required{j}, raw.Properties.VariableNames), ...
        'build_bs_power_diagnostics:MissingColumn', ...
        'Missing required raw column %s.', required{j});
end

%% Compute population geometry only for lambda values that actually occur
keys = unique(raw(:, {'experiment','horizon', 'lambda'}), 'rows');
popCells = cell(height(keys), 1);
for j = 1:height(keys)
    popCells{j} = one_horizon_lambda(keys.experiment(j),keys.horizon(j), keys.lambda(j), nQuadrature);
end
population = cat(1, popCells{:});

joined = innerjoin(raw, population, ...
    'Keys', {'experiment','horizon', 'lambda', 'region', 'testType'});

%% Aggregate finite-sample rejection and lambda-selection behavior
[G, design_id, design_group, experiment, horizon, Tval, region, testType, true_null] = findgroups( ...
    joined.design_id, joined.design_group, joined.experiment, joined.horizon, joined.Tval, ...
    joined.region, joined.testType, joined.true_null);

B = splitapply(@numel, joined.lambda, G);
valid_tests = splitapply(@(x) sum(isfinite(x)), joined.p_value, G);

lambda_median = splitapply(@median, joined.lambda, G);
lambda_inference_median = lambda_median;
if ismember('lambda_fit', joined.Properties.VariableNames)
    lambdaFit = joined.lambda_fit;
else
    lambdaFit = joined.lambda;
end
lambda_fit_median = splitapply(@median, lambdaFit, G);
lambda_changed_freq = splitapply(@(x) mean(x), double(joined.lambda < lambdaFit), G);
lambda_ratio_median = splitapply(@median, joined.lambda ./ lambdaFit, G);
lambda_p25 = splitapply(@(x) quantile(x, 0.25), joined.lambda, G);
lambda_p75 = splitapply(@(x) quantile(x, 0.75), joined.lambda, G);

cfg = config_bs_simulation_design('monthly','bs_increasing_t');
lambdaLower = min(cfg.training.cv.lambda_grid);
lambdaUpper = max(cfg.training.cv.lambda_grid);
lambda_lower_bound_freq = splitapply(@(x) mean(x == lambdaLower), joined.lambda, G);
lambda_upper_bound_freq = splitapply(@(x) mean(x == lambdaUpper), joined.lambda, G);

min_population_theta = splitapply(@median, joined.min_population_theta, G);
population_cone_distance = splitapply(@median, joined.population_cone_distance, G);

rej_1pct = splitapply(@(x) mean(x, 'omitnan'), joined.reject_1pct, G);
rej_5pct = splitapply(@(x) mean(x, 'omitnan'), joined.reject_5pct, G);
rej_10pct = splitapply(@(x) mean(x, 'omitnan'), joined.reject_10pct, G);
se_1pct = sqrt(rej_1pct .* (1 - rej_1pct) ./ valid_tests);
se_5pct = sqrt(rej_5pct .* (1 - rej_5pct) ./ valid_tests);
se_10pct = sqrt(rej_10pct .* (1 - rej_10pct) ./ valid_tests);

population_null = strings(numel(true_null), 1);
population_null(true_null) = "true";
population_null(~true_null) = "false";

diagnostic = strings(numel(true_null), 1);
diagnostic(true_null) = "size";
diagnostic(~true_null) = "power";

compact = table(design_id, design_group, experiment, horizon, Tval, B, valid_tests, ...
    region, testType, population_null, diagnostic, min_population_theta, ...
    population_cone_distance, lambda_median, lambda_inference_median, ...
    lambda_fit_median, lambda_changed_freq, lambda_ratio_median, ...
    lambda_p25, lambda_p75, lambda_lower_bound_freq, lambda_upper_bound_freq, ...
    rej_1pct, se_1pct, rej_5pct, se_5pct, rej_10pct, se_10pct, ...
    'VariableNames', {'design_id', 'design_group', 'experiment','horizon', 'Tval', ...
    'B', 'valid_tests', 'region', 'testType', 'population_null', 'diagnostic', ...
    'min_population_theta', 'population_cone_distance', ...
    'lambda_median', 'lambda_inference_median', 'lambda_fit_median', ...
    'lambda_changed_freq', 'lambda_ratio_median', 'lambda_p25', 'lambda_p75', ...
    'lambda_lower_bound_freq', 'lambda_upper_bound_freq', ...
    'rej_1pct', 'se_1pct', 'rej_5pct', 'se_5pct', 'rej_10pct', 'se_10pct'});

compact = sortrows(compact, {'design_id', 'testType', 'region'});

end

function out = one_horizon_lambda(experiment,horizon, lambda, nQuadrature)
%% Compute population derivative and covariance geometry for one horizon/lambda.

cfg = config_bs_simulation_design(char(horizon),char(experiment));
mon_grid = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
cfg.training.mon_grid = mon_grid;

pre = build_bs_alpha_precompute(cfg, mon_grid);
cfg.kernel.endo.lengthscale = pre.ell_m;
[K_h, K_M, K_V, basis] = build_kernel_norm_matrix(cfg, mon_grid, []);

[epsP, weights] = standard_normal_quadrature(nQuadrature);
logR = (cfg.bs.mu - 0.5 * cfg.bs.sigma^2) * cfg.bs.dt ...
    + cfg.bs.sigma * sqrt(cfg.bs.dt) * epsP;
R = exp(logR);
alpha = build_bs_alpha_from_returns(R, pre);

alpha_bar = alpha * weights;
alpha_weighted = alpha .* sqrt(weights.');
Q_alpha = alpha_weighted * alpha_weighted.';
Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

A = Q_alpha + lambda * K_h;
A = 0.5 * (A + A.');
h_star = solve_symmetric(A, alpha_bar);

model_state = struct();
model_state.alpha = alpha;
model_state.h_star = h_star(:);
model_state.H_star = reshape(h_star(:), basis.n + 1, basis.p);
model_state.lambda = lambda;
model_state.mon_grid = basis.mon_grid;
model_state.exo_grid = basis.exo_grid;
model_state.ls_endo = basis.ls_endo;
model_state.ls_exo = basis.ls_exo;
model_state.reference_point = basis.reference_point;
model_state.K_h = K_h;
model_state.K_M = K_M;
model_state.K_V = K_V;
model_state.alpha_bar = alpha_bar;
model_state.Q_alpha = Q_alpha;
model_state.T = nQuadrature;
model_state.Nh = numel(h_star);
model_state.cfg = cfg;

exo = build_exo_representer(cfg, model_state, struct());
grids = build_test_grids(cfg);
testTypes = cfg.testing.testTypes;

cells = cell(numel(grids) * numel(testTypes), 1);
row = 0;
for ig = 1:numel(grids)
    grid = grids(ig);
    for it = 1:numel(testTypes)
        testType = testTypes{it};
        [theta_regularized, L_Z] = build_theta_hat(cfg, model_state, grid, testType, exo);
        theta_population = true_bs_theta(cfg, grid.m, testType);
        Omega = population_score_covariance(alpha, weights, h_star, K_h, A, L_Z, lambda);

        stat_population = solve_wald_projection(theta_population, Omega, [], 1);
        stat_regularized = solve_wald_projection(theta_regularized, Omega, [], 1);

        row = row + 1;
        cells{row} = table(string(experiment),string(horizon), lambda, string(grid.name), string(testType), ...
            all(theta_population >= -cfg.testing.tau), ...
            min(theta_population), max(theta_population), ...
            sqrt(max(stat_population.W_T, 0)), ...
            sqrt(max(stat_regularized.W_T, 0)), ...
            'VariableNames', {'experiment','horizon', 'lambda', 'region', 'testType', ...
            'true_null', 'min_population_theta', 'max_population_theta', ...
            'population_cone_distance', 'regularized_cone_distance'});
    end
end

out = cat(1, cells{:});

end

function Omega = population_score_covariance(alpha, weights, h_star, K_h, A, L_Z, lambda)
%% Compute the iid population covariance of finite-grid influence scores.
%
% BS observations are independent, so the population long-run covariance is
% equal to the lag-zero covariance even though the finite-sample simulations
% use the empirical Bartlett HAC estimator.

U = solve_symmetric(A, L_Z);
eta = alpha.' * h_star;
psi_u = alpha.' * U;
h_u = h_star.' * K_h * U;
S = (1 - eta) .* psi_u - ones(size(alpha, 2), 1) * (lambda * h_u);
S_weighted = S .* sqrt(weights);
Omega = S_weighted.' * S_weighted;
Omega = 0.5 * (Omega + Omega.');

end

function theta = true_bs_theta(cfg, m, testType)
%% Evaluate the known population BS derivative restriction vector.

m = m(:);
a = (cfg.bs.mu - cfg.bs.r) / cfg.bs.sigma^2;
M = bs_true_sdf(m, cfg.bs.mu, cfg.bs.sigma, cfg.bs.dt, cfg.bs.r);
M_m = -a * M ./ m;
M_mm = a * (a + 1) * M ./ (m .^ 2);

switch lower(testType)
    case 'mono'
        theta = -M_m;
    case 'conv'
        theta = M_mm;
    case 'conc'
        theta = -M_mm;
    otherwise
        error('build_bs_power_diagnostics:BadTestType', ...
            'Unknown test type %s.', testType);
end

end

function grids = build_test_grids(cfg)
%% Build the retained BS midpoint grids without writing generated files.

specs = cfg.testing.regionSpecs;
mAll = build_global_shape_grid(cfg);

grids = repmat(struct('name', '', 'label', '', 'bounds', [], 'm', []), numel(specs), 1);
for j = 1:numel(specs)
    b = specs(j).bounds;
    if j < numel(specs)
        idx = mAll >= b(1) & mAll < b(2);
    else
        idx = mAll >= b(1) & mAll <= b(2);
    end
    grids(j).name = specs(j).name;
    grids(j).label = specs(j).label;
    grids(j).bounds = b;
    grids(j).m = mAll(idx);
end

end

function [z, weights] = standard_normal_quadrature(n)
%% Return Gauss--Hermite nodes and weights for a standard-normal expectation.

assert(isscalar(n) && n >= 10 && n == floor(n), ...
    'nQuadrature must be an integer of at least 10.');

offdiag = sqrt((1:(n - 1)) / 2);
J = diag(offdiag, 1) + diag(offdiag, -1);
[V, D] = eig(J);
[x, order] = sort(diag(D));
V = V(:, order);

z = sqrt(2) * x;
weights = (V(1, :) .^ 2).';
weights = weights / sum(weights);

end

function X = solve_symmetric(A, B)
%% Solve a symmetric linear system without changing the statistical object.

[L, flag] = chol(A, 'lower');
if flag == 0
    X = L' \ (L \ B);
else
    X = A \ B;
end

end
