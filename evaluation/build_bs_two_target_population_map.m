function population_map = build_bs_two_target_population_map(raw)
%% Build analytic-economic and regularized BS shape classifications.
%
% This helper is deterministic. It uses the same horizon-specific option support,
% reduced RKHS basis, midpoint test grid, and derivative sign conventions as the
% production BS simulation. For each inference lambda that actually appears in
% the saved Monte Carlo output it compares two population targets:
%
%   economic target: the analytic positive-HJ Black--Scholes benchmark;
%   inferential target: h_lambda, the ordinary Tikhonov population target.
%
% The known analytical BS SDF shape is also reported as a check, but it is not a
% third inferential target. No simulated dataset, CV, HAC estimate, or cone-test
% Monte Carlo is recomputed here.

required = {'horizon','lambda_target','region','testType'};
for j = 1:numel(required)
    assert(ismember(required{j}, raw.Properties.VariableNames), ...
        'build_bs_two_target_population_map:MissingColumn', ...
        'Missing required column %s.', required{j});
end

assert(exist('build_bs_population_moments_exact', 'file') == 2, ...
    ['build_bs_population_moments_exact.m is required. Install the helper ' ...
     'included with the two-target patch.']);

if ~ismember('experiment',raw.Properties.VariableNames)
    raw.experiment = repmat("bs_empirical_t",height(raw),1);
    raw.experiment(string(raw.horizon)=="monthly") = "bs_increasing_t";
end
groups = unique(raw(:,{'experiment','horizon'}),'rows','stable');
horizonCells = cell(height(groups), 1);
for ih = 1:height(groups)
    experiment = groups.experiment(ih);
    horizon = groups.horizon(ih);
    mask = string(raw.experiment)==string(experiment) & string(raw.horizon)==string(horizon);
    lambdaValues = unique(raw.lambda_target(mask));
    horizonCells{ih} = one_horizon(experiment, horizon, lambdaValues);
end

population_map = cat(1, horizonCells{:});
population_map = sortrows(population_map, ...
    {'experiment','horizon','lambda_target','testType','region'});

end

function out = one_horizon(experiment, horizon, lambdaValues)
%% Prepare one exact BS population operator and evaluate all used lambdas.

cfg = config_bs_simulation_design(char(horizon),char(experiment));
mon_grid = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
cfg.training.mon_grid = mon_grid;

pre = build_bs_alpha_precompute(cfg, mon_grid);
cfg.kernel.endo.lengthscale = pre.ell_m;
[K_h, K_M, K_V, basis] = build_kernel_norm_matrix(cfg, mon_grid, []);
pop = build_bs_population_moments_exact(cfg, mon_grid, pre);

Q = symmetrize(double(pop.Q_alpha));
K = symmetrize(double(K_h));
b = double(pop.alpha_bar(:));
p = numel(b);
thetaMarket = (cfg.bs.mu - cfg.bs.r) / cfg.bs.sigma;
projectionScale = exp(-(thetaMarket^2) * cfg.bs.dt);

% Whiten by the RKHS norm. In z = K^{1/2}h coordinates, the lambda=0
% Moore-Penrose solution is exactly the minimum-RKHS-norm population target h0.
[LK, flagK] = chol(K, 'lower');
assert(flagK == 0, 'K_h must be positive definite for the two-target diagnostic.');
B = symmetrize(LK \ Q / LK.');
rhs = LK \ b;
[h0, h0Rank, h0Tol, h0RangeResidual] = minimum_norm_h0(B, rhs, LK);
assert(h0RangeResidual <= 1e-8, ...
    ['The unregularized population right-hand side is not numerically in the ' ...
     'range of the whitened operator (relative residual %.3g).'], h0RangeResidual);

% Build production region/test loadings once and reuse them for every lambda.
[items, exo] = build_population_items(cfg, basis, K_h, K_M, K_V, pop, p);
for j = 1:numel(items)
    items(j).theta_h0 = items(j).L_Z.' * h0;
    items(j).theta_true = true_bs_theta(cfg, items(j).m, items(j).testType);
    items(j).h0_null_holds = all(items(j).theta_h0 >= -cfg.testing.tau);
    items(j).true_bs_null_holds = all(items(j).theta_true >= -cfg.testing.tau);
end

nRows = numel(lambdaValues) * numel(items);
cells = cell(nRows, 1);
k = 0;
for il = 1:numel(lambdaValues)
    lambda = double(lambdaValues(il));
    assert(isfinite(lambda) && lambda > 0, 'Inference lambda must be positive and finite.');

    A = symmetrize(Q + lambda * K);
    h_lambda = solve_spd(A, b);

    population_residual = norm(A * h_lambda - b) / ...
        max(norm(A, 'fro') * norm(h_lambda) + norm(b), eps);
    pricingErrorSquared = real(1 - projectionScale - 2*(h_lambda.'*b) ...
        + h_lambda.'*Q*h_lambda);
    populationSdfL2Error = sqrt(max(pricingErrorSquared, 0));

    for j = 1:numel(items)
        item = items(j);
        theta_lambda = item.L_Z.' * h_lambda;
        lambdaNull = all(theta_lambda >= -cfg.testing.tau);
        h0Null = item.h0_null_holds;
        trueNull = item.true_bs_null_holds;

        changesNull = lambdaNull ~= trueNull;
        inducesViolation = trueNull && ~lambdaNull;
        masksViolation = ~trueNull && lambdaNull;
        economicEuclideanConeDistance = norm(min(item.theta_true, 0));
        regularizedEuclideanConeDistance = norm(min(theta_lambda, 0));

        k = k + 1;
        cells{k} = table(string(experiment),string(horizon), lambda, string(item.region), ...
            string(item.testType), numel(item.m), min(item.m), max(item.m), ...
            trueNull, h0Null, h0Null == trueNull, lambdaNull, changesNull, ...
            inducesViolation, masksViolation, min(item.theta_h0), ...
            max(item.theta_h0), min(item.theta_true), max(item.theta_true), ...
            min(theta_lambda), max(theta_lambda), ...
            economicEuclideanConeDistance, regularizedEuclideanConeDistance, ...
            populationSdfL2Error, norm(theta_lambda - item.theta_h0), ...
            h0Rank, h0Tol, h0RangeResidual, population_residual, ...
            'VariableNames', {'experiment','horizon','lambda_target','region','testType', ...
            'n_grid','grid_min','grid_max','economic_benchmark_null_holds', ...
            'h0_appendix_null_holds','h0_agrees_with_economic_benchmark', ...
            'regularized_target_null_holds','regularization_changes_null', ...
            'regularization_induces_violation','regularization_masks_violation', ...
            'h0_appendix_min_theta','h0_appendix_max_theta', ...
            'economic_min_theta','economic_max_theta', ...
            'regularized_min_theta','regularized_max_theta', ...
            'economic_euclidean_cone_distance', ...
            'regularized_euclidean_cone_distance', ...
            'population_sdf_L2_error','theta_bias_l2', ...
            'h0_numerical_rank','h0_rank_tolerance', ...
            'h0_range_residual','regularized_system_residual'});
    end
end

out = cat(1, cells{:});

% Keep the exogenous object live through construction so this function follows
% the exact common loading path even though BS has no exogenous state.
assert(~isempty(exo.e), 'BS exogenous representer construction failed.');

end

function [items, exo] = build_population_items(cfg, basis, K_h, K_M, K_V, pop, p)
%% Build exact production grids and derivative loading matrices.

template = make_model_state(cfg, zeros(p,1), NaN, pop, K_h, K_M, K_V, basis);
exo = build_exo_representer(cfg, template, struct());

mAll = build_global_shape_grid(cfg);
specs = cfg.testing.regionSpecs;
testTypes = unique(string(cfg.testing.testTypes), 'stable');

items = repmat(struct('region','', 'testType','', 'm',[], 'L_Z',[], ...
    'theta_h0',[], 'theta_true',[], 'h0_null_holds',false, ...
    'true_bs_null_holds',false), numel(specs) * numel(testTypes), 1);

row = 0;
for ir = 1:numel(specs)
    bounds = specs(ir).bounds;
    if ir < numel(specs)
        idx = mAll >= bounds(1) & mAll < bounds(2);
    else
        idx = mAll >= bounds(1) & mAll <= bounds(2);
    end
    assert(any(idx), 'Production region %s contains no shape-test points.', specs(ir).name);
    m = mAll(idx);

    for it = 1:numel(testTypes)
        row = row + 1;
        testType = char(testTypes(it));
        [~, L_Z] = build_theta_hat(cfg, template, m, testType, exo);
        items(row).region = specs(ir).name;
        items(row).testType = testType;
        items(row).m = m(:);
        items(row).L_Z = L_Z;
    end
end

end

function theta = true_bs_theta(cfg, m, testType)
%% Evaluate the known complete-market BS shape vector on the production grid.

m = m(:);
a = (cfg.bs.mu - cfg.bs.r) / cfg.bs.sigma^2;
M = bs_true_sdf(m, cfg.bs.mu, cfg.bs.sigma, cfg.bs.dt, cfg.bs.r);
M_m = -a * M ./ m;
M_mm = a * (a + 1) * M ./ (m.^2);

switch lower(char(testType))
    case 'mono'
        theta = -M_m;
    case 'conv'
        theta = M_mm;
    case 'conc'
        theta = -M_mm;
    otherwise
        error('build_bs_two_target_population_map:BadTestType', ...
            'Unknown test type %s.', testType);
end

end

function [h0, numericalRank, tol, rangeResidual] = minimum_norm_h0(B, rhs, LK)
%% Solve Bz=rhs for the minimum-Euclidean-norm z and map back to h0.

[U, S, V] = svd(B, 'econ');
s = diag(S);
if isempty(s)
    error('build_bs_two_target_population_map:EmptySpectrum', ...
        'Whitened BS population operator has empty spectrum.');
end

tol = max(size(B)) * eps(max(s));
keep = s > tol;
numericalRank = sum(keep);
assert(numericalRank >= 1, 'Whitened BS population operator has zero numerical rank.');

z0 = V(:,keep) * ((U(:,keep).' * rhs) ./ s(keep));
h0 = LK.' \ z0;
rangeResidual = norm(B * z0 - rhs) / max(norm(rhs), eps);

end

function model = make_model_state(cfg, h, lambda, pop, K_h, K_M, K_V, basis)
%% Assemble the minimal state required by the common derivative-loading helper.

model = struct();
model.h_star = h(:);
model.H_star = reshape(h(:), basis.n + 1, basis.p);
model.lambda = lambda;
model.mon_grid = basis.mon_grid;
model.exo_grid = basis.exo_grid;
model.ls_endo = basis.ls_endo;
model.ls_exo = basis.ls_exo;
model.reference_point = basis.reference_point;
model.K_h = K_h;
model.K_M = K_M;
model.K_V = K_V;
model.alpha_bar = pop.alpha_bar;
model.Q_alpha = pop.Q_alpha;
model.T = Inf;
model.Nh = numel(h);
model.cfg = cfg;

end

function X = solve_spd(A, B)
%% Solve a symmetric positive-definite population system without altering it.

A = symmetrize(A);
[L, flag] = chol(A, 'lower');
assert(flag == 0, 'Regularized BS population system must be positive definite.');
X = L.' \ (L \ B);

end

function A = symmetrize(A)
%% Remove only floating-point asymmetry.

A = 0.5 * (A + A.');

end
