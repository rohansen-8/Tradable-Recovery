function cfg = config_empirical(horizon, lambdaMethod)
%% Configure the empirical estimator fit and shape inference for one maturity.
%
% lambdaMethod names the lambda-selection specification and is the single
% control point for empirical output locations:
%   results/empirical/<lambdaMethod>/<horizon>/   fitted state, shape tests, figures
%   results/empirical/<lambdaMethod>/             cross-maturity summaries
% lambdaMethod = 'eff_dim' (default, get_frozen_design().empirical_lambda_method)
% selects lambda by the effective-dimension rule; 'cv' selects it by
% one-step-ahead CV over the fixed grid. Everything else in cfg is identical
% for the two methods, including the cfg.training.cv block, which documents
% the CV specification and is simply unused under 'eff_dim'.
%
% Estimator conventions: fixed basis grids, option-level median lengthscales,
% per-factor jitter, and the projection loss. Shape tests are evaluated on the
% pre-specified test domain.

horizon = validatestring(char(string(horizon)), ...
    {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon');
root = get_project_root();
design = get_frozen_design();
if nargin < 2 || isempty(lambdaMethod)
    lambdaMethod = design.empirical_lambda_method;
end
lambdaMethod = validatestring(char(string(lambdaMethod)), {'eff_dim','cv'}, ...
    mfilename, 'lambdaMethod');
maturity = get_empirical_maturity_config(horizon);

cfg = struct();
cfg.projectRoot = root;
cfg.lambda_method = lambdaMethod;
cfg.real.horizon = horizon;
cfg.revision.id = sprintf('empirical_%s_%s', horizon, design.revision);

cfg.dataset.name = horizon;
cfg.results.methodPath = fullfile(results_root(), 'empirical', lambdaMethod);
cfg.results.path = fullfile(cfg.results.methodPath, horizon);
cfg.results.figuresPath = fullfile(cfg.results.path, 'figures');
cfg.dataset.path = cfg.results.path;
cfg.dataset.dfFile = fullfile(root, 'data', horizon, 'df.csv');
cfg.dataset.modelStateFile = fullfile(cfg.dataset.path, ...
    sprintf('%s_model_state.mat', horizon));

cfg.has_exo = true;
cfg.exo_cols = {'VIX'};
cfg.reference_point = design.reference_point;

cfg.kernel.endo.type = design.kernel_endo;
cfg.kernel.endo.lengthscale = [];
cfg.kernel.exo.type = design.kernel_exo;
cfg.kernel.exo.lengthscale = [];
cfg.kernel.jitter = design.jitter;

% Estimator basis: fixed configured ranges (independent of the data), plus the
% separate reference point m = 1 inside build_kernel_norm_matrix.
cfg.training.lengthscale_rule = design.lengthscale_rule;
cfg.training.nMon = design.estimator_grid_points;
cfg.training.mon_domain = maturity.estimator_domain;
cfg.training.mon_grid = linspace(maturity.estimator_domain(1), ...
    maturity.estimator_domain(2), design.estimator_grid_points).';
cfg.training.nExo = design.estimator_grid_points;
if maturity.exo_range_is_data_driven
    exoRange = local_panel_exo_range(cfg);
else
    exoRange = maturity.exo_range;
end
cfg.training.exo_domain = exoRange;
cfg.training.exo_grid = linspace(exoRange(1), exoRange(2), ...
    design.estimator_grid_points).';

% Lambda selection. 'eff_dim': N_T(lambda) = effdim_c * T^(1/3) on the fitted
% sample (select_lambda_effective_dimension). 'cv': one-step-ahead CV by a
% deterministic search over the fixed coarse grid (see cross_validate_lambda).
% The CV block below holds the CV specification under both methods.
cfg.training.lambda = [];
cfg.training.lambda_method = lambdaMethod;
cfg.training.use_cv = strcmp(lambdaMethod, 'cv');
cfg.training.effdim.c = design.effdim_c;
cfg.training.cv.method = design.cv_method;
cfg.training.cv.lambda_grid = design.lambda_grid;
cfg.training.cv.lambda_scale_mode = 'absolute';
cfg.training.cv.parallel = false;
cfg.training.cv.solver = 'fold_eig_woodbury';
cfg.training.cv.woodbury_refresh = 40;
cfg.training.cv.min_train_dates = local_min_train_dates(horizon);
cfg.training.cv.validation_block_size = 1;
cfg.training.cv.selection_rule = 'mean_loss';
cfg.training.cv.objective = 'projection_loss';
cfg.training.save_model_state_as_variable = true;

% Shape inference: 60 midpoints on the test domain; three contiguous economic
% regions (OTM put / near-ATM / OTM call); conditional VIX states selected
% post-fit by select_vix_levels; lambda_inference = lambda_fit.
cfg.testing.alpha = design.test_alpha;
cfg.testing.nMC = design.nMC;
cfg.testing.edge_trim_fraction = 0;
cfg.testing.testTypes = design.shape_tests;
cfg.testing.mon_domain = maturity.test_domain;
cfg.testing.grid.type = design.shape_grid_type;
cfg.testing.grid.n_points = design.shape_grid_points;
cfg.testing.grid.domain = maturity.test_domain;
cfg.testing.regionBounds = maturity.region_bounds;
cfg.testing.regionSpecs = build_final_region_specs(maturity.region_bounds);
cfg.testing.regionNames = {'left','center','right'};
cfg.testing.hac_bandwidth_rule = design.hac_rule;
cfg.testing.inference_lambda.rule = design.inference_rule;
cfg.testing.conditional_levels_rule = 'select_vix_levels';
cfg.testing.exo.clip_to_grid = true;

cfg.results.cvCheckpoint = fullfile(cfg.results.path, ...
    sprintf('%s_cv_losses.mat', horizon));

assert(numel(cfg.training.mon_grid) == 30 && numel(cfg.training.exo_grid) == 30);
assert(cfg.testing.grid.n_points == 60);
assert(isequal(cfg.training.cv.lambda_grid(:), logspace(-5,-2,31).'));
assert(cfg.training.mon_domain(1) <= cfg.testing.grid.domain(1) && ...
    cfg.testing.grid.domain(2) <= cfg.training.mon_domain(2));

end

function range = local_panel_exo_range(cfg)
%% Return the estimation panel's min/max of the exogenous column.

df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
exoCols = cellstr(cfg.exo_cols);
assert(isscalar(exoCols), 'Data-driven exogenous range supports one column.');
v = df.(exoCols{1});
assert(all(isfinite(v)), 'Exogenous column contains nonfinite values.');
range = [min(v), max(v)];
assert(range(1) < range(2), 'Degenerate exogenous range.');

end

function n = local_min_train_dates(horizon)
%% Return the one-step-ahead CV starting date count.

switch horizon
    case 'quarterly', n = 20;
    case 'monthly', n = 60;
    case 'weekly', n = 260;
    case 'daily', n = 1260;
end

end
