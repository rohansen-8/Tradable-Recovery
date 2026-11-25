function cfg = config_chj_simulation_design(lambdaMethod)
%% Configure the CHJ increasing-T simulation experiment.
%
% lambdaMethod selects the fitting-lambda rule: 'cv' (default; the
% one-step CV over lambda_grid, results under results/simulations/cv/) or
% 'eff_dim' (N_T(lambda) = effdim_c * T^(1/3), results under
% results/simulations/eff_dim/) or 'fixed_lambda' (one supplied lambda, no
% selection; bind it with apply_fixed_lambda; results under
% results/simulations/fixed_lambda/). Everything else is identical.
%
% The CHJ economy is a one-trading-day state-dependent option economy,
% maturity-matched to the daily empirical application. Three forward-moneyness
% objects are configured and stay separately named: generated option
% observations (generation_domain), the 30-point estimator basis plus
% reference m = 1 (estimator_domain), and the G = 52 midpoint shape-test grid
% (test_domain). All three share the
% common increasing-T domain supplied by get_experiment_design. Forward
% moneyness is the feature throughout, so observed option strikes are equally
% spaced directly in forward moneyness.

if nargin < 1 || isempty(lambdaMethod), lambdaMethod = 'cv'; end
lambdaMethod = validatestring(char(string(lambdaMethod)), ...
    {'cv','eff_dim','fixed_lambda'}, mfilename, 'lambdaMethod');
projectRoot = get_project_root();
design = get_frozen_design();
item = get_experiment_design('chj_increasing_t', 'daily');

cfg = struct();
cfg.projectRoot = projectRoot;

%% Generated-output locations
cfg.dataset.name = 'chj_increasing_t';
cfg.sim.model = 'CHJ';
cfg.sim.design = cfg.dataset.name;
cfg.sim.experiment = 'chj_increasing_t';

cfg.dataset.path = fullfile(projectRoot, 'data', 'CHJ', cfg.dataset.name);
cfg.dataset.dfFile = fullfile(cfg.dataset.path, 'df.csv');
cfg.dataset.modelStateFile = fullfile(cfg.dataset.path, 'model_state.mat');

cfg.results.path = fullfile(simulation_output_dir('simulations', lambdaMethod), cfg.dataset.name);
cfg.results.momentDir = fullfile(cfg.results.path, 'moments');
cfg.results.diagnosticDir = fullfile(cfg.results.path, 'diagnostics');
cfg.results.shapeDir = fullfile(cfg.results.path, 'shape_tests');
cfg.results.rawPath = fullfile(cfg.results.path, 'raw');
cfg.results.summaryPath = fullfile(cfg.results.path, 'summary');
cfg.results.diagnosticsPath = cfg.results.diagnosticDir;

%% CHJ economy: one-trading-day design
cfg.chj.S0    = 1;
cfg.chj.mu    = 0.08 / 252;
cfg.chj.h0    = (0.20^2) / 252;
cfg.chj.omega = 1e-6;
cfg.chj.beta  = 0.88;
cfg.chj.alpha = 5e-4;
cfg.chj.gamma = 0.5;
cfg.chj.xi    = 0.9;
cfg.chj.r     = 0.02 / 252;

%% Simulation design
cfg.sim.seed0 = 500001;
cfg.sim.seed = cfg.sim.seed0;
cfg.sim.T = 500;
cfg.sim.performance_T = item.T(:).';
cfg.sim.final_B = item.B;
cfg.sim.J = item.J;
cfg.sim.horizon_days = 1;
cfg.sim.m_min = item.generation_domain(1);
cfg.sim.m_max = item.generation_domain(2);
cfg.sim.generation_domain = item.generation_domain;
cfg.sim.option_m_grid = linspace(item.generation_domain(1), ...
    item.generation_domain(2), item.J).';
cfg.sim.option_grid_type = 'forward_moneyness_linspace';

cfg.exp_table = table( ...
    cfg.sim.performance_T(:), ...
    repmat(cfg.sim.final_B, numel(cfg.sim.performance_T), 1), ...
    'VariableNames', {'Tval', 'nSeeds'});

%% RKHS learning design
cfg.has_exo = true;
cfg.exo_cols = {'VIX'};
cfg.reference_point = design.reference_point;

cfg.kernel.endo.type = 'matern72';
cfg.kernel.endo.lengthscale = [];
cfg.kernel.endo.lengthscale_rule = 'median';
cfg.kernel.exo.type = 'laplace';
cfg.kernel.exo.lengthscale = [];
cfg.kernel.exo.lengthscale_rule = 'median';
cfg.kernel.jitter = design.jitter;

% Every independent simulated dataset selects the fitting penalty with the
% shape-blind one-step-ahead projection-loss CV rule on the fixed 31-point
% grid; shape inference then tests the fitted object (lambda_inference =
% lambda_fit).
cfg.training.lambda = [];
cfg.training.lambda_method = lambdaMethod;
cfg.training.effdim_c = design.effdim_c;
cfg.training.use_cv = strcmp(lambdaMethod, 'cv');
cfg.training.cv.method = '1_step_ahead';
cfg.training.cv.lambda_grid = design.lambda_grid;
cfg.training.cv.parallel = false;
cfg.training.cv.solver = 'fold_eig_woodbury';
cfg.training.cv.woodbury_refresh = 40;
cfg.training.cv.min_train_dates = 10;
cfg.training.cv.selection_rule = 'mean_loss';
cfg.training.cv.objective = 'projection_loss';
cfg.training.mon_grid = [];
cfg.training.n_m_grid = design.estimator_grid_points;
cfg.training.estimator_domain = item.estimator_domain;
cfg.training.mon_domain = item.estimator_domain;
cfg.training.mon_grid_type = 'forward_moneyness_linspace';
cfg.training.exo_grid = [];

% Match the empirical 30-point equally spaced state-grid architecture.
% For each replication, use its 5th and 95th percentile VIX values as support
% endpoints and place 30 equally spaced state points between them.
cfg.training.n_exo_grid = 30;
cfg.training.exo_grid_type = 'replication_q05_q95_linspace';
cfg.training.exo_support_quantiles = [0.05, 0.95];

%% Shape-test design
cfg.testing.regions = struct();
cfg.testing.regions.left = item.region_bounds(1,:);
cfg.testing.regions.center = item.region_bounds(2,:);
cfg.testing.regions.right = item.region_bounds(3,:);
cfg.testing.regionSpecs = build_final_region_specs(item.region_bounds);

cfg.testing.testTypes = design.shape_tests;
cfg.testing.shapeConvention = design.shape_convention;

cfg.testing.alpha = design.test_alpha;
cfg.testing.nMC = design.nMC;
cfg.testing.tau = 1e-8;
cfg.testing.hac_bandwidth_rule = design.hac_rule;
cfg.testing.inference_lambda = struct();
cfg.testing.inference_lambda.rule = design.inference_rule;

% Equal-date state averaging follows the empirical evaluation convention:
% one VIX observation per date, clipped to the fitted exogenous-grid support,
% then averaged with weight 1/T.
cfg.testing.exo.mode = 'equal_date';
cfg.testing.exo.clip_to_grid = true;
cfg.testing.exo.convention = 'clip_to_exo_grid_then_equal_date';

cfg.testing.grid.type = design.shape_grid_type;
cfg.testing.grid.domain = item.test_domain;
cfg.testing.grid.n_points = item.shape_grid_points;

%% Prespecified regularization-sensitivity path
% Evaluated on each production dataset with the primary CV-fit HAC metric held
% fixed. This diagnostic never enters tuning or rejection decisions.
cfg.diagnostics.recovery_T = 2000;
cfg.diagnostics.regularization_lambda_path = [1e-2; 1e-3; 1e-4; 1e-5];
% Keep this path for the separate near-population diagnostic.
cfg.diagnostics.run_per_rep_regularization_path = false;

%% Parallel execution
% Parallelism is across independent Monte Carlo datasets. The CV path is
% serial because all lambda candidates are evaluated jointly at each fold.
cfg.parallel.use = false;
cfg.parallel.nWorkers = [];

% This id changes whenever the structural design changes.
cfg.revision.id = sprintf('%s_chj_increasing_t', design.revision);

end
