function cfg = config_bs_simulation_design(horizon, experiment, lambdaMethod)
%% Configure one Black--Scholes simulation experiment and horizon.
%
% lambdaMethod selects the fitting-lambda rule: 'cv' (default; the
% one-step CV over lambda_grid, results under results/simulations/cv/) or
% 'eff_dim' (N_T(lambda) = effdim_c * T^(1/3), results under
% results/simulations/eff_dim/) or 'fixed_lambda' (one supplied lambda, no
% selection; bind it with apply_fixed_lambda; results under
% results/simulations/fixed_lambda/). Everything else is identical.
%
% Three forward-moneyness objects are configured and stay separately named:
% generated option observations (generation_domain), the 30-point estimator
% basis plus reference m = 1 (estimator_domain), and the midpoint shape-test
% grid (test_domain; G = 52 for bs_increasing_t, G = 60 for bs_empirical_t). All
% three share one common experiment-specific domain supplied by
% get_experiment_design.

if nargin < 1 || isempty(horizon), horizon = 'monthly'; end
horizon = validatestring(char(string(horizon)), ...
    {'daily','weekly','monthly','quarterly'}, mfilename, 'horizon');
if nargin < 2 || isempty(experiment)
    if strcmp(horizon, 'monthly'), experiment = 'bs_increasing_t';
    else, experiment = 'bs_empirical_t'; end
end
experiment = validatestring(char(string(experiment)), ...
    {'bs_increasing_t','bs_empirical_t'}, mfilename, 'experiment');
if nargin < 3 || isempty(lambdaMethod), lambdaMethod = 'cv'; end
lambdaMethod = validatestring(char(string(lambdaMethod)), ...
    {'cv','eff_dim','fixed_lambda'}, mfilename, 'lambdaMethod');
design = get_frozen_design();
item = get_experiment_design(experiment, horizon);
[dt, cvMinTrain] = horizon_design(horizon);
empiricalT = item.T_empirical;

% Generated option observations: the economically correct OTM contract on
% each side of m = 1 (selected inside the alpha builders; no generation grid
% contains m = 1 itself).
if strcmp(experiment, 'bs_increasing_t')
    optionGrid = linspace(item.generation_domain(1), ...
        item.generation_domain(2), item.J).';
    strikeDesign = struct('horizon',horizon, ...
        'source_file','empirical_maturity_config', ...
        'filter','not applicable','domain',item.generation_domain, ...
        'n_dates',NaN,'median_count_before_rounding',item.J,'J',item.J, ...
        'count_quantiles',repmat(item.J,1,5),'n_pooled',NaN, ...
        'moneyness_quantiles',quantile(optionGrid,[0 .1 .25 .5 .75 .9 1]), ...
        'probabilities',[],'option_grid',optionGrid);
else
    strikeDesign = build_bs_empirical_strike_design(horizon, ...
        item.generation_domain);
    optionGrid = strikeDesign.option_grid;
end

projectRoot = get_project_root();
namespace = experiment;
cfg = struct();
cfg.projectRoot = projectRoot;
cfg.results.path = fullfile(simulation_output_dir('simulations', lambdaMethod), namespace);
cfg.results.momentDir = fullfile(cfg.results.path, 'moments');
cfg.results.diagnosticDir = fullfile(cfg.results.path, 'diagnostics');
cfg.results.shapeDir = fullfile(cfg.results.path, 'shape_tests');
cfg.results.rawPath = fullfile(cfg.results.path, 'raw');
cfg.results.summaryPath = fullfile(cfg.results.path, 'summary');
cfg.results.figuresPath = fullfile(simulation_output_dir('figures', lambdaMethod), namespace);
cfg.results.diagnosticsPath = cfg.results.diagnosticDir;
cfg.dataset.name = namespace;
cfg.dataset.path = fullfile(projectRoot, 'data', 'BS', namespace);
cfg.dataset.dfFile = fullfile(cfg.dataset.path, 'df.csv');
cfg.dataset.modelStateFile = fullfile(cfg.dataset.path, 'model_state.mat');

cfg.bs = struct('mu',0.08,'sigma',0.20,'dt',dt,'S0',1,'r',0);
cfg.sim.model = 'BS';
cfg.sim.design = namespace;
cfg.sim.experiment = experiment;
cfg.sim.horizon = horizon;
cfg.sim.seed0 = 940001;
cfg.sim.seed = cfg.sim.seed0;
cfg.sim.empirical_T = empiricalT;
cfg.sim.performance_T = design.bs_increasing_t.T;
if strcmp(experiment,'bs_increasing_t'), cfg.sim.T = item.T(1); else, cfg.sim.T = empiricalT; end
cfg.sim.final_B = item.B;
cfg.sim.J = numel(optionGrid);
cfg.sim.m_min = item.generation_domain(1);
cfg.sim.m_max = item.generation_domain(2);
cfg.sim.generation_domain = item.generation_domain;
cfg.sim.option_m_grid = optionGrid;
cfg.sim.option_grid_type = item.option_grid_type;
cfg.sim.strike_design = strikeDesign;
cfg.exp_table = table(item.T(:), repmat(cfg.sim.final_B,numel(item.T),1), ...
    'VariableNames',{'Tval','nSeeds'});

cfg.has_exo = false;
cfg.exo_cols = {};
cfg.reference_point = design.reference_point;
cfg.kernel.endo = struct('type','matern72','lengthscale',[],'lengthscale_rule','median');
cfg.kernel.exo = struct('type','none','lengthscale',[],'lengthscale_rule','none');
cfg.kernel.jitter = design.jitter;
cfg.training.lambda = [];
cfg.training.lambda_method = lambdaMethod;
cfg.training.effdim_c = design.effdim_c;
cfg.training.use_cv = strcmp(lambdaMethod, 'cv');
cfg.training.cv.method = '1_step_ahead';
cfg.training.cv.lambda_grid = design.lambda_grid;
cfg.training.cv.parallel = false;
cfg.training.cv.solver = 'fold_eig';
cfg.training.cv.min_train_dates = cvMinTrain;
cfg.training.cv.selection_rule = 'mean_loss';
cfg.training.cv.objective = 'projection_loss';
cfg.training.mon_grid = [];
cfg.training.n_m_grid = design.estimator_grid_points;
cfg.training.estimator_domain = item.estimator_domain;
cfg.training.mon_domain = item.estimator_domain;
cfg.training.mon_grid_type = 'forward_moneyness_linspace';
cfg.training.exo_grid = [];
cfg.training.n_exo_grid = 1;
cfg.training.exo_grid_type = 'none';

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
cfg.testing.empirical_hac_bandwidth = bartlett_hac_bandwidth(empiricalT);
cfg.testing.inference_lambda.rule = design.inference_rule;
cfg.testing.exo.mode = 'none';
cfg.testing.grid.type = design.shape_grid_type;
cfg.testing.grid.domain = item.test_domain;
cfg.testing.grid.n_points = item.shape_grid_points;

cfg.diagnostics.recovery_T = 2000;
cfg.diagnostics.regularization_lambda_path = [1e-2;1e-3;1e-4;1e-5];
cfg.diagnostics.run_per_rep_regularization_path = false;
cfg.parallel.use = false;
cfg.parallel.nWorkers = [];
cfg.revision.id = sprintf('%s_%s_%s', design.revision, experiment, horizon);

end

function [dt, cvMinTrain] = horizon_design(horizon)
%% Return maturity and CV start for one horizon.

switch horizon
    case 'daily', dt=1/252; cvMinTrain=1260;
    case 'weekly', dt=5/252; cvMinTrain=260;
    case 'monthly', dt=1/12; cvMinTrain=60;
    case 'quarterly', dt=63/252; cvMinTrain=20;
end

end
