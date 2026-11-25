function item = get_experiment_design(experiment, horizon)
%% Return one simulation-experiment/horizon row of the simulation design.
%
% Three forward-moneyness domains are reported separately so downstream code
% keeps using them for their distinct purposes:
%   generation_domain : where simulated option observations are placed;
%   estimator_domain  : the 30-point RKHS basis support (plus reference m=1);
%   test_domain       : the midpoint shape-test support (G per experiment).
% All three are set to one common
% experiment-specific domain: [0.95, 1.05] for the two increasing-T studies,
% and exp(figure_log_range) for each bs_empirical_t maturity. This is a
% simulation-only alignment; the empirical application keeps the three
% distinct domains configured in get_empirical_maturity_config.

design = get_frozen_design();
experiment = validatestring(char(string(experiment)), ...
    {'bs_increasing_t','bs_empirical_t','chj_increasing_t'}, ...
    mfilename, 'experiment');
horizon = validatestring(char(string(horizon)), ...
    {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon');

item = struct();
item.experiment = experiment;
item.horizon = horizon;
item.lambda_grid = design.lambda_grid;
item.inference_rule = design.inference_rule;
item.hac_rule = design.hac_rule;
item.nMC = design.nMC;
item.estimator_grid_points = design.estimator_grid_points;
item.shape_grid_type = design.shape_grid_type;
item.shape_tests = design.shape_tests;
item.shape_convention = design.shape_convention;
item.revision = design.revision;

maturity = get_empirical_maturity_config(horizon);
item.T_empirical = maturity.T;

switch experiment
    case 'bs_increasing_t'
        assert(strcmp(horizon, 'monthly'), ...
            'bs_increasing_t is defined only for the monthly horizon.');
        src = design.bs_increasing_t;
        item.T = src.T;
        item.J = src.J;
        item.option_grid_type = src.option_grid_type;
        item.B = src.B;
        item.shape_grid_points = src.G;
        domain = double(src.domain(:).');
        cutoffs = double(src.region_cutoffs(:).');
        item.domain_source = 'frozen_common_increasing_t_domain';
    case 'chj_increasing_t'
        assert(strcmp(horizon, 'daily'), ...
            'chj_increasing_t is defined only for the daily horizon.');
        src = design.chj_increasing_t;
        item.T = src.T;
        item.J = src.J;
        item.option_grid_type = src.option_grid_type;
        item.B = src.B;
        item.shape_grid_points = src.G;
        domain = double(src.domain(:).');
        cutoffs = double(src.region_cutoffs(:).');
        item.domain_source = 'frozen_common_increasing_t_domain';
    otherwise
        src = design.bs_empirical_t;
        item.T = maturity.T;
        item.J = NaN;   % derived from the empirical panel by the strike design
        item.option_grid_type = src.option_grid_type;
        item.B = src.B;
        item.shape_grid_points = src.G;
        domain = exp(double(maturity.figure_log_range(:).'));
        cutoffs = double(maturity.region_cutoffs(:).');
        item.domain_source = src.domain_rule;
end

% One common simulation domain for generation, estimation, and inference.
item.generation_domain = domain;
item.estimator_domain = domain;
item.test_domain = domain;
item.region_cutoffs = cutoffs;
item.region_bounds = [domain(1), cutoffs(1); cutoffs(1), cutoffs(2); ...
    cutoffs(2), domain(2)];

assert(numel(domain) == 2 && all(isfinite(domain)) && domain(1) < domain(2), ...
    '%s domain must contain two increasing finite endpoints.', experiment);
assert(domain(1) < cutoffs(1) && cutoffs(1) < cutoffs(2) && ...
    cutoffs(2) < domain(2), ...
    '%s %s region cutoffs must lie strictly inside the simulation domain.', ...
    experiment, horizon);

end
