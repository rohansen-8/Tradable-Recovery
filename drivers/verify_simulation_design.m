function verification = verify_simulation_design()
%% Verify the pre-specified maturity-aligned simulation and empirical test design.
%
% Checked before any production simulation starts. In the simulation
% design each study aligns its three forward-moneyness
% objects on one common domain: generated option observations, the 30-point
% estimator basis (plus reference m = 1), and the midpoint shape-test grid
% (G = 52 for the increasing-T studies, G = 60 for bs_empirical_t). The two
% increasing-T studies share [0.95, 1.05]; bs_empirical_t uses
% exp(figure_log_range) for each maturity. The empirical application keeps
% its three distinct domains.

setpath();
design = get_frozen_design();
expectedT = [500 1000 2000 5000 10000];
assert(strcmp(design.revision,'public_v1'));
assert(numel(design.lambda_grid)==31 && ...
    abs(design.lambda_grid(1)-1e-5)<=1e-19 && ...
    abs(design.lambda_grid(end)-1e-2)<=1e-16 && all(diff(design.lambda_grid)>0));
assert(design.shape_grid_points==60 && design.estimator_grid_points==30);
assert(design.nMC==10000 && strcmp(design.inference_rule,'lambda_fit'));

% Fixed design constants.
assert(isequal(design.bs_increasing_t.domain,[0.95 1.05]));
assert(isequal(design.chj_increasing_t.domain,[0.95 1.05]));
assert(isequal(design.bs_increasing_t.region_cutoffs,[0.985 1.015]));
assert(isequal(design.chj_increasing_t.region_cutoffs,[0.985 1.015]));
assert(design.bs_increasing_t.B==1000 && design.chj_increasing_t.B==1000);
assert(design.bs_empirical_t.B==2000);
assert(design.bs_increasing_t.J==30 && design.chj_increasing_t.J==30);
assert(design.bs_increasing_t.G==52 && design.chj_increasing_t.G==52);
assert(design.bs_empirical_t.G==design.shape_grid_points);

rows = cell(14,1);
monthly = get_empirical_maturity_config('monthly');
itemA = get_experiment_design('bs_increasing_t','monthly');
cfgA = config_bs_simulation_design('monthly','bs_increasing_t');
assert(isequal(cfgA.sim.performance_T,expectedT));
% The 30 equally spaced strikes on [0.95, 1.05] straddle m = 1: 15 puts and
% 15 calls, with no contract on the ambiguous ATM point itself.
verify_cfg(cfgA,itemA,30,'forward_moneyness_linspace',1000,0);
verify_region_counts(cfgA,[18 16 18]);
verify_lengthscale_equivalence(cfgA,expectedT);
for j=1:numel(expectedT)
    rows{j}=manifest_row("bs_increasing_t","monthly",expectedT(j),cfgA,itemA);
end

horizons = design.bs_empirical_t.horizons;
expectedJ = [46;77;27;28];
empiricalT = zeros(numel(horizons),1);
for j=1:numel(horizons)
    maturity = get_empirical_maturity_config(horizons(j));
    itemB = get_experiment_design('bs_empirical_t',horizons(j));
    cfgB = config_bs_simulation_design(horizons(j),'bs_empirical_t');
    verify_cfg(cfgB,itemB,expectedJ(j),'forward_moneyness_linspace',2000,0);
    assert(isequal(itemB.generation_domain,exp(maturity.figure_log_range)));
    assert(isequal(itemB.region_cutoffs,maturity.region_cutoffs));
    assert(cfgB.sim.empirical_T==maturity.T);
    assert(cfgB.sim.strike_design.n_dates==maturity.T);
    assert(strcmp(cfgB.sim.strike_design.count_rule, ...
        'median_date_strike_count_inside_simulation_domain'));
    assert(round(cfgB.sim.strike_design.median_count_before_rounding)==expectedJ(j));
    empiricalT(j)=maturity.T;
    rows{5+j}=manifest_row("bs_empirical_t",horizons(j),maturity.T,cfgB,itemB);
end

daily = get_empirical_maturity_config('daily');
itemC = get_experiment_design('chj_increasing_t','daily');
cfgC = config_chj_simulation_design();
verify_cfg(cfgC,itemC,30,'forward_moneyness_linspace',1000,0);
verify_region_counts(cfgC,[18 16 18]);
assert(isequal(cfgC.sim.performance_T,expectedT));
assert(cfgC.training.n_exo_grid==30 && ...
    isequal(cfgC.training.exo_support_quantiles,[0.05 0.95]) && ...
    strcmp(cfgC.training.exo_grid_type,'replication_q05_q95_linspace'));
for j=1:numel(expectedT)
    rows{9+j}=manifest_row("chj_increasing_t","daily",expectedT(j),cfgC,itemC);
end

expectedBandwidth = [5 6 7 9 11 3 5 6 7 5 6 7 9 11];
actualBandwidth = arrayfun(@bartlett_hac_bandwidth, ...
    [expectedT empiricalT.' expectedT]);
assert(isequal(actualBandwidth,expectedBandwidth));

% The empirical application keeps its own domains, regions and kernel settings.
for j=1:numel(horizons)
    maturity = get_empirical_maturity_config(horizons(j));
    cfgE = config_empirical(horizons(j));
    assert(cfgE.testing.grid.n_points==60);
    assert(isequal(cfgE.testing.grid.domain,maturity.test_domain));
    assert(isequal(cfgE.testing.regionBounds,maturity.region_bounds));
    assert(isequal(cfgE.training.mon_domain,maturity.estimator_domain));
    assert(strcmp(cfgE.training.lengthscale_rule,design.lengthscale_rule));
    assert(cfgE.kernel.jitter==design.jitter);
end
assert(isequal(daily.support_domain, ...
    [0.967264263145474, 1.0208603191431]), ...
    'Empirical daily support constants must not drift.');
assert(isequal(monthly.support_domain, ...
    [0.744456334784572, 1.09089309666474]), ...
    'Empirical monthly support constants must not drift.');

verification = cat(1,rows{:});
fprintf('\nFROZEN SIMULATION DESIGN VERIFIED.\n');
disp(verification);

end

function verify_cfg(cfg,item,J,optionType,expectedB,nAtUnitMoneyness)
%% Verify one simulation design against its resolved experiment row.

design = get_frozen_design();
domain = item.generation_domain;

% One common simulation domain for generation, estimation, and inference.
assert(isequal(cfg.sim.generation_domain,domain));
assert(isequal([cfg.sim.m_min cfg.sim.m_max],domain));
assert(isequal(cfg.training.mon_domain,domain));
assert(isequal(cfg.training.estimator_domain,domain));
assert(isequal(cfg.testing.grid.domain,domain));
assert(isequal(item.estimator_domain,domain) && isequal(item.test_domain,domain));

assert(cfg.sim.J==J && numel(cfg.sim.option_m_grid)==J);
assert(strcmp(cfg.sim.option_grid_type,optionType));
assert(cfg.sim.final_B==expectedB);
assert(cfg.training.n_m_grid==30);
assert(cfg.testing.grid.n_points==item.shape_grid_points);
assert(strcmp(cfg.testing.grid.type,'global_midpoints'));
assert(isequal(vertcat(cfg.testing.regionSpecs.bounds),item.region_bounds));
assert(isequal(cfg.training.cv.lambda_grid,design.lambda_grid));
assert(isfield(cfg.kernel,'jitter') && cfg.kernel.jitter==design.jitter);
assert(strcmp(cfg.testing.inference_lambda.rule,'lambda_fit'));
assert(~isfield(cfg.testing.inference_lambda,'critical_value'));
assert(cfg.testing.nMC==10000);
assert(strcmp(cfg.testing.hac_bandwidth_rule,'nw_4_t100_2over9'));
assert(contains(cfg.testing.shapeConvention,'conv:-h_mm'));
assert(~contains(cfg.testing.shapeConvention,['M=1' '-h']));
assert(contains(cfg.revision.id,design.revision));

% Simulated locations are equally spaced across the whole domain. Contracts
% strictly below m = 1 are puts and strictly above are calls; the expected
% number of locations exactly at m = 1, where the OTM side is a convention
% rather than an implication, is stated per study.
assert(all(diff(cfg.sim.option_m_grid)>0));
assert(sum(abs(cfg.sim.option_m_grid-1)<=1e-12)==nAtUnitMoneyness, ...
    ['%s places %d simulated contracts exactly at m = 1; the finalized ' ...
     'design expects %d.'], cfg.sim.experiment, ...
    sum(abs(cfg.sim.option_m_grid-1)<=1e-12), nAtUnitMoneyness);
% Only the increasing-T domain is symmetric about m = 1, so only there does
% the even strike count split evenly into puts and calls.
if contains(cfg.sim.experiment, 'increasing_t')
    assert(sum(cfg.sim.option_m_grid<1)==J/2 && ...
        sum(cfg.sim.option_m_grid>1)==J/2, ...
        '%s must place %d puts and %d calls around m = 1.', ...
        cfg.sim.experiment, J/2, J/2);
end
assert(cfg.sim.option_m_grid(1)==domain(1));
assert(cfg.sim.option_m_grid(end)==domain(2));
expectedGrid = linspace(domain(1),domain(2),J).';
assert(isequal(cfg.sim.option_m_grid(:),expectedGrid));

estimatorGrid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2),30).';
assert(estimatorGrid(1)==domain(1) && estimatorGrid(end)==domain(2));
% The separate reference point m = 1 must not duplicate a basis node.
assert(~any(abs(estimatorGrid-cfg.reference_point)<=1e-12));

G = cfg.testing.grid.n_points;
m = build_global_shape_grid(cfg);
dm = (domain(2)-domain(1))/G;
expectedMidpoints = domain(1)+((0:G-1).'+0.5)*dm;
assert(isequal(m,expectedMidpoints));
grids = build_region_grids(cfg);
% Every test point belongs to exactly one region.
counts = arrayfun(@(g)numel(g.m),grids);
assert(sum(counts)==G && all(counts>0));
assert(isequal(sort(vertcat(grids.m)),m));
cut = item.region_cutoffs;
assert(~any(abs(m-cut(1))<=1e-12) && ~any(abs(m-cut(2))<=1e-12), ...
    'No shape-test point may land exactly on a region cutoff.');

end

function verify_region_counts(cfg,expected)
%% Assert the exact regional split of the midpoint shape grid.

grids = build_region_grids(cfg);
counts = arrayfun(@(g)numel(g.m),grids);
assert(isequal(counts(:).',expected(:).'), ...
    '%s region counts are %s; the frozen design expects %s.', ...
    cfg.sim.experiment, mat2str(counts(:).'), mat2str(expected(:).'));

end

function verify_lengthscale_equivalence(cfg,Tvals)
%% Require the realized BS moneyness lengthscale to be option_level_median.
%
% The lengthscale rule is option_level_median: the median pairwise distance over
% option-level rows, that is every strike repeated on each of the T dates.
% build_bs_alpha_precompute evaluates it through
% median_pairwise_lengthscale_weighted with multiplicity T, and the CHJ study
% makes the same call on its own option grid, so on the shared increasing-T
% grid the two studies must land on one value. This asserts the realized
% numbers; it defines no rule.

grid = cfg.sim.option_m_grid(:);
monGrid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
for k = 1:numel(Tvals)
    optionLevel = median_pairwise_lengthscale_weighted(grid, ...
        repmat(Tvals(k),numel(grid),1));
    cfgT = cfg;
    cfgT.sim.T = Tvals(k);
    pre = build_bs_alpha_precompute(cfgT, monGrid);
    assert(pre.ell_m == optionLevel, ...
        ['%s at T=%d: the realized moneyness lengthscale %.17g is not the ' ...
         'option_level_median %.17g.'], cfg.sim.experiment, Tvals(k), ...
        pre.ell_m, optionLevel);
end

% The CHJ study manifest must report the same value on the same grid.
chjLengthscales = unique(get_chj_study_design().kernel_endo_lengthscale);
assert(isscalar(chjLengthscales) && chjLengthscales == ...
    median_pairwise_lengthscale_weighted(grid, repmat(Tvals(1),numel(grid),1)), ...
    'CHJ does not realize option_level_median on the shared increasing-T grid.');

end

function row = manifest_row(experiment,horizon,T,cfg,item)
%% Build one concise verification row.

m = build_global_shape_grid(cfg);
counts = arrayfun(@(s)sum(m>=s.bounds(1) & m<=s.bounds(2)), ...
    cfg.testing.regionSpecs);
row = table(string(experiment),string(horizon),T,cfg.sim.final_B, ...
    cfg.sim.m_min,cfg.sim.m_max,cfg.sim.J,cfg.training.n_m_grid, ...
    cfg.testing.grid.n_points,counts(1),counts(2),counts(3), ...
    string(cfg.sim.option_grid_type),string(item.domain_source), ...
    cfg.testing.nMC,string(cfg.revision.id), ...
    'VariableNames',{'experiment','horizon','T','B','gen_m_min','gen_m_max','J', ...
    'estimator_grid_points','G','n_left','n_center','n_right', ...
    'option_grid_type','domain_source','nMC','revision_id'});

end
