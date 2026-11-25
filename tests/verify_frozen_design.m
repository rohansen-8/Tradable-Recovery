function report = verify_frozen_design()
%% Verify the complete pre-specified design against the configuration and README.md.
%
% Run before production. It checks, without fitting anything:
%   - the global design scalars (get_frozen_design);
%   - the per-maturity domain tables (get_empirical_maturity_config),
%     including the recomputed 20%-coverage support and the
%     figure-intersect-support test-domain rule from the data panels;
%   - the empirical configuration (estimator basis, exogenous grids,
%     option-level lengthscale rule, per-factor jitter, CV design, test
%     grids, regions, lambda_inference = lambda_fit);
%   - all three simulation experiments (three-domain separation, J rules,
%     OTM side selection, exogenous-grid rules);
%   - the shape-restriction sign conventions in build_theta_hat;
%   - agreement of README.md with the numeric domains.
% It fails loudly on any drift.

setpath();
design = get_frozen_design();

%% Global scalars
assert(strcmp(design.kernel_endo,'matern72') && strcmp(design.kernel_exo,'laplace'));
assert(strcmp(design.lengthscale_rule,'option_level_median'));
assert(design.jitter == 1e-10);
assert(design.reference_point == 1);
assert(design.estimator_grid_points == 30 && design.shape_grid_points == 60);
assert(strcmp(design.loss,'projection_loss'));
assert(isequal(design.lambda_grid(:), logspace(-5,-2,31).'));
assert(strcmp(design.cv_method,'1_step_ahead'));
assert(strcmp(design.inference_rule,'lambda_fit'));
assert(ismember(design.empirical_lambda_method, {'eff_dim','cv'}));
assert(strcmp(design.empirical_lambda_method,'eff_dim') && design.effdim_c == 1);
assert(design.nMC == 10000 && design.test_alpha == 0.05);
assert(strcmp(design.hac_rule,'nw_4_t100_2over9'));
assert(isequal(design.shape_tests,{'mono','conv','conc'}));
assert(~contains(design.shape_convention,'M=1-h'));
assert(contains(design.shape_convention,'mono:h_m>=0') && ...
    contains(design.shape_convention,'conv:-h_mm>=0') && ...
    contains(design.shape_convention,'conc:h_mm>=0'));
assert(design.vix_selector.n_levels == 5 && ...
    design.vix_selector.n_candidates == 50 && ...
    design.vix_selector.num_eval == 200);
assert(design.figure_plot_points == 300);

%% Per-maturity domains, empirical configuration, and panel-based rules
test_empirical_maturity_domain_identity();
horizons = {'quarterly','monthly','weekly','daily'};
rows = cell(numel(horizons),1);
for j = 1:numel(horizons)
    horizon = horizons{j};
    maturity = get_empirical_maturity_config(horizon);
    cfg = config_empirical(horizon);

    % Estimator basis: fixed ranges, 30 points, reference m = 1.
    assert(isequal(cfg.training.mon_domain, maturity.estimator_domain));
    assert(numel(cfg.training.mon_grid) == 30 && ...
        cfg.training.mon_grid(1) == maturity.estimator_domain(1) && ...
        cfg.training.mon_grid(end) == maturity.estimator_domain(2));
    assert(cfg.reference_point == 1);
    % Exogenous grid: fixed [5,65] except data-driven weekly.
    assert(numel(cfg.training.exo_grid) == 30);
    if maturity.exo_range_is_data_driven
        df = readtable(cfg.dataset.dfFile,'VariableNamingRule','preserve');
        assert(cfg.training.exo_grid(1) == min(df.VIX) && ...
            cfg.training.exo_grid(end) == max(df.VIX));
    else
        assert(isequal([cfg.training.exo_grid(1) cfg.training.exo_grid(end)], ...
            maturity.exo_range));
    end
    % Lengthscales from option-level rows, jitter per factor, fixed CV design.
    assert(strcmp(cfg.training.lengthscale_rule,'option_level_median'));
    assert(cfg.kernel.jitter == design.jitter);
    assert(isequal(cfg.training.cv.lambda_grid(:), design.lambda_grid(:)));
    assert(strcmp(cfg.training.cv.method,'1_step_ahead'));
    assert(strcmp(cfg.training.cv.objective,'projection_loss'));
    % Shape tests: 60 midpoints on the fixed test domain.
    assert(cfg.testing.grid.n_points == 60 && ...
        strcmp(cfg.testing.grid.type,'global_midpoints'));
    assert(isequal(cfg.testing.grid.domain, maturity.test_domain));
    assert(isequal(cfg.testing.regionBounds, maturity.region_bounds));
    assert(strcmp(cfg.testing.inference_lambda.rule,'lambda_fit'));
    assert(strcmp(cfg.testing.conditional_levels_rule,'select_vix_levels'));
    grids = build_region_grids(cfg);
    assert(numel(grids) == 3 && sum(arrayfun(@(g)numel(g.m),grids)) == 60);
    % Support and test domain recomputed from the data panel.
    dom = verify_supported_testing_domain(horizon);
    assert(dom.exact_match, ...
        '%s support/test domain does not exactly equal the panel derivation.', ...
        horizon);
    assert(dom.all_test_points_covered_20pct, ...
        '%s has test-grid points below 20%% date coverage on the live panel.', ...
        horizon);
    assert(dom.cutoffs_inside);
    rows{j} = table(string(horizon), maturity.support_domain(1), ...
        maturity.support_domain(2), maturity.test_domain(1), ...
        maturity.test_domain(2), maturity.estimator_domain(1), ...
        maturity.estimator_domain(2), dom.min_point_coverage, ...
        'VariableNames', {'horizon','support_lo','support_hi','test_lo', ...
        'test_hi','estimator_lo','estimator_hi','min_test_point_coverage'});
end

%% Lengthscale rule: option-level, not grid-based, single upstream source
cfgM = config_empirical('monthly');
dfM = readtable(cfgM.dataset.dfFile,'VariableNamingRule','preserve');
[lsE, lsX] = option_level_median_lengthscales(cfgM, dfM);
lsGrid = median_pairwise_lengthscale(cfgM.training.mon_grid);
assert(abs(lsE - lsGrid) > 1e-3, ...
    'Option-level and grid-based lengthscales must differ; rule check is vacuous.');
assert(lsE > 0 && all(lsX > 0));

%% Per-factor jitter enters K_M and K_V before the Kronecker product
cfg0 = cfgM; cfg0.kernel.endo.lengthscale = lsE; cfg0.kernel.exo.lengthscale = lsX;
[K_h1, K_M1, K_V1] = build_kernel_norm_matrix(cfg0, ...
    cfgM.training.mon_grid, cfgM.training.exo_grid);
cfg0.kernel.jitter = 0;
[~, K_M0, K_V0] = build_kernel_norm_matrix(cfg0, ...
    cfgM.training.mon_grid, cfgM.training.exo_grid);
% The added jitter is exact up to the ULP of each diagonal entry.
assert(max(abs(diag(K_M1-K_M0) - 1e-10)) <= 4*eps(max(abs(diag(K_M1)))));
assert(max(abs(diag(K_V1-K_V0) - 1e-10)) <= 4*eps(max(abs(diag(K_V1)))));
assert(max(max(abs((K_M1-K_M0) - diag(diag(K_M1-K_M0))))) == 0);
assert(max(max(abs(K_h1 - kron(K_V1, K_M1)))) == 0);

%% Shape-restriction signs on a tiny synthetic state
state = struct('mon_grid',[0.9;1.0;1.1],'reference_point',1,'ls_endo',0.15, ...
    'h_star',[0.1;-0.2;0.05;0.03]);
cfgS = struct('has_exo',false);
mChk = linspace(0.86,1.14,7).'; dh = 1e-6;
exo0 = build_exo_representer(cfgS, state);
thMono = build_theta_hat(cfgS, state, mChk, 'mono', exo0);
thConv = build_theta_hat(cfgS, state, mChk, 'conv', exo0);
thConc = build_theta_hat(cfgS, state, mChk, 'conc', exo0);
hp = evaluate_sdf_grid(cfgS, state, mChk+dh, exo0).h;
hm = evaluate_sdf_grid(cfgS, state, mChk-dh, exo0).h;
fdHm = (hp - hm) / (2*dh);
assert(max(abs(thMono(:) - fdHm)) <= 1e-6, 'mono loading is not h_m.');
assert(max(abs(thConv(:) + thConc(:))) <= 1e-14, 'conv must equal -conc.');
hpp = evaluate_sdf_grid(cfgS, state, mChk+2*dh, exo0).h;
hmm = evaluate_sdf_grid(cfgS, state, mChk-2*dh, exo0).h;
h0 = evaluate_sdf_grid(cfgS, state, mChk, exo0).h;
fdHmm = (hpp - 2*h0 + hmm) / (4*dh^2);
assert(max(abs(thConc(:) - fdHmm)) <= 1e-3, 'conc loading is not h_mm.');

%% Simulation experiments
verify_simulation_design();

%% README must document the same design numbers
readme = fileread(fullfile(get_project_root(), 'README.md'));
mustContain = {'logspace(-5,-2,31)','[0.7, 1.4]','[0.7, 1.3]','[0.8, 1.2]', ...
    '[0.56147167769263, 1.18831169511829]', ...
    '[0.744456334784572, 1.09089309666474]', ...
    '[0.917587749971634, 1.03688696897526]', ...
    '[0.967264263145474, 1.0208603191431]','exp(-0.25)', ...
    '0.880, 1.030','0.920, 1.030','0.945, 1.000','0.970, 1.010', ...
    'G = 60','1e-10','lambda_inference = lambda_fit', ...
    'floor(4*(T/100)^(2/9))','nMC = 10000','linspace(5, 65, 30)', ...
    'h_m >= 0','-h_mm >= 0','h_mm >= 0'};
for k = 1:numel(mustContain)
    assert(contains(readme, mustContain{k}), ...
        'README.md is missing frozen design text: %s', mustContain{k});
end
assert(~contains(readme, 'M = 1 - h is the model'), 'README identity check.');

report = cat(1, rows{:});
fprintf('\nFROZEN DESIGN VERIFIED.\n');
disp(report);

end
