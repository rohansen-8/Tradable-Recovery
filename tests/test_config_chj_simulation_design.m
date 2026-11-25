function tests = test_config_chj_simulation_design
%% Verify the CHJ simulation configuration.

tests = functiontests(localfunctions);

end

function setupOnce(testCase)
%% Add configuration and shared-rule folders for this unit-test file.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(rootDir, 'configs'));
addpath(fullfile(rootDir, 'testing'));
addpath(fullfile(rootDir, 'training'));

end

function testExperimentTable(testCase)
%% Verify the pre-specified five-sample-size CHJ study sequence.

cfg = config_chj_simulation_design();
% Common increasing-T simulation domain.
domain = [0.95 1.05];

verifyEqual(testCase, cfg.sim.model, 'CHJ');
verifyEqual(testCase, cfg.sim.design, 'chj_increasing_t');
verifyEqual(testCase, cfg.sim.seed0, 500001);
verifyEqual(testCase, cfg.sim.horizon_days, 1);
verifyEqual(testCase, cfg.sim.m_min, domain(1), 'AbsTol', 1e-14);
verifyEqual(testCase, cfg.sim.m_max, domain(2), 'AbsTol', 1e-14);
verifyEqual(testCase, cfg.sim.option_grid_type, 'forward_moneyness_linspace');
verifyEqual(testCase, cfg.exp_table.Tval, [500; 1000; 2000; 5000; 10000]);
verifyEqual(testCase, cfg.exp_table.nSeeds, repmat(1000, 5, 1));
verifyTrue(testCase, isempty(cfg.training.lambda));
verifyTrue(testCase, cfg.training.use_cv);
verifyEqual(testCase, cfg.training.cv.min_train_dates, 10);
end

function testRegionsAndTests(testCase)
%% Verify the regional nulls and common inference mechanics.

cfg = config_chj_simulation_design();
domain = [0.95 1.05];

verifyEqual(testCase, vertcat(cfg.testing.regionSpecs.bounds), ...
    [0.95 0.985; 0.985 1.015; 1.015 1.05]);
verifyEqual(testCase, cfg.testing.testTypes, {'mono', 'conv', 'conc'});
verifyEqual(testCase, cfg.testing.alpha, 0.05);
verifyEqual(testCase, cfg.testing.nMC, 10000);
verifyEqual(testCase, cfg.testing.hac_bandwidth_rule, 'nw_4_t100_2over9');
verifyEqual(testCase, cfg.testing.exo.mode, 'equal_date');
verifyTrue(testCase, cfg.testing.exo.clip_to_grid);
verifyEqual(testCase, cfg.testing.grid.type, 'global_midpoints');
verifyEqual(testCase, cfg.testing.grid.domain, domain, 'AbsTol', 1e-14);
verifyEqual(testCase, cfg.testing.grid.n_points, 52);
end

function testKernelRules(testCase)
%% Verify that both CHJ lengthscales are rule-generated rather than fixed.

cfg = config_chj_simulation_design();

verifyTrue(testCase, isempty(cfg.kernel.endo.lengthscale));
verifyEqual(testCase, cfg.kernel.endo.lengthscale_rule, 'median');
verifyTrue(testCase, isempty(cfg.kernel.exo.lengthscale));
verifyEqual(testCase, cfg.kernel.exo.lengthscale_rule, 'median');
verifyEqual(testCase, cfg.training.n_m_grid, 30);
verifyEqual(testCase, cfg.training.mon_grid_type, 'forward_moneyness_linspace');
verifyEqual(testCase, cfg.training.n_exo_grid, 30);
verifyEqual(testCase, cfg.training.exo_grid_type, 'replication_q05_q95_linspace');
verifyEqual(testCase, cfg.training.exo_support_quantiles, [0.05, 0.95]);
verifyEqual(testCase, (cfg.training.n_m_grid + 1) * cfg.training.n_exo_grid, 930);
end

function testCHJParameters(testCase)
%% Verify the parameters of the benchmark CHJ economy.

cfg = config_chj_simulation_design();

verifyEqual(testCase, cfg.chj.mu,    0.08 / 252, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.h0,    (0.20^2) / 252, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.omega, 1e-6, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.beta,  0.88, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.alpha, 5e-4, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.gamma, 0.5, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.xi,    0.9, 'AbsTol', 1e-15);
verifyEqual(testCase, cfg.chj.r,     0.02 / 252, 'AbsTol', 1e-15);
end
