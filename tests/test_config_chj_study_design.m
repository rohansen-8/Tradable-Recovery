function test_config_chj_study_design()
%% Verify the fixed CHJ structural configuration and common tuning rules.

cfg = config_chj_simulation_design();

assert(cfg.sim.horizon_days == 1);
assert(isequal(cfg.sim.performance_T, [500 1000 2000 5000 10000]));
assert(cfg.sim.final_B == 1000);
assert(cfg.sim.J == 30);
% All three simulation supports coincide on the fixed increasing-T domain.
assert(isequal([cfg.sim.m_min cfg.sim.m_max], [0.95 1.05]));
assert(isequal(cfg.training.estimator_domain, [0.95 1.05]));
assert(strcmp(cfg.sim.option_grid_type, 'forward_moneyness_linspace'));
assert(~isfield(cfg.sim, 'log_m_min') && ~isfield(cfg.sim, 'log_m_max'));

assert(isempty(cfg.kernel.endo.lengthscale));
assert(strcmp(cfg.kernel.endo.lengthscale_rule, 'median'));
assert(isempty(cfg.kernel.exo.lengthscale));
assert(strcmp(cfg.kernel.exo.lengthscale_rule, 'median'));

assert(isempty(cfg.training.lambda));
assert(cfg.training.use_cv);
assert(strcmp(cfg.training.cv.method, '1_step_ahead'));
assert(strcmp(cfg.training.cv.objective, 'projection_loss'));
assert(strcmp(cfg.training.cv.selection_rule, 'mean_loss'));
assert(strcmp(cfg.training.cv.solver, 'fold_eig_woodbury'));
assert(cfg.training.cv.min_train_dates == 10);
assert(cfg.training.cv.woodbury_refresh == 40);
assert(numel(cfg.training.cv.lambda_grid) == 31);
assert(abs(min(cfg.training.cv.lambda_grid) - 1e-5) <= 1e-12);
assert(abs(max(cfg.training.cv.lambda_grid) - 1e-2) <= 1e-12);

assert(abs(cfg.reference_point - 1) <= 1e-14);
assert(cfg.training.n_m_grid == 30);
assert(strcmp(cfg.training.mon_grid_type, 'forward_moneyness_linspace'));
assert(cfg.training.n_exo_grid == 30);
assert(strcmp(cfg.training.exo_grid_type, 'replication_q05_q95_linspace'));
assert(isequal(cfg.training.exo_support_quantiles, [0.05, 0.95]));
assert((cfg.training.n_m_grid + 1) * cfg.training.n_exo_grid == 930);
assert(strcmp(cfg.testing.exo.mode, 'equal_date'));
assert(cfg.testing.exo.clip_to_grid);
assert(strcmp(cfg.testing.hac_bandwidth_rule, 'nw_4_t100_2over9'));
assert(strcmp(cfg.testing.inference_lambda.rule, 'lambda_fit'));
assert(~cfg.diagnostics.run_per_rep_regularization_path);
assert(isequal(cfg.testing.testTypes, {'mono', 'conv', 'conc'}));
assert(strcmp(cfg.testing.grid.type, 'global_midpoints'));
assert(isequal(cfg.testing.grid.domain, [0.95 1.05]));
assert(isequal(vertcat(cfg.testing.regionSpecs.bounds), ...
    [0.95 0.985; 0.985 1.015; 1.015 1.05]));
assert(cfg.testing.grid.n_points == 52);

fprintf('test_config_chj_study_design passed\n');

end
