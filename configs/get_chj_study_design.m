function design = get_chj_study_design()
%% Return the five-sample-size CHJ structural study design.
%
% CHJ option maturity is one trading day. The study varies only the retained
% sample-size sequence; generated observations use the common increasing-T
% forward-moneyness domain at every T.

cfg = config_chj_simulation_design();
Tvals = cfg.sim.performance_T(:);
n = numel(Tvals);

option_m_grid = chj_option_moneyness_grid(cfg);
ls_endo = NaN(n, 1);
for j = 1:n
    ls_endo(j) = median_pairwise_lengthscale_weighted( ...
        option_m_grid, repmat(Tvals(j), numel(option_m_grid), 1));
end

design = table();
design.design_id = (1:n).';
design.design_group = repmat("daily_one_day_T_sequence", n, 1);
design.Tval = Tvals;
design.horizon_days = repmat(cfg.sim.horizon_days, n, 1);
design.option_panel_m_points = repmat(cfg.sim.J, n, 1);
design.option_panel_m_grid_type = repmat(string(cfg.sim.option_grid_type), n, 1);
design.option_panel_m_min = repmat(cfg.sim.m_min, n, 1);
design.option_panel_m_max = repmat(cfg.sim.m_max, n, 1);
design.mon_grid_min = repmat(cfg.training.estimator_domain(1), n, 1);
design.mon_grid_max = repmat(cfg.training.estimator_domain(2), n, 1);
design.mon_grid_type = repmat(string(cfg.training.mon_grid_type), n, 1);
design.cv_min_train_dates = repmat(cfg.training.cv.min_train_dates, n, 1);
design.lambda_mode = repmat("cv", n, 1);
design.lambda_grid_n = repmat(numel(cfg.training.cv.lambda_grid), n, 1);
design.lambda_grid_min = repmat(min(cfg.training.cv.lambda_grid), n, 1);
design.lambda_grid_max = repmat(max(cfg.training.cv.lambda_grid), n, 1);
design.kernel_endo_lengthscale_rule = repmat(string(cfg.kernel.endo.lengthscale_rule), n, 1);
design.kernel_endo_lengthscale = ls_endo;
design.kernel_exo_lengthscale_rule = repmat(string(cfg.kernel.exo.lengthscale_rule), n, 1);
design.kernel_exo_lengthscale = NaN(n, 1);
design.n_m_grid = repmat(cfg.training.n_m_grid, n, 1);
design.exo_grid_type = repmat(string(cfg.training.exo_grid_type), n, 1);
design.n_exo_grid = repmat(cfg.training.n_exo_grid, n, 1);
design.exo_grid_min = NaN(n, 1);
design.exo_grid_max = NaN(n, 1);
design.basis_dimension = repmat((cfg.training.n_m_grid + 1) * cfg.training.n_exo_grid, n, 1);
design.vix_units = repmat("annualized_volatility", n, 1);
design.exo_average_mode = repmat(string(cfg.testing.exo.mode), n, 1);
design.exo_clip_to_grid = repmat(logical(cfg.testing.exo.clip_to_grid), n, 1);
design.test_grid_type = repmat(string(cfg.testing.grid.type), n, 1);
design.test_grid_n_points = repmat(cfg.testing.grid.n_points, n, 1);
design.test_grid_min = repmat(cfg.testing.grid.domain(1), n, 1);
design.test_grid_max = repmat(cfg.testing.grid.domain(2), n, 1);
design.hac_bandwidth_rule = repmat(string(cfg.testing.hac_bandwidth_rule), n, 1);
design.hac_bandwidth = arrayfun(@bartlett_hac_bandwidth, Tvals);
design.inference_lambda_rule = repmat(string(cfg.testing.inference_lambda.rule), n, 1);
design.regularization_lambda_path = repmat(string(mat2str(cfg.diagnostics.regularization_lambda_path.')), n, 1);
design.seed0 = cfg.sim.seed0 + 1000000 * (design.design_id - 1);
design.revision_id = repmat(string(cfg.revision.id), n, 1);

end
