function fingerprint = build_empirical_design_fingerprint(cfg, model_state, nMC)
%% Build the strict empirical fit-and-inference checkpoint fingerprint.
%
% A checkpoint is reusable only when every design choice matches this
% structure exactly (assert_design_fingerprint), so silent design drift can
% never reuse results from a different design.
%
% Method-specific fields. For lambda_method 'cv' (also the default when
% cfg.lambda_method is absent) the fingerprint contains lambda_grid,
% cv_method, cv_min_train_dates and cv_loss_rows; for 'eff_dim' it contains
% lambda_method, effdim_c and effdim_target instead. Checkpoints of the two
% methods can therefore never be confused.

fingerprint = struct();
fingerprint.pipeline = 'empirical_shape_tests';
fingerprint.horizon = char(string(cfg.real.horizon));
fingerprint.T = double(model_state.T);
fingerprint.estimator_domain = double(cfg.training.mon_domain(:).');
fingerprint.estimator_grid = double(model_state.mon_grid(:).');
fingerprint.exogenous_domain = double(cfg.training.exo_domain(:).');
fingerprint.exogenous_grid = double(model_state.exo_grid(:).');
fingerprint.test_domain = double(cfg.testing.grid.domain(:).');
fingerprint.G = double(cfg.testing.grid.n_points);
fingerprint.grid_type = char(string(cfg.testing.grid.type));
fingerprint.region_bounds = vertcat(cfg.testing.regionSpecs.bounds);
lambdaMethod = 'cv';
if isfield(cfg, 'lambda_method') && ~isempty(cfg.lambda_method)
    lambdaMethod = char(string(cfg.lambda_method));
end
isEffDim = strcmp(lambdaMethod, 'eff_dim');
assert(isEffDim == (isfield(model_state, 'lambda_method') ...
    && strcmp(model_state.lambda_method, 'eff_dim')), ...
    'build_empirical_design_fingerprint:MethodMismatch', ...
    'cfg.lambda_method (%s) does not match the saved fitted state.', lambdaMethod);

fingerprint.lambda_fit = double(model_state.lambda);
if ~isEffDim
    fingerprint.lambda_grid = double(cfg.training.cv.lambda_grid(:).');
end
fingerprint.inference_rule = char(string(cfg.testing.inference_lambda.rule));
fingerprint.lengthscale_rule = char(string(cfg.training.lengthscale_rule));
fingerprint.ls_endo = double(model_state.ls_endo);
fingerprint.ls_exo = double(model_state.ls_exo);
fingerprint.kernel_jitter = double(cfg.kernel.jitter);
if isEffDim
    fingerprint.lambda_method = 'eff_dim';
    fingerprint.effdim_c = double(model_state.effdim.c);
    fingerprint.effdim_target = double(model_state.effdim.target);
else
    fingerprint.cv_method = char(string(cfg.training.cv.method));
    fingerprint.cv_min_train_dates = double(cfg.training.cv.min_train_dates);
    fingerprint.cv_loss_rows = double(size(model_state.cv.loss_by_date, 1));
end
fingerprint.shape_tests = cellstr(string(cfg.testing.testTypes(:)));
fingerprint.shape_sign_convention = ...
    'mono=h_m; conv=-h_mm; conc=h_mm; forward_moneyness';
fingerprint.conditional_levels_rule = ...
    char(string(cfg.testing.conditional_levels_rule));
fingerprint.covariance_type = 'hac_bartlett';
fingerprint.hac_rule = char(string(cfg.testing.hac_bandwidth_rule));
fingerprint.covariance_rank_tolerance = 1e-10;
fingerprint.equal_date_mode = 'equal_date_clipped_to_fitted_exogenous_grid';
fingerprint.nMC = double(nMC);
fingerprint.design_revision = char(string(cfg.revision.id));

end
