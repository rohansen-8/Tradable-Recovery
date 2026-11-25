function fingerprint = build_design_fingerprint(cfg, Tvals, B, nMC)
%% Build strict checkpoint metadata for one final inference design.

Tvals = double(Tvals(:).');
assert(all(isfinite(Tvals) & Tvals >= 1 & Tvals == floor(Tvals)));
assert(isscalar(B) && B >= 1 && B == floor(B));
assert(isscalar(nMC) && nMC >= 1 && nMC == floor(nMC));
assert(isfield(cfg, 'revision') && isfield(cfg.revision, 'id'));

fingerprint = struct();
fingerprint.experiment = char(string(cfg.sim.experiment));
assert(ismember(fingerprint.experiment, ...
    {'bs_increasing_t','bs_empirical_t','chj_increasing_t'}), ...
    'Unknown experiment %s.', fingerprint.experiment);
if isfield(cfg.sim,'horizon'), fingerprint.horizon = char(string(cfg.sim.horizon));
else, fingerprint.horizon = 'daily'; end
fingerprint.T = Tvals;
fingerprint.B = double(B);
fingerprint.generation_domain = double(cfg.sim.generation_domain(:).');
fingerprint.estimator_domain = double(cfg.training.estimator_domain(:).');
fingerprint.test_domain = double(cfg.testing.grid.domain(:).');
fingerprint.J = double(cfg.sim.J);
if isfield(cfg.sim,'option_m_grid')
    fingerprint.strike_vector = double(cfg.sim.option_m_grid(:).');
else
    fingerprint.strike_vector = linspace(cfg.sim.m_min,cfg.sim.m_max,cfg.sim.J);
end
fingerprint.estimator_grid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2), cfg.training.n_m_grid);
fingerprint.estimator_grid_points = double(cfg.training.n_m_grid);
fingerprint.kernel_jitter = double(cfg.kernel.jitter);
fingerprint.G = double(cfg.testing.grid.n_points);
fingerprint.grid_type = char(string(cfg.testing.grid.type));
fingerprint.region_bounds = vertcat(cfg.testing.regionSpecs.bounds);
% For 'cv' the fingerprint contains the fitting lambda grid; for 'eff_dim' it
% contains lambda_method, effdim_c and effdim_target; for 'fixed_lambda' it
% contains the supplied lambda. Checkpoints of different methods can
% therefore never resume one another.
lambdaMethod = 'cv';
if isfield(cfg.training, 'lambda_method') && ~isempty(cfg.training.lambda_method)
    lambdaMethod = char(string(cfg.training.lambda_method));
end
switch lambdaMethod
    case 'cv'
        fingerprint.fitting_lambda_grid = double(cfg.training.cv.lambda_grid(:).');
    case 'eff_dim'
        fingerprint.lambda_method = 'eff_dim';
        fingerprint.effdim_c = double(cfg.training.effdim_c);
        fingerprint.effdim_target = 'effdim_c*T^(1/3)';
    case 'fixed_lambda'
        % The supplied lambda is part of the design, so a checkpoint can never
        % resume under a different value (apply_fixed_lambda binds it).
        lam = cfg.training.lambda;
        assert(isscalar(lam) && isfinite(lam) && lam > 0, ...
            'build_design_fingerprint:FixedLambdaUnbound', ...
            'fixed_lambda designs need one positive lambda bound by apply_fixed_lambda.');
        fingerprint.lambda_method = 'fixed_lambda';
        fingerprint.fixed_lambda = double(lam);
    otherwise
        error('build_design_fingerprint:UnknownLambdaMethod', ...
            'Unknown simulation lambda method %s.', lambdaMethod);
end
fingerprint.inference_rule = char(string(cfg.testing.inference_lambda.rule));
fingerprint.hac_rule = char(string(cfg.testing.hac_bandwidth_rule));
fingerprint.nMC = double(nMC);
fingerprint.design_revision = char(string(cfg.revision.id));
fingerprint.dgp_revision = char(string(cfg.revision.id));

end
