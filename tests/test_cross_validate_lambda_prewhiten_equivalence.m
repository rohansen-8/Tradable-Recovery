function test_cross_validate_lambda_prewhiten_equivalence()
%% Verify prewhitened CV against the fold-eigendecomposition implementation.

rng(90210, 'twister');
Nh = 36;
T = 44;
R = randn(Nh);
K_h = R * R.' + 0.5 * eye(Nh);
alpha = randn(Nh, T);
alphaPanel = struct('alpha', alpha);

cfg = config_chj_simulation_design();
cfg.training.cv.min_train_dates = 10;
cfg.training.cv.lambda_grid = logspace(-5, -2, 31).';

cfgOld = cfg;
cfgOld.training.cv.solver = 'fold_eig';
cvOld = cross_validate_lambda(cfgOld, alphaPanel, K_h);

cfgNew = cfg;
cfgNew.training.cv.solver = 'fold_eig_prewhiten';
cvNew = cross_validate_lambda(cfgNew, alphaPanel, K_h);

lossScale = max(1, max(abs(cvOld.mean_loss)));
err = max(abs(cvOld.mean_loss - cvNew.mean_loss));

assert(err <= 1e-10 * lossScale, ...
    'Prewhitened and fold-eig CV losses differ materially.');
assert(isequal(cvOld.n_valid_dates, cvNew.n_valid_dates), ...
    'Valid-date counts differ.');
assert(cvOld.lambda == cvNew.lambda, ...
    'The two exact CV implementations selected different lambdas.');

fprintf('test_cross_validate_lambda_prewhiten_equivalence passed; max loss difference = %.3e\n', err);
end
