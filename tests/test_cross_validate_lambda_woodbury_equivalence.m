function test_cross_validate_lambda_woodbury_equivalence()
%% Verify Woodbury-refresh CV against the exact prewhitened fold-EVD solver.

rng(81226, 'twister');
Nh = 36;
T = 52;

R = randn(Nh);
K_h = R * R.' + 0.5 * eye(Nh);
alpha = randn(Nh, T);
alphaPanel = struct('alpha', alpha);

cfg = config_chj_simulation_design();
cfg.training.cv.min_train_dates = 10;
cfg.training.cv.lambda_grid = logspace(-5, -2, 31).';

cfgRef = cfg;
cfgRef.training.cv.solver = 'fold_eig_prewhiten';
cvRef = cross_validate_lambda(cfgRef, alphaPanel, K_h);

for refreshEvery = [1 5 10 20]
    cfgWB = cfg;
    cfgWB.training.cv.solver = 'fold_eig_woodbury';
    cfgWB.training.cv.woodbury_refresh = refreshEvery;

    cvWB = cross_validate_lambda(cfgWB, alphaPanel, K_h);

    lossScale = max(1, max(abs(cvRef.mean_loss)));
    err = max(abs(cvRef.mean_loss - cvWB.mean_loss));

    assert(err <= 5e-10 * lossScale, ...
        'Woodbury and prewhitened CV losses differ materially.');
    assert(isequal(cvRef.n_valid_dates, cvWB.n_valid_dates), ...
        'Valid-date counts differ.');
    assert(cvRef.lambda == cvWB.lambda, ...
        'Woodbury and prewhitened CV selected different lambdas.');

    fprintf('refresh=%d: max loss difference = %.3e; lambda = %.12g\n', ...
        refreshEvery, err, cvWB.lambda);
end

fprintf('test_cross_validate_lambda_woodbury_equivalence passed.\n');
end
