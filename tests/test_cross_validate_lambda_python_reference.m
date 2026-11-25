function test_cross_validate_lambda_python_reference()
%% Verify MATLAB one-step lambda CV against the Python reference fixture.

    setpath();

    fixtureDir = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
    alpha = readmatrix(fullfile(fixtureDir, 'python_cv_alpha.csv'));
    K_h = readmatrix(fullfile(fixtureDir, 'python_cv_K_h.csv'));
    ref = readtable(fullfile(fixtureDir, 'python_cv_results.csv'));

    minTrain = 8;

    cfg = struct();
    cfg.training.cv.lambda_grid = ref.lambda;
    cfg.training.cv.min_train_dates = minTrain;
    cfg.training.cv.parallel = false;
    cfg.parallel.use = false;

    alphaPanel = struct();
    alphaPanel.alpha = alpha;

    cvDirect = cross_validate_lambda(cfg, alphaPanel, K_h);
    cfg.training.cv.solver = 'fold_eig';
    cvFoldEig = cross_validate_lambda(cfg, alphaPanel, K_h);

    tol = 1e-10;
    assert_close(cvDirect.mean_loss, ref.mean_loss, tol, ...
        'Direct CV losses differ from the Python reference.');
    assert_close(cvFoldEig.mean_loss, ref.mean_loss, tol, ...
        'Fold-eig CV losses differ from the Python reference.');
    assert_close(cvFoldEig.mean_loss, cvDirect.mean_loss, tol, ...
        'Fold-eig and direct CV losses differ.');

    [~, refIdx] = min(ref.mean_loss);
    refLambda = ref.lambda(refIdx);
    assert(abs(cvDirect.lambda - refLambda) <= tol, ...
        'Direct selected lambda differs from the Python reference grid search.');
    assert(abs(cvFoldEig.lambda - refLambda) <= tol, ...
        'Fold-eig selected lambda differs from the Python reference grid search.');

    expectedValIdx = ((minTrain + 1):size(alpha, 2)).';
    assert(isequal(cvFoldEig.valIdx, expectedValIdx), ...
        'Validation indices do not match one-step-ahead expanding windows.');
    assert(strcmp(cvFoldEig.method, '1_step_ahead'));
    assert(strcmp(cvFoldEig.objective, 'projection_loss'));
    assert(strcmp(cvFoldEig.selection_rule, 'mean_loss'));
    assert(strcmp(cvFoldEig.solver, 'fold_eig'));

    fprintf('test_cross_validate_lambda_python_reference passed.\n');
end

function assert_close(A, B, tol, msg)
%% Compare arrays with a uniform absolute tolerance.

    err = max(abs(A(:) - B(:)));
    if ~(isfinite(err) && err <= tol)
        error('test_cross_validate_lambda_python_reference:AssertClose', ...
            '%s max error = %.3e.', msg, err);
    end
end
