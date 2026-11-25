function test_cross_validate_lambda_scale_invariance()
%% Verify normalized one-step CV is scale invariant and solver consistent.

setpath();

rng(81626, 'twister');
Nh = 14;
T = 40;
R = randn(Nh);
K_h = R*R.' + eye(Nh);
alpha = randn(Nh,T);
alphaPanel = struct('alpha', alpha);

cfg = struct();
cfg.training.cv.lambda_grid = logspace(-5,-2,31).';
cfg.training.cv.lambda_scale_mode = 'mean_generalized_eigenvalue';
cfg.training.cv.min_train_dates = 12;
cfg.training.cv.parallel = false;
cfg.training.cv.solver = 'fold_eig';
cfg.parallel.use = false;

cv0 = cross_validate_lambda(cfg, alphaPanel, K_h);

a = 7.5;
cvAlpha = cross_validate_lambda(cfg, struct('alpha', a*alpha), K_h);

b = 4.0;
cvPenalty = cross_validate_lambda(cfg, alphaPanel, b*K_h);

tol = 2e-10;
assert(cv0.lambda_multiplier == cvAlpha.lambda_multiplier);
assert(cv0.lambda_multiplier == cvPenalty.lambda_multiplier);
assert(max(abs(cv0.mean_loss - cvAlpha.mean_loss)) <= tol);
assert(max(abs(cv0.mean_loss - cvPenalty.mean_loss)) <= tol);
assert(abs(cvAlpha.lambda/(a^2*cv0.lambda) - 1) <= tol);
assert(abs(cvPenalty.lambda/(cv0.lambda/b) - 1) <= tol);

cfg.training.cv.solver = 'fold_eig_prewhiten';
cvPre = cross_validate_lambda(cfg, alphaPanel, K_h);
assert(max(abs(cv0.mean_loss - cvPre.mean_loss)) <= tol);
assert(cv0.lambda_multiplier == cvPre.lambda_multiplier);

cfg.training.cv.solver = 'fold_eig_woodbury';
cfg.training.cv.woodbury_refresh = 5;
cvWB = cross_validate_lambda(cfg, alphaPanel, K_h);
assert(max(abs(cv0.mean_loss - cvWB.mean_loss)) <= 5e-10);
assert(cv0.lambda_multiplier == cvWB.lambda_multiplier);

cfg.training.cv.solver = 'direct';
cvDirect = cross_validate_lambda(cfg, alphaPanel, K_h);
assert(max(abs(cv0.mean_loss - cvDirect.mean_loss)) <= tol);
assert(cv0.lambda_multiplier == cvDirect.lambda_multiplier);

fprintf('test_cross_validate_lambda_scale_invariance passed.\n');
end
