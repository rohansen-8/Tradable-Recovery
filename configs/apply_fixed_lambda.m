function cfg = apply_fixed_lambda(cfg, lambda)
%% Bind one supplied lambda to a fixed_lambda simulation configuration.
% No-op for 'cv' and 'eff_dim'. For 'fixed_lambda' the lambda must be one
% positive finite scalar; it is stored as cfg.training.lambda (so it enters the
% design fingerprint) and the results path gains the deterministic subfolder
% lambda_<tag> (fixed_lambda_tag). No selection rule is involved.

if ~strcmp(char(string(cfg.training.lambda_method)), 'fixed_lambda'), return; end
assert(isnumeric(lambda) && isscalar(lambda) && isfinite(lambda) && lambda > 0, ...
    'apply_fixed_lambda:BadLambda', 'fixed_lambda requires one positive finite lambda.');
cfg.training.lambda = double(lambda);
cfg.training.use_cv = false;
cfg.results.path = fullfile(cfg.results.path, fixed_lambda_tag(lambda));
end
