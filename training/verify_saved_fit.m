function verify_saved_fit(model_state, cfg, label)
%% Independently re-verify a saved empirical fitted state before inference.
%
% Shared checks: required fields, basis grids, lengthscales, and reproduction
% of h_star from its saved linear system (Q_alpha + lambda K_h) h = alpha_bar.
% Method-specific checks on how lambda was selected:
%   'cv'      the saved lambda lies on the fixed CV grid;
%   'eff_dim' the effective-dimension rule is re-solved from the saved
%             payoff panel alpha, K_h, T and c (build_payoff_factor and one
%             singular-value decomposition) and must return the saved lambda;
%             h_star is reproduced by the same QR least-squares solve used in
%             training (solve_projection_qr).

if nargin < 3 || isempty(label), label = 'fit'; end
label = char(string(label));

lambdaMethod = 'cv';
if isfield(cfg, 'lambda_method') && ~isempty(cfg.lambda_method)
    lambdaMethod = char(string(cfg.lambda_method));
end
isEffDim = strcmp(lambdaMethod, 'eff_dim');
savedIsEffDim = isfield(model_state, 'lambda_method') ...
    && strcmp(model_state.lambda_method, 'eff_dim');
assert(isEffDim == savedIsEffDim, ...
    '%s saved state was not selected by lambda_method = %s.', label, lambdaMethod);

required = {'h_star','lambda','mon_grid','exo_grid','ls_endo','ls_exo', ...
    'K_h','Q_alpha','alpha_bar','alpha','T'};
if isEffDim
    required{end+1} = 'effdim';
else
    required{end+1} = 'cv';
end
for j = 1:numel(required)
    assert(isfield(model_state, required{j}), ...
        '%s saved state missing %s.', label, required{j});
end

if isEffDim
    payoff = build_payoff_factor(model_state.alpha);
    sel = select_lambda_effective_dimension(payoff, ...
        model_state.K_h, model_state.T, model_state.effdim.c);
    assert(abs(sel.target - model_state.effdim.target) <= 1e-12 * sel.target, ...
        '%s saved effective-dimension target changed.', label);
    assert(abs(sel.lambda - model_state.lambda) <= ...
        1e-8 * model_state.lambda, ...
        '%s saved lambda does not re-solve the effective-dimension rule (%.6g vs %.6g).', ...
        label, model_state.lambda, sel.lambda);
else
    assert(any(abs(cfg.training.cv.lambda_grid - model_state.lambda) <= ...
        1e-13 * max(1, abs(model_state.lambda))), ...
        '%s fitted lambda is outside the frozen grid.', label);
end

assert(max(abs(model_state.mon_grid(:) - cfg.training.mon_grid(:))) <= 1e-12, ...
    '%s saved estimator basis grid changed.', label);
assert(max(abs(model_state.exo_grid(:) - cfg.training.exo_grid(:))) <= 1e-12, ...
    '%s saved exogenous basis grid changed.', label);
assert(isfinite(model_state.ls_endo) && all(isfinite(model_state.ls_exo)), ...
    '%s saved kernel lengthscales are invalid.', label);

if isEffDim
    h = solve_projection_qr(payoff, model_state.K_h, model_state.lambda);
else
    A = model_state.Q_alpha + model_state.lambda * model_state.K_h;
    A = 0.5 * (A + A.');
    [L, p] = chol(A, 'lower');
    if p == 0
        h = L' \ (L \ model_state.alpha_bar(:));
    else
        h = A \ model_state.alpha_bar(:);
    end
end
relErr = norm(h(:) - model_state.h_star(:)) / max(1, norm(model_state.h_star(:)));
assert(relErr <= 1e-10, ...
    '%s saved fitted curve cannot be reproduced (rel err %.3g).', label, relErr);
end
