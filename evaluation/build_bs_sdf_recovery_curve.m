function rows = build_bs_sdf_recovery_curve(cfg, model_state, Tval, rep, seed, pre)
%% Build one BS SDF-recovery curve from an already fitted dataset.
%
% No refit or CV is performed. The estimated SDF is the HJ-normalized SDF of
% the manuscript (eq. hj_normalized_sdf),
%     M(m) = 1 - Rscr(h)(m) = 1 - h(m) + E^Q[h],
% where Rscr(h) is the zero-cost traded payoff of the fitted rule. In the BS
% design that payoff is exactly alpha(R)'h with alpha(R) built from the same
% option grid, prices and strike weights as the estimation panel
% (build_bs_alpha_from_returns), so M(m) = 1 - alpha(m)'h is the estimator's
% own payoff SDF evaluated at gross return R = m. It does not depend on the
% level of h, which the option payoffs do not identify (only its derivatives
% enter), so it stays well defined when lambda is small. The estimated and theoretical curves are then
% normalized to mean one on the production shape-test grid before the
% pointwise estimated-minus-true difference is formed.
%
% pre: build_bs_alpha_precompute output used for the fit (required).

assert(nargin >= 6 && isstruct(pre) && isfield(pre, 'mGrid'), ...
    'build_bs_sdf_recovery_curve:NoPrecompute', ...
    'The HJ-normalized recovery needs the BS alpha precompute used in the fit.');
h = model_state.h_star(:);
assert(numel(h) == pre.n + 1, 'build_bs_sdf_recovery_curve:BadDimensions', ...
    'h_star and the alpha precompute have inconsistent dimensions.');

m = build_global_shape_grid(cfg);
m = m(:);
alphaGrid = build_bs_alpha_from_returns(m, pre);   % (n+1)-by-K
hjRaw = 1 - alphaGrid.' * h;
hjMean = mean(hjRaw);
assert(isfinite(hjMean) && hjMean > 0, 'build_bs_sdf_recovery_curve:NonPositiveMean', ...
    'HJ-normalized SDF has a non-positive grid mean (%.6g).', hjMean);
est = hjRaw / hjMean;

trueSdf = bs_true_sdf(m, cfg.bs.mu, cfg.bs.sigma, cfg.bs.dt, cfg.bs.r);
trueSdf = trueSdf(:) / mean(trueSdf(:));
difference = est - trueSdf;

n = numel(m);
rows = table(repmat(Tval,n,1), repmat(rep,n,1), repmat(seed,n,1), ...
    repmat(model_state.lambda,n,1), m, est, trueSdf, difference, ...
    repmat(hjMean,n,1), repmat("hj_normalized",n,1), ...
    'VariableNames', {'Tval','rep','seed','lambda','m', ...
    'estimated_sdf','true_sdf','difference','hj_grid_mean','recovery_measure'});

assert(abs(mean(est) - 1) < 5e-11 && abs(mean(trueSdf) - 1) < 5e-11, ...
    'BS recovery curves are not mean-one normalized.');
assert(abs(mean(difference)) < 1e-10, ...
    'BS recovery difference must have zero grid mean after normalization.');

end
