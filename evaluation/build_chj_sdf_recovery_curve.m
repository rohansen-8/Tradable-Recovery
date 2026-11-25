function rows = build_chj_sdf_recovery_curve(cfg, model_state, path, Tval, rep, seed, pre)
%% Build one CHJ SDF-recovery curve from an already fitted dataset.
%
% No refit or CV is performed. On every date t the estimated SDF is the
% HJ-normalized SDF of the manuscript (eq. hj_normalized_sdf),
%     M_t(m) = 1 - Rscr_t(h)(m) = 1 - alpha_t(m)'h,
% the estimator's own zero-cost payoff SDF evaluated at gross return R = m:
% alpha_t(m) = kron(kV_t, zeta_t(m)) uses the same option grid, strike
% weights, date-t forward prices (conditional variance h_t) and VIX kernel
% section kV_t as the estimation panel (build_chj_alpha_numeric). It does not
% depend on the level of h, which the payoffs do not identify, so it stays
% well defined when lambda is small. The theoretical SDF is the analytic CHJ
% SDF on the same date and state h_t. Each
% date-specific estimated and theoretical curve is normalized to mean one on
% the production shape-test grid, the curves are averaged across dates, and
% the pointwise estimated-minus-true difference is formed.
%
% pre: build_chj_alpha_precompute output used for the fit (required; only its
% moneyness, option-grid and quadrature objects are used).

assert(nargin >= 7 && isstruct(pre) && isfield(pre, 'option_m_grid'), ...
    'build_chj_sdf_recovery_curve:NoPrecompute', ...
    'The HJ-normalized recovery needs the CHJ alpha precompute used in the fit.');

m = build_global_shape_grid(cfg);
m = m(:);
H = model_state.H_star;
exoGrid = model_state.exo_grid(:);
assert(isequal(size(H), [numel(pre.mon_grid) + 1, numel(exoGrid)]), ...
    'build_chj_sdf_recovery_curve:BadDimensions', ...
    'H_star and the alpha precompute have inconsistent dimensions.');

% path.h has T+1 entries; h(t), t = 1..T, is the conditioning variance of
% date t (VIX(t) = sqrt(252 h(t))) used to price the date-t options.
vix = path.VIX(:);
nDates = numel(vix);
assert(numel(path.h) >= nDates, 'build_chj_sdf_recovery_curve:BadPath', ...
    'path.h is shorter than path.VIX.');
hState = path.h(1:nDates);
hState = hState(:);

% kV_t exactly as in the estimation panel (observed VIX, fitted lengthscale).
KV = laplace_k(exoGrid, vix.', model_state.ls_exo);        % p-by-T
HKV = H * KV;                                               % (n+1)-by-T

estRaw = zeros(numel(m), nDates);
for t = 1:nDates
    zeta = build_chj_zeta_from_returns(m, hState(t), pre);  % (n+1)-by-K
    estRaw(:, t) = 1 - zeta.' * HKV(:, t);
end
hjMeans = mean(estRaw, 1);
assert(all(isfinite(hjMeans)) && all(hjMeans > 0), ...
    'build_chj_sdf_recovery_curve:NonPositiveMean', ...
    'HJ-normalized SDF has a non-positive grid mean on %d of %d dates (min %.6g).', ...
    sum(~(hjMeans > 0)), nDates, min(hjMeans));
est = mean(estRaw ./ hjMeans, 2);

params = calculate_chj_parameters(cfg.chj.mu, cfg.chj.xi, cfg.chj.omega, ...
    cfg.chj.beta, cfg.chj.alpha, cfg.chj.gamma, cfg.chj.r);
trueRaw = local_true_sdf_dates(m, hState, cfg.chj, params);
trueSdf = mean(trueRaw ./ mean(trueRaw, 1), 2);
difference = est - trueSdf;

n = numel(m);
fracOutside = mean(vix < min(exoGrid) | vix > max(exoGrid));
rows = table(repmat(Tval,n,1), repmat(rep,n,1), repmat(seed,n,1), ...
    repmat(model_state.lambda,n,1), m, est, trueSdf, difference, ...
    repmat(fracOutside,n,1), repmat(min(hjMeans),n,1), ...
    repmat("hj_normalized",n,1), ...
    'VariableNames', {'Tval','rep','seed','lambda','m', ...
    'estimated_sdf','true_sdf','difference','exo_frac_outside_grid', ...
    'hj_min_date_grid_mean','recovery_measure'});

assert(abs(mean(est) - 1) < 5e-11 && abs(mean(trueSdf) - 1) < 5e-11, ...
    'CHJ recovery curves are not mean-one normalized.');
assert(abs(mean(difference)) < 1e-10, ...
    'CHJ recovery difference must have zero grid mean after normalization.');

end

function trueSdf = local_true_sdf_dates(mGrid, hEval, chj, params)
%% Evaluate the analytical CHJ SDF for all dates on the common moneyness grid.

m = mGrid(:);
h = hEval(:).';
assert(all(m > 0) && all(h > 0), ...
    'CHJ recovery evaluation requires positive moneyness and variance.');

sqrtH = sqrt(h);
z = (log(m) - chj.r - (chj.mu - 0.5) .* h) ./ sqrtH;
varianceChange = chj.omega + (chj.beta - 1) .* h ...
    + chj.alpha .* (z - chj.gamma .* sqrtH) .^ 2;
trueSdf = (m .^ params.phi) ...
    .* exp(params.delta + params.eta .* h + chj.xi .* varianceChange);

assert(all(isfinite(trueSdf(:))) && all(trueSdf(:) > 0), ...
    'Analytical CHJ SDF contains invalid values.');

end
