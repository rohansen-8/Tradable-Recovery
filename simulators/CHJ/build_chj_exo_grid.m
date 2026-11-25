function exo_grid = build_chj_exo_grid(cfg, VIX)
%% Build 30 equally spaced CHJ state points on the replication q05-q95 support.
%
% The support endpoints are the 5th and 95th percentiles of the simulated
% VIX path. Given those endpoints, the coefficient grid follows the empirical
% estimator: 30 equally spaced state points.

VIX = VIX(:);
nGrid = cfg.training.n_exo_grid;
q = cfg.training.exo_support_quantiles(:).';

assert(nGrid == 30, ...
    'CHJ revision requires 30 exogenous coefficient points.');
assert(numel(VIX) >= 2 && all(isfinite(VIX)), ...
    'VIX must contain at least two finite simulated state observations.');
assert(numel(q) == 2 && q(1) >= 0 && q(2) <= 1 && q(1) < q(2), ...
    'CHJ exogenous-support quantiles must be valid increasing probabilities.');

bounds = quantile(VIX, q);
assert(all(isfinite(bounds)) && bounds(1) < bounds(2), ...
    'CHJ q05-q95 state support must be finite and nondegenerate.');

exo_grid = linspace(bounds(1), bounds(2), nGrid).';

assert(numel(exo_grid) == 30 && all(diff(exo_grid) > 0), ...
    'CHJ exogenous grid must contain 30 strictly increasing points.');

end
