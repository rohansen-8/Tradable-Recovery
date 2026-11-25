function ell = median_pairwise_lengthscale(x)
%% Return the median of all pairwise distances for one scalar coordinate.
%
% This is the deterministic MATLAB analogue of scipy.spatial.distance.pdist
% followed by median, which is the empirical lengthscale convention. Zero
% pairwise distances are retained, exactly as in pdist.

x = x(:);
assert(numel(x) >= 2, ...
    'median_pairwise_lengthscale requires at least two observations.');
assert(all(isfinite(x)), ...
    'Lengthscale observations must be finite.');

[support,~,group] = unique(x);
counts = accumarray(group,1);
ell = median_pairwise_lengthscale_weighted(support,counts);

assert(isfinite(ell) && ell > 0, ...
    'Median pairwise distance must be finite and strictly positive.');

end
