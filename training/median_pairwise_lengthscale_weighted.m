function ell = median_pairwise_lengthscale_weighted(x, counts)
%% Return the median pairwise distance for an expanded weighted scalar sample.
%
% x lists scalar support values and counts gives the number of times each
% value appears in the conceptual training sample. The calculation includes
% zero distances between repeated observations and exactly matches the median
% pairwise-distance rule without materializing the expanded sample.

x = x(:);
counts = counts(:);

assert(numel(x) == numel(counts) && numel(x) >= 1, ...
    'x and counts must have the same nonzero length.');
assert(all(isfinite(x)), 'Lengthscale support values must be finite.');
assert(all(isfinite(counts) & counts >= 0 & counts == floor(counts)), ...
    'counts must contain finite nonnegative integers.');

keep = counts > 0;
x = x(keep);
counts = counts(keep);
assert(sum(counts) >= 2, ...
    'The expanded sample must contain at least two observations.');

% Merge repeated support values so within-value zero pairs are handled once.
[xSorted, order] = sort(x);
counts = counts(order);
[xUnique, ~, group] = unique(xSorted);
weights = accumarray(group, counts, [], @sum);

N = sum(weights);
nPairs = N * (N - 1) / 2;

if mod(nPairs, 2) == 1
    ell = kth_weighted_pair_distance(xUnique, weights, (nPairs + 1) / 2);
else
    d1 = kth_weighted_pair_distance(xUnique, weights, nPairs / 2);
    d2 = kth_weighted_pair_distance(xUnique, weights, nPairs / 2 + 1);
    ell = 0.5 * (d1 + d2);
end

assert(isfinite(ell) && ell > 0, ...
    'Median pairwise distance must be finite and strictly positive.');

end

function d = kth_weighted_pair_distance(x, weights, k)
%% Select one order statistic from the weighted pair-distance multiset.

zeroPairs = sum(weights .* (weights - 1) / 2);
if zeroPairs >= k
    d = 0;
    return
end

% For modest support, enumerate distinct between-support distances and their
% multiplicities. This is exact and cheap for the CHJ option moneyness grid.
n = numel(x);
if n <= 1000
    [ii, jj] = find(triu(true(n), 1));
    distances = x(jj) - x(ii);
    multiplicities = weights(ii) .* weights(jj);

    values = [0; distances(:)];
    mult = [zeroPairs; multiplicities(:)];
    [values, order] = sort(values);
    mult = mult(order);
    cumulative = cumsum(mult);
    d = values(find(cumulative >= k, 1, 'first'));
    return
end

% Large one-dimensional supports use a monotone pair-count search. This avoids
% O(n^2) storage for T=5000/10000 replication-specific VIX paths.
lo = 0;
hi = x(end) - x(1);

for iter = 1:80
    mid = lo + 0.5 * (hi - lo);
    if count_weighted_pairs_leq(x, weights, mid) >= k
        hi = mid;
    else
        lo = mid;
    end

    if hi - lo <= 8 * eps(max(1, abs(hi)))
        break
    end
end

d = hi;

end

function count = count_weighted_pairs_leq(x, weights, d)
%% Count expanded-sample unordered pairs with distance at most d.

n = numel(x);
prefix = [0; cumsum(weights)];
count = sum(weights .* (weights - 1) / 2);
left = 1;

for right = 1:n
    while x(right) - x(left) > d
        left = left + 1;
    end

    if right > left
        weightLeft = prefix(right) - prefix(left);
        count = count + weights(right) * weightLeft;
    end
end

end
