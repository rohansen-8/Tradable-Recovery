function test_median_pairwise_lengthscale_weighted()
%% Verify weighted pairwise medians against explicit expanded training samples.

x = [0.95; 1.00; 1.05];
counts = [4; 3; 5];
expanded = repelem(x, counts);
expected = median_pairwise_lengthscale(expanded);
actual = median_pairwise_lengthscale_weighted(x, counts);
assert(abs(actual - expected) <= 1e-12);

% Option-row repetition of the kind the CHJ panel produces: each date-level
% state appears J times. The multiplicity below is arbitrary, not the
% production strike count; the identity must hold for any J.
rng(817, 'twister');
v = randn(25, 1);
J = 31;
expandedV = repelem(v, J);
expectedV = median_pairwise_lengthscale(expandedV);
actualV = median_pairwise_lengthscale_weighted(v, repmat(J, numel(v), 1));
assert(abs(actualV - expectedV) <= 1e-12 * max(1, abs(expectedV)));

% Exercise the scalable weighted branch without materializing pair distances.
rng(818, 'twister');
vLarge = randn(1001, 1);
expandedLarge = repelem(vLarge, 2);
expectedLarge = median_pairwise_lengthscale(expandedLarge);
actualLarge = median_pairwise_lengthscale_weighted( ...
    vLarge, repmat(2, numel(vLarge), 1));
assert(abs(actualLarge - expectedLarge) <= 1e-10 * max(1, abs(expectedLarge)));

fprintf('test_median_pairwise_lengthscale_weighted passed\n');

end
