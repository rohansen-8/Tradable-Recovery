function test_median_pairwise_lengthscale_scalable()
%% Verify scalable median-pairwise selection against direct pair distances.

rng(719, 'twister');
x = randn(2001, 1);
x(1:5) = x(6); % retain duplicate/zero pair distances in the reference object

D = abs(x - x.');
d = D(triu(true(numel(x)), 1));
expected = median(d);
actual = median_pairwise_lengthscale(x);

assert(abs(actual - expected) <= 1e-10 * max(1, abs(expected)), ...
    'Scalable median-pairwise result does not match the direct definition.');

fprintf('test_median_pairwise_lengthscale_scalable passed\n');

end
