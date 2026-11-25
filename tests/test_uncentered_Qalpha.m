function test_uncentered_Qalpha()
%% Verify Q_alpha is the uncentered second moment alpha*alpha'/T.

rng(1);
alpha = randn(5, 11);

moments = build_return_second_moment(alpha);

alpha_bar = moments.alpha_bar;
Q_alpha = moments.Q_alpha;

T = size(alpha, 2);
Q_expected = (alpha * alpha.') / T;
Q_expected = 0.5 * (Q_expected + Q_expected.');
alpha_expected = mean(alpha, 2);

tol = 1e-12;

assert(norm(alpha_bar - alpha_expected, 'fro') < tol, ...
    'alpha_bar is not mean(alpha,2).');

assert(norm(Q_alpha - Q_expected, 'fro') < tol, ...
    'Q_alpha must be the uncentered second moment alpha*alpha''/T.');

alpha_centered = alpha - mean(alpha, 2);
Q_centered = (alpha_centered * alpha_centered.') / T;
Q_centered = 0.5 * (Q_centered + Q_centered.');

assert(norm(Q_alpha - Q_centered, 'fro') > 1e-8, ...
    'Q_alpha appears to be centered; it must be uncentered.');

fprintf('test_uncentered_Qalpha passed.\n');
end
