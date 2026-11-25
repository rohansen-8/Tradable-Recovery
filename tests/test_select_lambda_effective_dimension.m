function test_select_lambda_effective_dimension()
%% Synthetic checks of the effective-dimension lambda selector.
%
% 1. Reduced Nh-by-Nh eigenvalues equal the nonzero eigenvalues of the direct
%    T-by-T Gram matrix G_T / T, for T > Nh and T < Nh.
% 2. The root satisfies tr[G_T (G_T + lambda T I)^{-1}] = T^(1/3), evaluated
%    on the direct T-by-T matrix.
% 3. Scaling: G_T -> s*G_T (via alpha -> sqrt(s)*alpha or K_h -> K_h/s) gives
%    lambda -> s*lambda.
% 4. No grid and no floor: a small-scale problem returns lambda < 1e-5.
% 5. Unattainable targets and invalid inputs raise explicit errors.
% 6. Ill-conditioned kernel (the fixed Matern 7/2 x Laplace basis with
%    jitter 1e-10): reduced and direct formulations agree.
% 7. Exactly rank-deficient payoff panel with the target just below its rank
%    (the BS empirical-T daily situation): the numerical rank is the exact
%    rank, not inflated by a Gram-formation rounding floor.
% The selector takes the payoff factor build_payoff_factor(alpha).
% Synthetic matrices only; no data, no fitting, no results written.

rng(20261120, 'twister');

%% 1-2. Well-conditioned K, T > Nh and T < Nh
Nh = 40;
B = randn(Nh);
K = B * B.' + Nh * eye(Nh);
for T = [60 25]
    alpha = randn(Nh, T) .* linspace(3, 0.1, Nh).';
    sel = select_lambda_effective_dimension(build_payoff_factor(alpha), K, T, 1);

    G = alpha.' * (K \ alpha);  G = 0.5 * (G + G.');
    muD = sort(max(eig(G / T), 0), 'descend');
    r = min(Nh, T);
    eigErr = max(abs(sel.eigenvalues(1:r) - muD(1:r))) / muD(1);
    assert(eigErr <= 1e-11, 'Reduced vs direct eigenvalues differ (%.3g), T = %d.', eigErr, T);
    assert(all(muD(r+1:end) <= 1e-12 * muD(1)), 'Direct Gram has extra nonzero eigenvalues.');
    assert(sel.numerical_rank == r, 'Numerical rank %d, expected %d.', sel.numerical_rank, r);

    Ntrace = trace(G / (G + sel.lambda * T * eye(T)));
    assert(abs(Ntrace - T^(1/3)) <= 1e-9 * T^(1/3), ...
        'Direct trace formula gives %.15g, target %.15g.', Ntrace, T^(1/3));
    assert(abs(sel.residual) <= 1e-10 * max(1, sel.target));
    fprintf('  T = %3d: lambda = %.6e, eigErr = %.2e, trace residual = %.2e, rank = %d\n', ...
        T, sel.lambda, eigErr, Ntrace - T^(1/3), sel.numerical_rank);
end

%% 3. Scaling
T = 60;
alpha = randn(Nh, T);
P = build_payoff_factor(alpha);
base = select_lambda_effective_dimension(P, K, T, 1);
for s = [1e-3 7.3 1e4]
    a = select_lambda_effective_dimension(build_payoff_factor(sqrt(s) * alpha), K, T, 1);
    b = select_lambda_effective_dimension(P, K / s, T, 1);
    assert(abs(a.lambda / (s * base.lambda) - 1) <= 1e-10, 'Q-scaling failed for s = %g.', s);
    assert(abs(b.lambda / (s * base.lambda) - 1) <= 1e-10, 'K-scaling failed for s = %g.', s);
end
fprintf('  scaling: lambda(s*G) = s*lambda(G) to 1e-10 for s in {1e-3, 7.3, 1e4}\n');

%% 4. No grid, no floor
tiny = select_lambda_effective_dimension(build_payoff_factor(sqrt(1e-9) * alpha), K, T, 1);
assert(tiny.lambda < 1e-5 && abs(tiny.residual) <= 1e-10 * tiny.target, ...
    'Selector must solve continuously below the old 1e-5 floor.');
fprintf('  no floor: lambda = %.3e (< 1e-5) solves the rule\n', tiny.lambda);

%% 5. Failures
T = 8;                                   % rank(G_T) <= 8
alpha = randn(Nh, T);
P = build_payoff_factor(alpha);
expect_error(@() select_lambda_effective_dimension(P, K, T, 4), ...
    'select_lambda_effective_dimension:Unattainable');      % target = 8 = rank
expect_error(@() select_lambda_effective_dimension(P, K, T, 10), ...
    'select_lambda_effective_dimension:Unattainable');      % target > rank
expect_error(@() select_lambda_effective_dimension(P, -K, T, 1), ...
    'select_lambda_effective_dimension:KhNotPD');
expect_error(@() select_lambda_effective_dimension(build_payoff_factor(zeros(Nh, T)), K, T, 1), ...
    'select_lambda_effective_dimension:ZeroGram');
expect_error(@() select_lambda_effective_dimension(P, K, T, 0), ...
    'select_lambda_effective_dimension:BadC');
Pn = P; Pn.F(1) = NaN;
expect_error(@() select_lambda_effective_dimension(Pn, K, T, 1), ...
    'select_lambda_effective_dimension:NonFinite');
expect_error(@() select_lambda_effective_dimension(alpha * alpha.' / T, K, T, 1), ...
    'select_lambda_effective_dimension:BadPayoff');          % Q_alpha is not accepted
expect_error(@() select_lambda_effective_dimension(P, K, T + 1, 1), ...
    'select_lambda_effective_dimension:BadT');
fprintf('  failures: unattainable target, non-PD K_h, zero Gram, bad c, NaN, Q_alpha input, T mismatch raise errors\n');

%% 6. Fixed basis kernel (930 x 930, jitter 1e-10)
cfg = struct('reference_point', 1, 'has_exo', true);
cfg.kernel.endo = struct('type', 'matern72', 'lengthscale', 0.08);
cfg.kernel.exo = struct('type', 'laplace', 'lengthscale', 9);
cfg.kernel.jitter = 1e-10;
Kh = build_kernel_norm_matrix(cfg, linspace(0.75, 1.09, 30).', linspace(5, 65, 30).');
T = 300;
Z = randn(size(Kh, 1), T);
alpha = Kh * Z;                          % representers in the span of K_h
sel = select_lambda_effective_dimension(build_payoff_factor(alpha), Kh, T, 1);
G = Z.' * Kh * Z;  G = 0.5 * (G + G.');  % = alpha' K_h^{-1} alpha, no inverse
muD = sort(max(eig(G / T), 0), 'descend');
k = 25;                                  % eigenvalues that drive N near the root
relTop = max(abs(sel.eigenvalues(1:k) - muD(1:k)) ./ muD(1:k));
Ntrace = trace(G / (G + sel.lambda * T * eye(T)));
assert(relTop <= 1e-6, 'Top eigenvalues differ under ill-conditioning (%.3g).', relTop);
assert(abs(Ntrace - T^(1/3)) <= 1e-6 * T^(1/3), ...
    'Direct N_T at the selected lambda is %.10g, target %.10g.', Ntrace, T^(1/3));
fprintf(['  frozen basis: rcond(K_h) = %.2e, ' ...
    'top-%d eig rel err = %.2e, direct N - target = %.2e\n'], sel.rcond_K_h, ...
    k, relTop, Ntrace - T^(1/3));

%% 7. Exact rank r with target just below r (fixed basis, T^(1/3) = 12.6 < 13)
T = 2000; r = 13;
Zr = randn(size(Kh, 1), r) * diag(logspace(0, -5.5, r));   % weakest direction ~1e-11 in mu
alpha = Kh * (Zr * randn(r, T));                            % rank(alpha) = r exactly
sel = select_lambda_effective_dimension(build_payoff_factor(alpha), Kh, T, 1);
assert(sel.numerical_rank == r, 'Rank-%d panel reported numerical rank %d.', r, sel.numerical_rank);
assert(abs(sel.residual) <= 1e-10 * sel.target);
assert(sel.eigenvalues(r + 1) <= 1e-20 * sel.eigenvalues(1), ...
    'Eigenvalue %d is %.3g relative, expected rounding of a zero.', r + 1, ...
    sel.eigenvalues(r + 1) / sel.eigenvalues(1));
fprintf('  exact rank %d, target %.3f: numerical rank %d, mu_{r+1}/mu_1 = %.1e, lambda = %.3e\n', ...
    r, sel.target, sel.numerical_rank, sel.eigenvalues(r + 1) / sel.eigenvalues(1), sel.lambda);

disp('test_select_lambda_effective_dimension passed.');
end

function expect_error(fn, id)
try
    fn();
catch err
    assert(strcmp(err.identifier, id), 'Expected %s, got %s: %s', id, err.identifier, err.message);
    return
end
error('Expected error %s was not raised.', id);
end
