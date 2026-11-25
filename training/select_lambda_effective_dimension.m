function sel = select_lambda_effective_dimension(payoff, K_h, T, c)
%% Select lambda by the empirical effective-dimension rule of the paper.
%
% Solves (manuscript eq. effective_dimension_rule)
%
%     N_T(lambda) = tr[G_T (G_T + lambda T I_T)^{-1}]
%                 = sum_j mu_j / (mu_j + lambda)  =  c * T^(1/3),
%
% where mu_j are the eigenvalues of G_T / T and G_T = alpha' K_h^{-1} alpha is
% the T-by-T Gram matrix of the payoff representers in the restricted basis.
% c = 1 is the paper's rule.
%
% Reduced representation. With K_h = R'R (Cholesky) and the triangular payoff
% factor F of build_payoff_factor (F'F = Q_alpha = alpha*alpha'/T),
%     G_T / T = (R^{-T} alpha)'(R^{-T} alpha) / T,
% whose nonzero eigenvalues equal those of W = R^{-T} Q_alpha R^{-1}
% = (F R^{-1})'(F R^{-1}), i.e. the generalized eigenvalues of (Q_alpha, K_h).
% They are computed as the squared singular values of F R^{-1}; neither W,
% Q_alpha nor the T-by-T Gram matrix is formed, so eigenvalues far below the
% largest are not lost to the rounding floor (about eps*cond(K_h) relative)
% that forming and diagonalizing W imposes.
%
% F R^{-1} has min(T, Nh) singular values, so at most min(Nh, T) eigenvalues
% are nonzero, as in the T-by-T formulation; the remaining entries of the
% saved Nh-vector are zero. The other T - rank eigenvalues of G_T / T are
% zero and contribute nothing to N_T.
%
% N_T is continuous and strictly decreasing in lambda, from the numerical rank
% r of G_T (lambda -> 0) to 0 (lambda -> inf), so a finite positive solution
% exists and is unique iff 0 < c*T^(1/3) < r. Otherwise this function errors
% (identifier select_lambda_effective_dimension:Unattainable); it never
% substitutes a fallback lambda. The root is found in log(lambda) on a bracket
% with no grid and no lower floor. The rule uses training-sample information
% only: the payoff panel and K_h of the sample being fitted.
%
% Inputs
%   payoff  : build_payoff_factor(alpha) for the Nh-by-T payoff panel alpha
%   K_h     : Nh-by-Nh RKHS norm matrix (symmetric positive definite)
%   T       : number of holding periods (dates); must equal payoff.T
%   c       : positive target constant (default 1)
%
% Output sel (saved with the fitted state):
%   lambda, target, N_at_lambda, residual, eigenvalues (descending, clipped
%   at 0, at most min(Nh,T) nonzero), numerical_rank, rank_tolerance,
%   rank_upper_bound = min(Nh, T), discarded_eigenvalue_ratio,
%   min_raw_eigenvalue, whitening_asymmetry (0: no whitened matrix is
%   formed), rcond_K_h, bracket, solver.

if nargin < 4 || isempty(c), c = 1; end

assert(isstruct(payoff) && all(isfield(payoff, {'F','T','Nh'})), ...
    'select_lambda_effective_dimension:BadPayoff', ...
    'payoff must be build_payoff_factor(alpha) (the payoff panel, not Q_alpha).');
Nh = payoff.Nh;
assert(isequal(size(K_h), [Nh Nh]) && size(payoff.F, 2) == Nh, ...
    'select_lambda_effective_dimension:BadSize', ...
    'K_h must be Nh-by-Nh with Nh the payoff dimension.');
assert(all(isfinite(payoff.F(:))) && all(isfinite(K_h(:))), ...
    'select_lambda_effective_dimension:NonFinite', 'The payoff factor and K_h must be finite.');
assert(isscalar(T) && T >= 1 && T == floor(T), ...
    'select_lambda_effective_dimension:BadT', 'T must be a positive integer.');
assert(T == payoff.T, 'select_lambda_effective_dimension:BadT', ...
    'T (%d) differs from the payoff panel length (%d).', T, payoff.T);
assert(isscalar(c) && isfinite(c) && c > 0, ...
    'select_lambda_effective_dimension:BadC', 'c must be a positive finite scalar.');

K_h = 0.5 * (K_h + K_h.');

% -- Generalized eigenvalues of (Q_alpha, K_h): squared singular values of F R^{-1}
[R, flag] = chol(K_h);
assert(flag == 0, 'select_lambda_effective_dimension:KhNotPD', ...
    'K_h is not numerically positive definite (chol failed at column %d).', flag);
rcondK = rcond(K_h);

muRaw = svd(payoff.F / R).^2;                  % eig of R^{-T} Q_alpha R^{-1}
whiteningAsymmetry = 0;                        % no whitened matrix is formed
muMax = max([muRaw; 0]);
assert(muMax > 0, 'select_lambda_effective_dimension:ZeroGram', ...
    'G_T is numerically zero.');
mu = zeros(Nh, 1);
mu(1:numel(muRaw)) = sort(muRaw, 'descend');
rankBound = min(Nh, T);
discardedRatio = 0;
if rankBound < Nh
    discardedRatio = mu(rankBound + 1) / muMax;
    mu(rankBound + 1:end) = 0;
end

% Relative numerical-rank tolerance.
rankTol = 1e-12;
numRank = sum(mu > rankTol * muMax);
target = c * T^(1/3);

if ~(target > 0 && target < numRank)
    error('select_lambda_effective_dimension:Unattainable', ...
        ['Target effective dimension %.6g = %.6g * T^(1/3), T = %d, must lie ' ...
         'strictly between 0 and the numerical rank of G_T (%d); no finite ' ...
         'positive lambda solves N_T(lambda) = target.'], target, c, T, numRank);
end

% -- Scalar root in log(lambda) ---------------------------------------------
gap = @(s) effective_dimension(mu, exp(s)) - target;
lo = log(muMax) - 40;              % N(lo) ~ numRank > target
hi = log(sum(mu)) + 40;            % N(hi) ~ 0 < target
gLo = gap(lo); gHi = gap(hi);
assert(gLo > 0 && gHi < 0, 'select_lambda_effective_dimension:Bracket', ...
    'Root bracket failed: gap(lo) = %.3g, gap(hi) = %.3g.', gLo, gHi);
[sStar, gStar, exitflag, out] = fzero(gap, [lo hi], ...
    optimset('TolX', 1e-13, 'Display', 'off'));   % root tolerance in log(lambda)
assert(exitflag == 1, 'select_lambda_effective_dimension:NoConvergence', ...
    'fzero failed to converge (exitflag %d).', exitflag);

lambda = exp(sStar);
Nstar = effective_dimension(mu, lambda);
residual = Nstar - target;
assert(isfinite(lambda) && lambda > 0, ...
    'select_lambda_effective_dimension:BadLambda', ...
    'Selected lambda is not finite and positive.');
% Acceptance tolerance on the effective-dimension residual.
assert(abs(residual) < 1e-8 * max(1, target), ...
    'select_lambda_effective_dimension:Residual', ...
    'Effective-dimension residual %.3g exceeds tolerance.', residual);

sel = struct();
sel.method = 'eff_dim';
sel.rule = 'N_T(lambda) = c * T^(1/3); mu = eig(G_T/T) = generalized eig(Q_alpha, K_h)';
sel.c = c;
sel.T = T;
sel.target = target;
sel.lambda = lambda;
sel.N_at_lambda = Nstar;
sel.residual = residual;
sel.eigenvalues = mu;
sel.numerical_rank = numRank;
sel.rank_tolerance = rankTol;
sel.rank_upper_bound = rankBound;
sel.discarded_eigenvalue_ratio = discardedRatio;
sel.min_raw_eigenvalue = min(muRaw);
sel.whitening_asymmetry = whiteningAsymmetry;
sel.rcond_K_h = rcondK;
sel.bracket = exp([lo hi]);
sel.solver = struct('name', 'fzero_log_lambda', 'TolX', 1e-13, ...
    'iterations', out.iterations, 'funcCount', out.funcCount, 'gap', gStar);

end

function N = effective_dimension(mu, lambda)
%% N_T(lambda) = sum_j mu_j / (mu_j + lambda).
N = sum(mu ./ (mu + lambda));
end
