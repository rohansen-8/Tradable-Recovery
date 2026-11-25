function [h_star, R_sys, solver_status] = solve_projection_qr(payoff, K_h, lambda)
%% Solve the regularized projection estimator by QR least squares.
%
% h_star solves (Q_alpha + lambda K_h) h = alpha_bar,
% equivalently it minimizes T^{-1} sum_t (1 - alpha_t' h)^2 + lambda h' K_h h.
% With payoff = build_payoff_factor(alpha) and K_h = R_K' R_K this is the
% least-squares problem
%     min_h || [b; 0] - [F; sqrt(lambda) R_K] h ||,
% solved by one QR factorization of [F, b; sqrt(lambda) R_K, 0]. Its leading
% triangular block R_sys satisfies R_sys' R_sys = Q_alpha + lambda K_h, so it
% also gives every action of (Q_alpha + lambda K_h)^{-1} needed downstream
% (derivative scores, HAC covariance): X = R_sys \ (R_sys' \ B). The normal
% equations are never formed.

assert(isstruct(payoff) && all(isfield(payoff, {'F','b','Nh'})), ...
    'solve_projection_qr:BadPayoff', 'payoff must come from build_payoff_factor.');
Nh = payoff.Nh;
assert(isequal(size(K_h), [Nh Nh]), 'solve_projection_qr:BadSize', ...
    'K_h must be Nh-by-Nh.');
assert(isscalar(lambda) && isfinite(lambda) && lambda > 0, ...
    'solve_projection_qr:BadLambda', 'lambda must be finite and positive.');

[R_K, flag] = chol(0.5 * (K_h + K_h.'));
assert(flag == 0, 'solve_projection_qr:KhNotPD', ...
    'K_h is not numerically positive definite.');

[~, R] = qr([payoff.F, payoff.b; sqrt(lambda) * R_K, zeros(Nh, 1)], 0);
R_sys = R(1:Nh, 1:Nh);
h_star = R_sys \ R(1:Nh, Nh + 1);

solver_status = struct('flag', 0, 'method', 'qr');

end
