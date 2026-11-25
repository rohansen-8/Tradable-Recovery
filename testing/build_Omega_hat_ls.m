function [Omega_hat, S, info] = build_Omega_hat_ls(model_state, L_Z)
%% Least-squares plug-in covariance for finite-grid shape tests.
%
% F_t = (1 - eta_t) psi_t - lambda h_star,
% S_tj = <F_t, u_j>,  u_j = Sigma_lambda^{-1} phi_j,
% Omega_hat = S' S / T.
%
% When model_state.solve_factor is present (the triangular R_sys of
% solve_projection_qr, R_sys' R_sys = Q_alpha + lambda K_h), u_j is obtained
% from it by two triangular solves and the normal-equation matrix is not
% formed. Otherwise a Cholesky factorization of Q_alpha + lambda K_h is used.

    alpha = model_state.alpha;
    K_h = model_state.K_h;
    h_star = model_state.h_star(:);
    lambda = model_state.lambda;

    T = size(alpha, 2);

    useFactor = isfield(model_state, 'solve_factor') && ~isempty(model_state.solve_factor);
    if useFactor
        R_sys = model_state.solve_factor;
        U = R_sys \ (R_sys.' \ L_Z);
        conditionNumber = cond(R_sys)^2;
    else
        if isfield(model_state, 'Q_alpha') && ~isempty(model_state.Q_alpha)
            Q_alpha = model_state.Q_alpha;
        else
            Q_alpha = (alpha * alpha.') / T;
        end

        A = Q_alpha + lambda * K_h;
        A = 0.5 * (A + A.');

        U = solve_symmetric(A, L_Z);
        conditionNumber = cond(A);
    end

    eta = alpha.' * h_star;
    theta_hat = L_Z.' * h_star;
    psi_u = alpha.' * U;
    h_u = h_star.' * K_h * U;

    S = bsxfun(@times, 1 - eta, psi_u) - ones(T, 1) * (lambda * h_u);
    Omega_hat = (S.' * S) / T;
    Omega_hat = 0.5 * (Omega_hat + Omega_hat.');

    info = struct();
    info.T = T;
    info.eta = eta;
    info.theta_hat = theta_hat;
    info.condition_number = conditionNumber;
end

function X = solve_symmetric(A, B)
    [L, flag] = chol(A, 'lower');
    if flag == 0
        X = L' \ (L \ B);
    else
        X = A \ B;
    end
end
