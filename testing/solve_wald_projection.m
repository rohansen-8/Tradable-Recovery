function out = solve_wald_projection(theta_hat, Omega_hat, S, T, opts)
%% One-sided Wald statistic for H0: theta in R_+^r.
%
% W_T = T min_{w >= 0} (theta_hat - w)' Omega_hat^dagger (theta_hat - w).
%
% The computation is based on Omega_hat itself. The optional S input is
% accepted but not used to define the statistic.
% When repeated projections share the same Omega_hat, opts.geometry may hold
% the precomputed whitening and NNLS geometry returned by
% build_wald_projection_geometry.m.

    if nargin < 5 || isempty(opts)
        opts = struct();
    end
    opts = default_opts(opts);

    theta_hat = theta_hat(:);
    r = numel(theta_hat);

    if nargin < 2 || isempty(Omega_hat) || size(Omega_hat, 1) ~= r || size(Omega_hat, 2) ~= r
        error('solve_wald_projection:BadOmega', 'Omega_hat must be r-by-r.');
    end
    if nargin < 4 || isempty(T) || ~isscalar(T) || T <= 0
        error('solve_wald_projection:BadSampleSize', 'T must be a positive scalar.');
    end

    Omega_hat = 0.5 * (Omega_hat + Omega_hat.');

    if all(theta_hat >= -opts.earlyExitTol)
        out = struct();
        out.W_T = 0;
        out.w_star = max(theta_hat, 0);
        out.status = 'ok';
        out.early_exit = true;
        out.solver_method = 'none';
        out.solver_status = struct('exitflag', NaN, 'rank', 0);
        return;
    end

    geometry = opts.geometry;
    if isempty(geometry)
        geometry = build_wald_projection_geometry(Omega_hat, opts);
    else
        validate_geometry(geometry, Omega_hat, r);
    end

    if ~geometry.valid
        out = failed_output(r, geometry.status);
        return;
    end

    A = geometry.A;
    b = A * theta_hat;

    [x_scaled, ~, ~, exitflag] = lsqnonneg( ...
        geometry.As, b, geometry.nnlsOpts);

    w_star = x_scaled ./ geometry.scale.';
    w_star = max(w_star, 0);

    resid = A * (theta_hat - w_star);
    W_T = T * (resid.' * resid);

    out = struct();
    out.W_T = max(real(W_T), 0);
    out.w_star = w_star;
    out.status = ternary(exitflag > 0 && isfinite(W_T), 'ok', 'failed');
    out.early_exit = false;
    out.solver_method = 'eig_omega_nnls';
    out.solver_status = struct( ...
        'exitflag', exitflag, ...
        'rank', geometry.rank, ...
        'eigenvalues', geometry.eigenvalues, ...
        'svdTolRel', geometry.svdTolRel);
end

function opts = default_opts(opts)
%% Fill numerical options for the one-sided projection.

    if ~isfield(opts, 'earlyExitTol'), opts.earlyExitTol = 1e-10; end
    if ~isfield(opts, 'svdTolRel'), opts.svdTolRel = 1e-10; end
    if ~isfield(opts, 'nnlsTolX'), opts.nnlsTolX = 1e-8; end
    if ~isfield(opts, 'nnlsMaxIter'), opts.nnlsMaxIter = 5e4; end
    if ~isfield(opts, 'geometry'), opts.geometry = []; end
end

function validate_geometry(geometry, Omega_hat, r)
%% Guard against accidentally reusing geometry for a different covariance.

    required = {'r', 'Omega_hat', 'valid', 'status', 'A', 'As', ...
        'scale', 'nnlsOpts', 'rank', 'eigenvalues', 'svdTolRel'};

    for j = 1:numel(required)
        if ~isfield(geometry, required{j})
            error('solve_wald_projection:BadGeometry', ...
                'Precomputed geometry is missing field %s.', required{j});
        end
    end

    if geometry.r ~= r
        error('solve_wald_projection:BadGeometry', ...
            'Precomputed geometry has the wrong dimension.');
    end

    err = norm(geometry.Omega_hat - Omega_hat, 'fro');
    scale = max(1, norm(Omega_hat, 'fro'));

    if err > 1e-12 * scale
        error('solve_wald_projection:BadGeometry', ...
            'Precomputed geometry does not match Omega_hat.');
    end
end

function out = failed_output(r, reason)
%% Return the standard failed-solver record.

    out = struct();
    out.W_T = NaN;
    out.w_star = NaN(r, 1);
    out.status = reason;
    out.early_exit = false;
    out.solver_method = 'eig_omega_nnls';
    out.solver_status = struct();
end

function y = ternary(cond, a, b)
%% Return one of two values from a scalar condition.

    if cond
        y = a;
    else
        y = b;
    end
end
