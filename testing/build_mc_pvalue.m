function [p_value, critical_value, W_mc, meta] = build_mc_pvalue(W_T, Omega_hat, S, T, alpha, nMC, opts)
%% Monte Carlo p-value for the one-sided Wald statistic.
%
% Draw Y ~ N(0, Omega_hat / T), then recompute
%   W_T(Y) = T min_{w >= 0} (Y - w)' Omega_hat^dagger (Y - w).
%
% Omega_hat is fixed across all Monte Carlo draws, so its Gaussian draw factor
% and one-sided projection geometry are computed once and reused.
%
% The optional S input is accepted but not used.

    if nargin < 5 || isempty(alpha), alpha = 0.05; end
    if nargin < 6 || isempty(nMC), nMC = 10000; end
    if nargin < 7 || isempty(opts), opts = struct(); end
    opts = default_opts(opts);

    Omega_hat = 0.5 * (Omega_hat + Omega_hat.');
    drawFactor = build_draw_factor(Omega_hat, T);
    geometry = build_wald_projection_geometry(Omega_hat, opts.wald_opts);

    wald_opts = opts.wald_opts;
    wald_opts.geometry = geometry;

    W_mc = NaN(nMC, 1);
    status = strings(nMC, 1);

    if opts.parallel
        parfor b = 1:nMC
            [W_mc(b), status(b)] = one_draw( ...
                b, drawFactor, Omega_hat, T, opts.rng_seed, wald_opts);
        end
    else
        for b = 1:nMC
            [W_mc(b), status(b)] = one_draw( ...
                b, drawFactor, Omega_hat, T, opts.rng_seed, wald_opts);
        end
    end

    valid = isfinite(W_mc);
    if ~any(valid)
        p_value = NaN;
        critical_value = NaN;
    else
        p_value = (sum(W_mc(valid) >= W_T) + 1) / (sum(valid) + 1);
        critical_value = quantile(W_mc(valid), 1 - alpha);
    end

    meta = struct();
    meta.nMC = nMC;
    meta.n_valid = sum(valid);
    meta.n_failed = sum(~valid);
    meta.status = status;
    meta.parallel = opts.parallel;
    meta.rng_seed = opts.rng_seed;
    meta.geometry_reused = true;
end

function [Wb, status] = one_draw( ...
    b, drawFactor, Omega_hat, T, rng_seed, wald_opts)
%% Generate one null draw and evaluate it using cached projection geometry.

    Wb = NaN;
    status = "failed";

    try
        if ~isempty(rng_seed)
            rng(rng_seed + b, 'twister');
        end

        Y = drawFactor * randn(size(drawFactor, 2), 1);
        stat = solve_wald_projection(Y, Omega_hat, [], T, wald_opts);
        Wb = stat.W_T;

        if isfield(stat, 'status')
            status = string(stat.status);
        else
            status = "ok";
        end
    catch
        Wb = NaN;
    end
end

function drawFactor = build_draw_factor(Omega_hat, T)
%% Factor Omega_hat / T once for all Gaussian Monte Carlo draws.

    V = Omega_hat / T;
    [C, p] = chol(V, 'lower');

    if p == 0
        drawFactor = C;
        return;
    end

    [E, D] = eig(V);
    d = max(real(diag(D)), 0);
    drawFactor = real(E) * diag(sqrt(d));
end

function opts = default_opts(opts)
%% Fill Monte Carlo calibration options.

    if ~isfield(opts, 'parallel'), opts.parallel = false; end
    if ~isfield(opts, 'rng_seed'), opts.rng_seed = []; end
    if ~isfield(opts, 'wald_opts'), opts.wald_opts = struct(); end
end
