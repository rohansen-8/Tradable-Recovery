function test_build_mc_pvalue_geometry_reuse
%% Verify cached MC geometry against the uncached reference computation.
%
% The cached routine must reproduce the same null draws and one-sided
% Wald statistics as recomputing the covariance geometry on every draw.  The
% test covers both positive-definite and rank-deficient covariance matrices.

    Omega_pd = [ ...
        1.00,  0.25, -0.10; ...
        0.25,  0.80,  0.20; ...
       -0.10,  0.20,  0.60];

    B = [1.0, 0.2; 0.4, 0.9; -0.3, 0.5];
    Omega_psd = B * B.';

    check_case(Omega_pd, 173, 80, 510001);
    check_case(Omega_psd, 173, 80, 520001);

    fprintf('test_build_mc_pvalue_geometry_reuse passed.\n');
end

function check_case(Omega_hat, T, nMC, seed)
%% Compare cached and uncached Monte Carlo paths for one covariance matrix.

    W_T = 1.25;
    alpha = 0.05;

    opts = struct();
    opts.parallel = false;
    opts.rng_seed = seed;
    opts.wald_opts = struct();

    [p_fast, cv_fast, W_fast, meta_fast] = build_mc_pvalue( ...
        W_T, Omega_hat, [], T, alpha, nMC, opts);

    [p_ref, cv_ref, W_ref] = reference_mc_pvalue( ...
        W_T, Omega_hat, T, alpha, nMC, opts);

    assert(isequal(isfinite(W_fast), isfinite(W_ref)));

    good = isfinite(W_ref);
    if any(good)
        maxErr = max(abs(W_fast(good) - W_ref(good)));
        assert(maxErr < 1e-10, ...
            'Cached and uncached MC Wald statistics differ by %.3e.', maxErr);
    end

    assert(abs(p_fast - p_ref) < 1e-14 || (isnan(p_fast) && isnan(p_ref)));
    assert(abs(cv_fast - cv_ref) < 1e-10 || (isnan(cv_fast) && isnan(cv_ref)));
    assert(meta_fast.geometry_reused);
end

function [p_value, critical_value, W_mc] = reference_mc_pvalue( ...
    W_T, Omega_hat, T, alpha, nMC, opts)
%% Reference path that deliberately recomputes fixed geometry every draw.

    W_mc = NaN(nMC, 1);

    for b = 1:nMC
        rng(opts.rng_seed + b, 'twister');
        Y = reference_draw_mc_theta(Omega_hat, T);
        stat = solve_wald_projection(Y, Omega_hat, [], T, opts.wald_opts);
        W_mc(b) = stat.W_T;
    end

    valid = isfinite(W_mc);

    if ~any(valid)
        p_value = NaN;
        critical_value = NaN;
    else
        p_value = (sum(W_mc(valid) >= W_T) + 1) / (sum(valid) + 1);
        critical_value = quantile(W_mc(valid), 1 - alpha);
    end
end

function Y = reference_draw_mc_theta(Omega_hat, T)
%% Direct Gaussian draw construction (Cholesky, eigen fallback if singular).

    V = 0.5 * (Omega_hat + Omega_hat.') / T;
    [C, p] = chol(V, 'lower');

    if p == 0
        Y = C * randn(size(V, 1), 1);
        return;
    end

    [E, D] = eig(V);
    d = max(real(diag(D)), 0);
    Y = real(E) * (sqrt(d) .* randn(size(V, 1), 1));
end
