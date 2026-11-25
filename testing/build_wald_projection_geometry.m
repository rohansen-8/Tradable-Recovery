function geometry = build_wald_projection_geometry(Omega_hat, opts)
%% Precompute covariance geometry for repeated one-sided Wald projections.
%
% For fixed Omega_hat, the eigenspace whitening matrix, NNLS column scaling,
% and solver options are identical for every projected vector.  Caching these
% objects avoids repeating the same O(r^3) decomposition during Monte Carlo
% calibration without changing the Wald statistic or its null distribution.

    if nargin < 2 || isempty(opts)
        opts = struct();
    end
    opts = default_opts(opts);

    if isempty(Omega_hat) || size(Omega_hat, 1) ~= size(Omega_hat, 2)
        error('build_wald_projection_geometry:BadOmega', ...
            'Omega_hat must be a nonempty square matrix.');
    end

    Omega_hat = 0.5 * (Omega_hat + Omega_hat.');
    r = size(Omega_hat, 1);

    [V, D] = eig(Omega_hat);
    eigvals = real(diag(D));
    [eigvals, ord] = sort(eigvals, 'descend');
    V = real(V(:, ord));

    eigMax = max(eigvals);

    geometry = struct();
    geometry.r = r;
    geometry.Omega_hat = Omega_hat;
    geometry.valid = false;
    geometry.status = 'empty_covariance';
    geometry.A = [];
    geometry.As = [];
    geometry.scale = [];
    geometry.nnlsOpts = [];
    geometry.rank = 0;
    geometry.eigenvalues = eigvals;
    geometry.svdTolRel = opts.svdTolRel;

    if isempty(eigMax) || eigMax <= 0 || ~isfinite(eigMax)
        return;
    end

    keep = eigvals >= opts.svdTolRel * eigMax;
    if ~any(keep)
        keep(1) = true;
    end

    A = diag(1 ./ sqrt(eigvals(keep))) * V(:, keep).';
    scale = max(vecnorm(A, 2, 1), eps);
    As = bsxfun(@rdivide, A, scale);

    geometry.valid = true;
    geometry.status = 'ok';
    geometry.A = A;
    geometry.As = As;
    geometry.scale = scale;
    geometry.nnlsOpts = optimset( ...
        'Display', 'off', ...
        'TolX', opts.nnlsTolX, ...
        'MaxIter', opts.nnlsMaxIter);
    geometry.rank = sum(keep);
end

function opts = default_opts(opts)
%% Fill numerical options used by the existing Wald projection solver.

    if ~isfield(opts, 'svdTolRel'), opts.svdTolRel = 1e-10; end
    if ~isfield(opts, 'nnlsTolX'), opts.nnlsTolX = 1e-8; end
    if ~isfield(opts, 'nnlsMaxIter'), opts.nnlsMaxIter = 5e4; end
end
