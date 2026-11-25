function [K_h, K_M, K_V, basis] = build_kernel_norm_matrix(cfg, mon_grid, exo_grid)
%% Build the restricted RKHS norm matrix K_h.
%
% If cfg.has_exo = false, K_h = K_M.
% If cfg.has_exo = true,  K_h = K_V \otimes K_M.

    if nargin < 2 || isempty(mon_grid)
        error('build_kernel_norm_matrix:MissingGrid', 'mon_grid is required.');
    end
    if nargin < 3
        exo_grid = [];
    end

    mon_grid = mon_grid(:);
    zbar = cfg.reference_point;
    ell_m = cfg.kernel.endo.lengthscale;
    if isempty(ell_m)
        ell_m = default_lengthscale(mon_grid);
    end

    n = numel(mon_grid);

    % Jitter is added to K_M and K_V separately, before the Kronecker
    % product, so the regularizer is kron(K_V + jI, K_M + jI), not
    % kron(K_V,K_M) + jI. The empirical and simulation estimators set
    % cfg.kernel.jitter = 1e-10.
    jitter = 0;
    if isfield(cfg, 'kernel') && isfield(cfg.kernel, 'jitter') ...
            && ~isempty(cfg.kernel.jitter)
        jitter = double(cfg.kernel.jitter);
        assert(isscalar(jitter) && isfinite(jitter) && jitter >= 0, ...
            'cfg.kernel.jitter must be a nonnegative finite scalar.');
    end

    K_M = zeros(n + 1, n + 1);
    K_M(1, 1) = matern72_dxy(zbar, zbar, ell_m);
    K_M(1, 2:end) = matern72_dxyy(zbar, mon_grid.', ell_m);
    K_M(2:end, 1) = matern72_dxxy(mon_grid, zbar, ell_m);
    K_M(2:end, 2:end) = matern72_dxxyy(mon_grid, mon_grid.', ell_m);
    K_M = 0.5 * (K_M + K_M.');
    if jitter > 0
        K_M = K_M + jitter * eye(n + 1);
    end

    if cfg.has_exo
        if isempty(exo_grid)
            error('build_kernel_norm_matrix:MissingExoGrid', ...
                'exo_grid is required when cfg.has_exo = true.');
        end
        K_V = exo_kernel_matrix(exo_grid, exo_grid, cfg);
        K_V = 0.5 * (K_V + K_V.');
        if jitter > 0
            K_V = K_V + jitter * eye(size(K_V, 1));
        end
        K_h = kron(K_V, K_M);
    else
        K_V = 1;
        K_h = K_M;
    end

    K_h = 0.5 * (K_h + K_h.');

    basis = struct();
    basis.mon_grid = mon_grid;
    basis.exo_grid = exo_grid;
    basis.reference_point = zbar;
    basis.ls_endo = ell_m;
    if cfg.has_exo
        basis.ls_exo = get_exo_lengthscale(exo_grid, cfg);
    else
        basis.ls_exo = [];
    end
    basis.n = n;
    basis.p = size(K_V, 1);
end

function K = exo_kernel_matrix(X, Y, cfg)
    X = as_matrix(X);
    Y = as_matrix(Y);

    if size(X, 2) ~= size(Y, 2)
        error('build_kernel_norm_matrix:BadExoGrid', ...
            'exo_grid inputs must have the same number of columns.');
    end

    ell = cfg.kernel.exo.lengthscale;
    if isempty(ell)
        ell = default_exo_lengthscale(X);
    end
    if isscalar(ell)
        ell = repmat(ell, 1, size(X, 2));
    end

    switch lower(cfg.kernel.exo.type)
        case 'laplace'
            K = ones(size(X, 1), size(Y, 1));
            for k = 1:size(X, 2)
                K = K .* laplace_k(X(:, k), Y(:, k).', ell(k));
            end
        otherwise
            error('build_kernel_norm_matrix:BadExoKernel', ...
                'Unsupported exogenous kernel type: %s.', cfg.kernel.exo.type);
    end
end

function ell = default_lengthscale(x)
%% Return the empirical median-pairwise-distance lengthscale.

    ell = median_pairwise_lengthscale(x);
end

function ell = default_exo_lengthscale(X)
% Median pairwise-distance lengthscale for each exogenous coordinate.

    ell = zeros(1, size(X, 2));

    for j = 1:size(X, 2)
        x = X(:, j);
        x = x(:);

        D = abs(x - x.');
        d = D(triu(true(numel(x)), 1));
        d = d(isfinite(d) & d > 0);

        ell(j) = median(d);

        if ~isfinite(ell(j)) || ell(j) <= 0
            ell(j) = 1;
        end
    end
end

function ell = get_exo_lengthscale(X, cfg)
    ell = cfg.kernel.exo.lengthscale;
    if isempty(ell)
        ell = default_exo_lengthscale(as_matrix(X));
    end
end

function X = as_matrix(X)
    if isempty(X)
        X = [];
    elseif isvector(X)
        X = X(:);
    end
end
