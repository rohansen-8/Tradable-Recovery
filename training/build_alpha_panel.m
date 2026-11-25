function alphaPanel = build_alpha_panel(cfg, df, mon_grid, exo_grid)
%% Build alpha = [alpha_0,...,alpha_{T-1}] for the restricted basis.
%
% R^pi_{t+1}(h) = alpha_t' h.
%
% Interpretation:
%   alpha_t' * h_star is the realized fitted replicating_portfolio payoff.
%   The empirical payoff SDF is M_payoff_t = 1 - alpha_t' * h_star.
%   This is distinct from the plotted function-value curve of h(m,v).
%
% Naming convention:
%   h_star                 : restricted-grid coefficient vector
%   replicating_portfolio  : fitted payoff alpha_t' * h_star
%   M_payoff               : payoff SDF, 1 - replicating_portfolio
% If cfg.has_exo = false, alpha_t = zeta_t.
% If cfg.has_exo = true,  alpha_t = k_t^V \otimes zeta_t.

    if nargin < 2 || isempty(df)
        df = readtable(cfg.dataset.dfFile);
    end
    if nargin < 3 || isempty(mon_grid)
        mon_grid = default_mon_grid(cfg, df);
    end
    if nargin < 4
        exo_grid = [];
    end
    if cfg.has_exo && isempty(exo_grid)
        exo_grid = default_exo_grid(cfg, df);
    end

    mon_grid = mon_grid(:);
    ell_m = cfg.kernel.endo.lengthscale;
    if isempty(ell_m)
        ell_m = default_lengthscale(mon_grid);
    end

    zbar = cfg.reference_point;

    names = df.Properties.VariableNames;
    timeName = pick_name(names, {'t', 'date'});
    mName = pick_name(names, {'m', 'frwd_moneyness'});
    ROName = pick_name(names, {'RO', 'OPT_NET_R_PAYOFF'});
    wName = pick_name(names, {'w', 'weight'});

    df = sortrows(df, {timeName, mName});

    m = df.(mName);
    RO = df.(ROName);
    w = df.(wName);
    Delta = get_delta(df);

    [dateGroups, uniqueDates] = findgroups(df.(timeName));
    T = numel(uniqueDates);
    n = numel(mon_grid);
    p = max(1, size(as_matrix(exo_grid), 1));
    Nh = (n + 1) * p;

    alpha = zeros(Nh, T);
    zeta = zeros(n + 1, T);
    kV = zeros(p, T);
    J_t = zeros(T, 1);

    exoObs = [];
    if cfg.has_exo
        exoObs = as_matrix(df{:, cfg.exo_cols});
    end

    for t = 1:T
        idx = find(dateGroups == t);
        J_t(t) = numel(idx);

        Delta_t = unique_scalar(Delta(idx), 'Delta_t');
        m_t = m(idx);
        payoff_t = w(idx) .* RO(idx);

        zeta_t = zeros(n + 1, 1);
        zeta_t(1) = Delta_t * matern72_dxy(zbar, zbar, ell_m) ...
                  + sum(payoff_t.' .* matern72_dxyy(zbar, m_t.', ell_m));

        zeta_t(2:end) = Delta_t * matern72_dxyy(zbar, mon_grid, ell_m) ...
                      + matern72_dxxyy(mon_grid, m_t.', ell_m) * payoff_t;

        zeta(:, t) = zeta_t;

        if cfg.has_exo
            v_t = exoObs(idx(1), :);
            kV_t = exo_kernel_vector(v_t, exo_grid, cfg);
            kV(:, t) = kV_t;
            alpha(:, t) = kron(kV_t, zeta_t);
        else
            kV(:, t) = 1;
            alpha(:, t) = zeta_t;
        end
    end

    alphaPanel = struct();
    alphaPanel.alpha = alpha;
    alphaPanel.zeta = zeta;
    alphaPanel.kV = kV;
    alphaPanel.mon_grid = mon_grid;
    alphaPanel.exo_grid = exo_grid;
    alphaPanel.uniqueDates = uniqueDates;
    alphaPanel.J_t = J_t;
    alphaPanel.T = T;
    alphaPanel.Nh = Nh;
end

function mon_grid = default_mon_grid(cfg, df)
    if isfield(cfg.training, 'mon_grid') && ~isempty(cfg.training.mon_grid)
        mon_grid = cfg.training.mon_grid(:);
        return;
    end

    if isfield(cfg.training, 'estimator_domain') && ~isempty(cfg.training.estimator_domain)
        nGrid = get_training_field(cfg, 'n_m_grid', 81);
        mon_grid = linspace(cfg.training.estimator_domain(1), ...
            cfg.training.estimator_domain(2), nGrid).';
        return;
    end

    if isfield(cfg, 'sim') && all(isfield(cfg.sim, {'m_min', 'm_max', 'J'}))
        mon_grid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.sim.J).';
        return;
    end

    names = df.Properties.VariableNames;
    mName = pick_name(names, {'m', 'frwd_moneyness'});
    m = df.(mName);

    nGrid = get_training_field(cfg, 'n_m_grid', 81);
    lo = quantile(m, 0.01);
    hi = quantile(m, 0.99);
    mon_grid = linspace(lo, hi, nGrid).';
end

function exo_grid = default_exo_grid(cfg, df)
    if isfield(cfg.training, 'exo_grid') && ~isempty(cfg.training.exo_grid)
        exo_grid = cfg.training.exo_grid;
        return;
    end

    X = as_matrix(df{:, cfg.exo_cols});
    nGrid = get_training_field(cfg, 'n_exo_grid', 5);

    gridType = 'quantile';
    if isfield(cfg.training, 'exo_grid_type') && ~isempty(cfg.training.exo_grid_type)
        gridType = lower(cfg.training.exo_grid_type);
    end

    switch gridType
        case 'linspace'
            exo_grid = zeros(nGrid, size(X, 2));
            for j = 1:size(X, 2)
                lo = min(X(:, j));
                hi = max(X(:, j));

                if ~isfinite(lo) || ~isfinite(hi) || lo == hi
                    error('build_alpha_panel:BadExoGrid', ...
                        'Cannot build equally spaced exogenous grid for column %d.', j);
                end

                exo_grid(:, j) = linspace(lo, hi, nGrid).';
            end

        case 'quantile'
            if size(X, 2) == 1 && isfield(cfg.testing, 'volLevels') && ~isempty(cfg.testing.volLevels)
                exo_grid = cfg.testing.volLevels(:);
                return;
            end

            q = linspace(0.05, 0.95, nGrid).';
            exo_grid = zeros(nGrid, size(X, 2));
            for j = 1:size(X, 2)
                exo_grid(:, j) = quantile(X(:, j), q);
            end
            exo_grid = unique(exo_grid, 'rows');

        otherwise
            error('build_alpha_panel:BadExoGridType', ...
                'Unsupported cfg.training.exo_grid_type: %s.', gridType);
    end
end

function Delta = get_delta(df)
    names = df.Properties.VariableNames;
    if any(strcmp(names, 'Delta'))
        Delta = df.Delta;
    elseif any(strcmp(names, 'R'))
        Delta = df.R - 1;
    elseif any(strcmp(names, 'SP500_ret'))
        Delta = df.SP500_ret;
    elseif any(strcmp(names, 'underlying_ret'))
        Delta = df.underlying_ret;
    else
        error('build_alpha_panel:MissingDelta', ...
            'Could not find Delta, R, SP500_ret, or underlying_ret.');
    end
end

function k = exo_kernel_vector(v, exo_grid, cfg)
    X = as_matrix(exo_grid);
    v = reshape(v, 1, []);

    ell = cfg.kernel.exo.lengthscale;
    if isempty(ell)
        ell = default_exo_lengthscale(X);
    end
    if isscalar(ell)
        ell = repmat(ell, 1, size(X, 2));
    end

    switch lower(cfg.kernel.exo.type)
        case 'laplace'
            k = ones(size(X, 1), 1);
            for j = 1:size(X, 2)
                k = k .* laplace_k(X(:, j), v(j), ell(j));
            end
        otherwise
            error('build_alpha_panel:BadExoKernel', ...
                'Unsupported exogenous kernel type: %s.', cfg.kernel.exo.type);
    end
end

function ell = default_lengthscale(x)
% Median pairwise-distance lengthscale for one-dimensional grids.

    x = x(:);
    D = abs(x - x.');
    d = D(triu(true(numel(x)), 1));
    d = d(isfinite(d) & d > 0);

    ell = median(d);

    if ~isfinite(ell) || ell <= 0
        ell = 0.10;
    end
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

function value = get_training_field(cfg, name, defaultValue)
    if isfield(cfg, 'training') && isfield(cfg.training, name) && ~isempty(cfg.training.(name))
        value = cfg.training.(name);
    else
        value = defaultValue;
    end
end

function name = pick_name(names, candidates)
    for j = 1:numel(candidates)
        if any(strcmp(names, candidates{j}))
            name = candidates{j};
            return;
        end
    end
    error('build_alpha_panel:MissingColumn', ...
        'Missing one of these columns: %s.', strjoin(candidates, ', '));
end

function x = unique_scalar(x, label)
    x0 = x(1);
    if any(abs(x - x0) > 1e-12)
        error('build_alpha_panel:NotDateConstant', '%s is not constant within date.', label);
    end
    x = x0;
end

function X = as_matrix(X)
    if isempty(X)
        X = [];
    elseif isvector(X)
        X = X(:);
    end
end
