function [theta_hat, L_Z, info] = build_theta_hat(cfg, model_state, grid, testType, exo)
%% Build theta_hat = L_Z' h_star on a finite moneyness grid.
%
% All derivatives are with respect to raw forward moneyness. The shape
% restrictions are defined directly on the fitted function h:
%
% mono: theta(z) = h_m(z, .)   >= 0 under H0.
% conv: theta(z) = -h_mm(z, .) >= 0 under H0.
% conc: theta(z) = h_mm(z, .)  >= 0 under H0.

    if nargin < 5 || isempty(exo)
        exo = build_exo_representer(cfg, model_state);
    end

    Z = get_grid_vector(grid);
    testType = char(testType);

    h_star = model_state.h_star(:);
    mon_grid = model_state.mon_grid(:);
    zbar = model_state.reference_point;
    ell_m = model_state.ls_endo;
    e = exo.e(:);

    n = numel(mon_grid);
    p = numel(e);
    r = numel(Z);

    if numel(h_star) ~= (n + 1) * p
        error('build_theta_hat:BadDimensions', ...
            'h_star, mon_grid, and exogenous loading dimensions are inconsistent.');
    end

    L_Z = zeros((n + 1) * p, r);

    for j = 1:r
        z = Z(j);

        k11 = matern72_dxy(zbar, z, ell_m);
        k21 = matern72_dxxy(mon_grid, z, ell_m);
        k21 = k21(:);

        switch lower(testType)
            case 'mono'
                d = [k11; k21];

            case 'conv'
                k12 = matern72_dxyy(zbar, z, ell_m);
                k22 = matern72_dxxyy(mon_grid, z, ell_m);
                k22 = k22(:);
                d = -[k12; k22];

            case 'conc'
                k12 = matern72_dxyy(zbar, z, ell_m);
                k22 = matern72_dxxyy(mon_grid, z, ell_m);
                k22 = k22(:);
                d = [k12; k22];

            otherwise
                error('build_theta_hat:BadTestType', 'testType must be mono, conv, or conc.');
        end

        L_Z(:, j) = kron(e, d);
    end

    theta_hat = L_Z.' * h_star;

    info = struct();
    info.grid = Z;
    info.testType = testType;
    info.exo = exo;
    info.n = n;
    info.p = p;
    info.r = r;
end

function Z = get_grid_vector(grid)
    if isstruct(grid)
        if isfield(grid, 'm')
            Z = grid.m;
        else
            error('build_theta_hat:BadGrid', 'Grid struct must contain field m.');
        end
    else
        Z = grid;
    end

    Z = Z(:);
    if any(~isfinite(Z)) || any(Z <= 0)
        error('build_theta_hat:BadGrid', 'The moneyness test grid must be positive and finite.');
    end
end
