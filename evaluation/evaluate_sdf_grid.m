function out = evaluate_sdf_grid(cfg, model_state, m_grid, exo)
%% Evaluate the fitted function h on a raw forward-moneyness grid.
%
% Kernel evaluation is always in raw forward moneyness. out.M is the
% function-value curve plotted in the conditional SDF figures; it is used for
% plotting only and never by the shape tests.
%
% Inputs:
%   cfg         : project config
%   model_state : fitted model_state from train_projection_model
%   m_grid      : evaluation grid in raw forward moneyness
%   exo         : optional exogenous representer from build_exo_representer
%
% Output:
%   out.m       : moneyness grid
%   out.h       : fitted h(m,.)
%   out.M       : plotted curve 1 - h(m,.), figures only

    if nargin < 4 || isempty(exo)
        exo = build_exo_representer(cfg, model_state);
    end

    m_grid = m_grid(:);
    if any(~isfinite(m_grid)) || any(m_grid <= 0)
        error('evaluate_sdf_grid:BadGrid', ...
            'm_grid must contain positive finite forward moneyness values.');
    end

    h_star = model_state.h_star(:);
    mon_grid = model_state.mon_grid(:);
    zbar = model_state.reference_point;
    ell_m = model_state.ls_endo;
    e = exo.e(:);

    n = numel(mon_grid);
    p = numel(e);

    if numel(h_star) ~= (n + 1) * p
        error('evaluate_sdf_grid:BadDimensions', ...
            'h_star, mon_grid, and exogenous loading dimensions are inconsistent.');
    end

    H = reshape(h_star, n + 1, p);

    Bm = [matern72_dy(m_grid, zbar, ell_m), ...
          matern72_dyy(m_grid, mon_grid.', ell_m)];

    h = Bm * H * e;
    M = 1 - h;

    out = struct();
    out.m = m_grid;
    out.h = h;
    out.M = M;
    out.exo = exo;
end
