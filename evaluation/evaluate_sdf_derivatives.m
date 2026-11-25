function out = evaluate_sdf_derivatives(cfg, model_state, m_grid, exo)
%% Evaluate first and second raw-forward-moneyness derivatives of h.
%
% Shape convention (defined directly on h):
%   mono : h_m >= 0
%   conv : -h_mm >= 0
%   conc : h_mm >= 0
%
% Output:
%   out.h_m     : first derivative of h
%   out.h_mm    : second derivative of h
%   out.M_m     : -h_m  (derivative of out.M of evaluate_sdf_grid; diagnostics only)
%   out.M_mm    : -h_mm (second derivative of out.M; diagnostics only)

    if nargin < 4 || isempty(exo)
        exo = build_exo_representer(cfg, model_state);
    end

    m_grid = m_grid(:);
    if any(~isfinite(m_grid)) || any(m_grid <= 0)
        error('evaluate_sdf_derivatives:BadGrid', ...
            'm_grid must contain positive finite forward moneyness values.');
    end

    h_m = build_theta_hat(cfg, model_state, m_grid, 'mono', exo);
    M_mm = build_theta_hat(cfg, model_state, m_grid, 'conv', exo);

    h_m = h_m(:);
    M_mm = M_mm(:);

    h_mm = -M_mm;
    M_m = -h_m;

    out = struct();
    out.m = m_grid;
    out.h_m = h_m;
    out.h_mm = h_mm;
    out.M_m = M_m;
    out.M_mm = M_mm;
    out.exo = exo;
end
