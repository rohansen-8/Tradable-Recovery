function model_state = train_projection_model(cfg, df)
%% Train the least-squares SDF projection on the restricted RKHS basis.
%
% The fitted coefficient solves
%       (Q_alpha + lambda K_h) h_star = alpha_bar,
% where Q_alpha is the uncentered second moment of alpha_t.
%
% Lambda selection is the only step that depends on cfg.training.lambda_method
% (set by config_empirical; simulation configurations do not set it):
%   'eff_dim'  select_lambda_effective_dimension on the triangular payoff
%              factor of build_payoff_factor and K_h; the same factor gives
%              the QR least-squares solve of the same system
%              (solve_projection_qr), so the normal equations are not formed;
%   otherwise  one-step-ahead CV when use_cv is set or no lambda is given,
%              else the fixed cfg.training.lambda.
% The final solve, saved state and every downstream consumer are shared.

    if nargin < 2
        df = [];
    end

    % The empirical estimator takes both median lengthscales from the
    % option-level rows, not the basis grids. The values are computed once
    % here and passed into both alpha construction and K_h construction.
    % Simulations use the grid-based rule (the default).
    if strcmp(resolve_lengthscale_rule(cfg), 'option_level_median')
        if isempty(df)
            df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
        end
        [cfg.kernel.endo.lengthscale, cfg.kernel.exo.lengthscale] = ...
            option_level_median_lengthscales(cfg, df);
    end

    alphaPanel = build_alpha_panel(cfg, df);
    [K_h, K_M, K_V, basis] = build_kernel_norm_matrix(cfg, ...
        alphaPanel.mon_grid, alphaPanel.exo_grid);
    moments = build_return_second_moment(alphaPanel.alpha);

    cv = [];
    effdim = [];
    lambdaMethod = resolve_lambda_method(cfg);
    if strcmp(lambdaMethod, 'eff_dim')
        payoff = build_payoff_factor(alphaPanel.alpha);
        effdim = select_lambda_effective_dimension(payoff, K_h, ...
            moments.T, cfg.training.effdim.c);
        lambda = effdim.lambda;
    elseif should_cross_validate(cfg)
        cv = run_lambda_cross_validation(cfg, alphaPanel, K_h);
        lambda = cv.lambda;
    else
        lambda = cfg.training.lambda;
    end

    if strcmp(lambdaMethod, 'eff_dim')
        [h_star, R_sys, solver_status] = solve_projection_qr(payoff, K_h, lambda);
        conditionNumber = cond(R_sys)^2;
    else
        A = moments.Q_alpha + lambda * K_h;
        A = 0.5 * (A + A.');
        [h_star, solver_status] = solve_projection_system(A, moments.alpha_bar);
        conditionNumber = cond(A);
    end

    cfg.training.lambda = lambda;

    model_state = struct();
    model_state.h_star = h_star;
    model_state.H_star = reshape(h_star, basis.n + 1, basis.p);
    model_state.lambda = lambda;
    model_state.cv = cv;
    if strcmp(lambdaMethod, 'eff_dim')
        model_state.lambda_method = 'eff_dim';
        model_state.effdim = effdim;
    end

    model_state.mon_grid = basis.mon_grid;
    model_state.exo_grid = basis.exo_grid;
    model_state.ls_endo = basis.ls_endo;
    model_state.ls_exo = basis.ls_exo;
    model_state.reference_point = cfg.reference_point;

    model_state.K_h = K_h;
    model_state.K_M = K_M;
    model_state.K_V = K_V;
    model_state.alpha = alphaPanel.alpha;
    model_state.zeta = alphaPanel.zeta;
    model_state.kV = alphaPanel.kV;
    model_state.alpha_bar = moments.alpha_bar;
    model_state.Q_alpha = moments.Q_alpha;

    model_state.T = moments.T;
    model_state.Nh = numel(h_star);
    model_state.condition_number = conditionNumber;
    model_state.solver_status = solver_status;
    model_state.cfg = cfg;

    if isfield(cfg, 'real')
        model_state.meta = build_real_model_meta(cfg, model_state);
    end

    save_model_state(cfg, model_state);
end

function cv = run_lambda_cross_validation(cfg, alphaPanel, K_h)
%% Dispatch lambda CV by config method.
%
% The supported path is shape-blind expanding one-step-ahead CV.

    method = 'expanding_one_step';
    if isfield(cfg.training, 'cv') && isfield(cfg.training.cv, 'method') ...
            && ~isempty(cfg.training.cv.method)
        method = lower(char(string(cfg.training.cv.method)));
    end

    switch method
        case {'1_step_ahead', 'expanding_one_step'}
            cv = cross_validate_lambda(cfg, alphaPanel, K_h);
        otherwise
            error('train_projection_model:BadCVMethod', ...
                'Unknown lambda CV method: %s.', method);
    end
end

function rule = resolve_lengthscale_rule(cfg)
%% Return the kernel lengthscale convention, defaulting to the grid-median
%% rule when the configuration does not set one.

    rule = 'grid_median';
    if isfield(cfg, 'training') && isfield(cfg.training, 'lengthscale_rule') ...
            && ~isempty(cfg.training.lengthscale_rule)
        rule = lower(char(string(cfg.training.lengthscale_rule)));
    end

end

function method = resolve_lambda_method(cfg)
%% Return 'eff_dim' when the configuration asks for the effective-dimension
%% rule, else 'legacy' (the CV / fixed-lambda branch).

    method = 'legacy';
    if isfield(cfg, 'training') && isfield(cfg.training, 'lambda_method') ...
            && ~isempty(cfg.training.lambda_method)
        method = validatestring(char(string(cfg.training.lambda_method)), ...
            {'eff_dim', 'cv'}, mfilename, 'cfg.training.lambda_method');
        if strcmp(method, 'cv'), method = 'legacy'; end
    end
end

function tf = should_cross_validate(cfg)
%% Return whether lambda should be selected by cross-validation.

    tf = false;
    if isfield(cfg.training, 'use_cv') && cfg.training.use_cv
        tf = true;
    end
    if ~isfield(cfg.training, 'lambda') || isempty(cfg.training.lambda)
        tf = true;
    end
end

function [h_star, solver_status] = solve_projection_system(A, b)
%% Solve the final regularized projection system.

    solver_status = struct();
    solver_status.method = 'chol';
    solver_status.flag = 0;

    [L, flag] = chol(A, 'lower');
    if flag == 0
        h_star = L' \ (L \ b);
        return;
    end

    solver_status.method = 'backslash';
    solver_status.flag = flag;
    h_star = A \ b;
end
