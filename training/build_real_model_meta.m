function meta = build_real_model_meta(cfg, model_state)
%% Build metadata for real-data fitted SDF model_state.
%
% This metadata is meant to make downstream diagnostics and shape tests
% reproducible without inspecting the training code.
%
% Core coefficient notation follows code_grid:
%   h_star      : fitted restricted coefficient vector
%   alpha       : date-level alpha panel, alpha_t
%   alpha_bar   : T^{-1} sum_t alpha_t
%   Q_alpha     : T^{-1} sum_t alpha_t alpha_t'
%
% The fitted Carr--Madan replicating portfolio is:
%   alpha_t' * h_star.

    df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');

    [G, dates] = findgroups(df.date);
    nDates = numel(dates);

    meta = struct();

    meta.horizon = string(cfg.real.horizon);

    meta.dataset = struct();
    meta.dataset.dfFile = string(cfg.dataset.dfFile);
    meta.dataset.modelStateFile = string(cfg.dataset.modelStateFile);
    meta.dataset.n_rows = height(df);
    meta.dataset.n_dates = nDates;

    meta.basis = struct();
    meta.basis.n_mon_grid = numel(model_state.mon_grid);
    meta.basis.n_exo_grid = size(model_state.exo_grid, 1);
    meta.basis.basis_size = numel(model_state.h_star);
    meta.basis.coefficient_layout = ...
        "h_star = vec(H_star), H_star is (n_mon_grid + 1)-by-n_exo_grid";

    meta.domains = struct();
    meta.domains.training_mon_domain = cfg.training.mon_domain;
    meta.domains.testing_mon_domain = cfg.testing.mon_domain;
    meta.domains.training_exo_domain = cfg.training.exo_domain;
    meta.domains.moneyness_definition = "forward moneyness = frwd_moneyness";
    meta.domains.testing_edge_trim_fraction = cfg.testing.edge_trim_fraction;

    meta.grids = struct();
    meta.grids.mon_grid = model_state.mon_grid;
    meta.grids.exo_grid = model_state.exo_grid;
    meta.grids.training_mon_grid_rule = ...
        "30-point linspace over the fixed co-author moneyness range";
    meta.grids.training_exo_grid_rule = ...
        "30-point linspace over the fixed co-author VIX range, or the panel min/max when it is unset";
    meta.grids.conditional_exo_level_rule = ...
        "five greedy furthest-point selector states on the fitted model (select_vix_levels)";

    meta.kernel = struct();
    meta.kernel.endo_type = string(cfg.kernel.endo.type);
    meta.kernel.exo_type = string(cfg.kernel.exo.type);
    meta.kernel.ls_endo = model_state.ls_endo;
    meta.kernel.ls_exo = model_state.ls_exo;
    meta.kernel.lengthscale_rule = ...
        "median pairwise distance over the option-level estimation rows";
    meta.kernel.jitter = cfg.kernel.jitter;
    meta.kernel.jitter_rule = ...
        "added to K_M and K_V separately before the Kronecker product";

    meta.estimator = struct();
    meta.estimator.lambda = model_state.lambda;
    meta.estimator.system = "(Q_alpha + lambda K_h) h_star = alpha_bar";
    meta.estimator.alpha_name = "alpha";
    meta.estimator.alpha_definition = ...
        "date-level loading alpha_t satisfying replicating_portfolio_t = alpha_t' * h_star";
    meta.estimator.alpha_bar_name = "alpha_bar";
    meta.estimator.Q_alpha_name = "Q_alpha";
    meta.estimator.Q_alpha_definition = "T^{-1} sum_t alpha_t alpha_t'";
    meta.estimator.Q_alpha_centering = "uncentered";

    meta.cv = struct();
    if isfield(model_state, 'cv') && ~isempty(model_state.cv)
        meta.cv.method = string(model_state.cv.method);
        meta.cv.objective = string(model_state.cv.objective);
        meta.cv.selection_rule = string(model_state.cv.selection_rule);
        meta.cv.lambda_grid = model_state.cv.lambda_grid;
    else
        meta.cv.method = "fixed_lambda";
        meta.cv.objective = string(cfg.training.cv.objective);
        meta.cv.selection_rule = string(cfg.training.cv.selection_rule);
        meta.cv.lambda_grid = cfg.training.cv.lambda_grid;
    end
    meta.cv.admissible_lambda_grid = cfg.training.cv.lambda_grid;
    meta.cv.selected_lambda_source = ...
        "one-step expanding-window CV over logspace(-5,-2,31), or a fixed override";
    if isfield(model_state, 'lambda_method') && strcmp(model_state.lambda_method, 'eff_dim')
        % CV is not run under the effective-dimension rule; the cv block above
        % only records the unused CV specification.
        meta.cv.method = "not_used";
        meta.cv.selected_lambda_source = "not_used (lambda_method = eff_dim)";
        meta.lambda_selection = struct( ...
            'method', "eff_dim", ...
            'rule', "N_T(lambda) = c * T^(1/3), N_T = tr[G_T (G_T + lambda T I)^{-1}]", ...
            'c', model_state.effdim.c, ...
            'T', model_state.effdim.T, ...
            'target', model_state.effdim.target, ...
            'lambda', model_state.effdim.lambda, ...
            'residual', model_state.effdim.residual, ...
            'numerical_rank', model_state.effdim.numerical_rank, ...
            'grid_or_floor', "none (continuous root in log lambda)");
    end

    meta.shape = struct();
    meta.shape.object = "fitted function h(m,v); tests are on h_m and h_mm";
    meta.shape.mono = "mono: h_m >= 0";
    meta.shape.conv = "conv: -h_mm >= 0";
    meta.shape.test_identifiers = ["mono", "conv", "conc"];
    meta.shape.region_names = string({cfg.testing.regionSpecs.name});
    meta.shape.region_bounds = vertcat(cfg.testing.regionSpecs.bounds);

    meta.normalization = struct();
    meta.normalization.method = "HJ curve normalization for diagnostics/plots";
    meta.normalization.raw_model_state = ...
        "h_star is stored unnormalized; raw shape signs are invariant to positive HJ rescaling";
    meta.normalization.note = ...
        "Formal shape tests may use raw h_star because mono/conv signs are unchanged by positive affine SDF normalization.";

    meta.created_by = "train_projection_model + build_real_model_meta";
    meta.created_on = string(datetime("now"));

    % Date-level sanity checks.
    meta.sanity = struct();
    meta.sanity.SP500_ret_constant_by_date = is_date_constant(df, G, "SP500_ret");
    meta.sanity.VIX_constant_by_date = is_date_constant(df, G, "VIX");
end

function ok = is_date_constant(df, G, colName)

    ok = false;

    if ~ismember(char(colName), df.Properties.VariableNames)
        return;
    end

    x = df.(char(colName));
    tol = 1e-12;

    for g = 1:max(G)
        vals = x(G == g);
        vals = vals(isfinite(vals));

        if isempty(vals)
            continue;
        end

        if max(vals) - min(vals) > tol
            return;
        end
    end

    ok = true;
end
