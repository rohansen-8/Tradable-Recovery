function design = get_frozen_design()
%% Return the global design for estimation, inference, and simulation.
%
% This function and get_empirical_maturity_config are the two implementation
% authorities for the pre-specified design documented in README.md. Global scalar
% choices live here; per-maturity domains and cutoffs live in
% get_empirical_maturity_config. The verification suite asserts that both
% agree with README.md.

design = struct();
design.revision = 'public_v1';

% -- Empirical estimator --
design.kernel_endo = 'matern72';
design.kernel_exo = 'laplace';
design.lengthscale_rule = 'option_level_median';   % median pairwise distance
                                                   % over option-level rows
design.jitter = 1e-10;             % added to K_M and K_V separately, pre-kron
design.reference_point = 1;
design.estimator_grid_points = 30;
design.loss = 'projection_loss';

% -- Lambda selection (empirical) --
% empirical_lambda_method is the single switch for the empirical estimator:
%   'eff_dim' (baseline, manuscript eq. effective_dimension_rule): lambda
%             solves N_T(lambda) = effdim_c * T^(1/3) on the fitted sample,
%             continuously, with no grid and no floor;
%   'cv'      (robustness): one-step-ahead CV over lambda_grid.
% simulation_lambda_method is the rule used by the simulation production
% entry points (run_simulation_production, run_bs_study, run_chj_study) and
% by the simulation reporting functions; 'eff_dim' results are written under
% results/simulations/eff_dim/ and CV results under results/simulations/cv/.
% The simulation config functions themselves default to 'cv' when called
% without a method. Shape inference always tests the fitted object:
% lambda_inference = lambda_fit, with no second selection.
design.empirical_lambda_method = 'eff_dim';
design.simulation_lambda_method = 'eff_dim';
design.effdim_c = 1;
design.lambda_grid = logspace(-5, -2, 31).';
design.cv_method = '1_step_ahead';
design.inference_rule = 'lambda_fit';

% -- Shape inference --
design.shape_grid_points = 60;
design.shape_grid_type = 'global_midpoints';
design.shape_tests = {'mono','conv','conc'};
design.shape_convention = ...
    'mono:h_m>=0; conv:-h_mm>=0; conc:h_mm>=0; forward_moneyness';
design.hac_rule = 'nw_4_t100_2over9';
design.nMC = 10000;
design.test_alpha = 0.05;

% -- Conditional-figure and conditional-test VIX states --
% Five VIX levels chosen per maturity by a greedy furthest-point selector on
% the fitted model (select_vix_levels); integer legend labels are labels only.
design.vix_selector = struct('n_levels', 5, 'n_candidates', 50, ...
    'num_eval', 200);
design.figure_plot_points = 300;
design.figure_scaling = 'global_mean';   % one mu over all plotted curves

% -- Simulation studies --
% Every simulation experiment aligns its three forward-moneyness supports:
% generated option observations, the estimator basis, and the shape-test grid
% all use one common experiment-specific domain. The two increasing-T studies
% share the fixed domain [0.95, 1.05], the fixed region cutoffs
% (0.985, 1.015), J = 30 strikes and a G = 52 midpoint shape grid;
% bs_empirical_t uses exp(figure_log_range) per maturity, keeps that
% maturity's empirical region cutoffs, and keeps the global 60-point
% shape grid.
%
% J = 30 is even, so the equally spaced grid straddles m = 1 (15 puts, 15
% calls) rather than placing a contract on the ATM point, where the OTM side
% would be ambiguous. It is also the smallest such count with J + 1 >= N_h:
% alpha_t = [k00; g0]*Delta_t + [k0m; Kmm]*payoff_t maps R^(J+1) into the
% 31-dimensional moneyness basis, so J < 30 would leave basis directions
% spanned by the penalty alone. G = 52 keeps every midpoint clear of both
% region cutoffs.
design.bs_increasing_t = struct('experiment', 'bs_increasing_t', ...
    'horizon', 'monthly', 'T', [500 1000 2000 5000 10000], 'J', 30, ...
    'option_grid_type', 'forward_moneyness_linspace', ...
    'domain', [0.95 1.05], 'region_cutoffs', [0.985 1.015], ...
    'G', 52, 'B', 1000);
design.bs_empirical_t = struct('experiment', 'bs_empirical_t', ...
    'horizons', ["quarterly";"monthly";"weekly";"daily"], ...
    'option_grid_type', 'forward_moneyness_linspace', ...
    'domain_rule', 'exp_coauthor_display_log_range', ...
    'J_rule', 'empirical_median_date_strike_count_on_display_domain', ...
    'G', 60, 'B', 2000);
design.chj_increasing_t = struct('experiment', 'chj_increasing_t', ...
    'horizon', 'daily', 'T', [500 1000 2000 5000 10000], 'J', 30, ...
    'option_grid_type', 'forward_moneyness_linspace', ...
    'domain', [0.95 1.05], 'region_cutoffs', [0.985 1.015], ...
    'G', 52, 'B', 1000);

end
