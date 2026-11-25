function oos = config_empirical_oos(horizon, opts)
%% Configure the expanding-window OOS attribution for one maturity (eff_dim).
%
% The estimator configuration is config_empirical(horizon, 'eff_dim'); this
% file adds only the OOS protocol and the output locations.
%
% Protocol (manuscript Sec. 5, Panel C; OptPortOpt_V2 robustness/OOS/oos_run.py):
%   dates with fewer than min_options contracts are dropped before the split;
%   K = ceil(train_frac * T) initial dates; expanding window from the first
%   date; one refit per OOS holding period; lambda re-solved in every window
%   by the effective-dimension rule (pre-specified constant effdim_c; opts.effdim_c
%   overrides it for the target sensitivity, which requires a non-default outRoot).
% Information rule:
%   'realized_payoff'  (primary) a training date s enters the fit for OOS date
%                      tau only if its payoff is realized by the formation
%                      close of tau (realization_date(s) <= tau, load_oos_panel);
%   'python_reference' (alternative, no realization embargo) every date s < tau,
%                      as in oos_run.py.
%
% Outputs: <outRoot>/<rule folder>/<horizon>/, outRoot defaulting to
% <canonical root>/results/empirical_oos/eff_dim. Any outRoot inside
% results/empirical/ (the shape-test outputs) is refused.

if nargin < 2 || isempty(opts), opts = struct(); end
horizon = validatestring(char(string(horizon)), ...
    {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon');

cfg = config_empirical(horizon, 'eff_dim');
design = get_frozen_design();
maturity = get_empirical_maturity_config(horizon);
root = cfg.projectRoot;

oos = struct();
oos.horizon = horizon;
oos.role = 'primary';
if strcmp(horizon, 'quarterly'), oos.role = 'supplementary'; end

p = struct();
p.protocol_version = 'tr_oos_attribution_v1';
p.lambda_method = cfg.lambda_method;
p.effdim_c = get_opt(opts, 'effdim_c', design.effdim_c);
cfg.training.effdim.c = p.effdim_c;
p.train_frac = 0.7;
p.min_options = 10;
p.information_rule = validatestring(char(get_opt(opts, 'information_rule', ...
    'realized_payoff')), {'realized_payoff','python_reference'}, mfilename, ...
    'information_rule');
p.annualization = local_annualization(horizon);
p.region_cutoffs = maturity.region_cutoffs(:).';
p.test_domain = maturity.test_domain(:).';
p.estimator_domain = maturity.estimator_domain(:).';
p.exo_range_is_data_driven = maturity.exo_range_is_data_driven;
p.exo_grid_rule = 'config_empirical';
if p.exo_range_is_data_driven
    p.exo_grid_rule = 'training_window_min_max';   % as in oos_run.py
end
oos.protocol = p;
oos.cfg = cfg;

defaultRoot = local_abs(fullfile(results_root(), 'empirical_oos', 'eff_dim'));
canonicalRoot = local_abs(fullfile(root, 'results', 'empirical_oos', 'eff_dim'));
outRoot = local_abs(char(get_opt(opts, 'outRoot', defaultRoot)));
productionRoot = [local_abs(fullfile(results_root(), 'empirical')) filesep];
assert(~startsWith([outRoot filesep], productionRoot), ...
    'config_empirical_oos:ProtectedPath', ...
    'outRoot %s lies inside the frozen empirical results folder.', outRoot);

switch p.information_rule
    case 'realized_payoff',  ruleFolder = 'primary';
    case 'python_reference', ruleFolder = 'reconciliation_python_protocol';
end
oos.paths.outRoot = outRoot;
oos.paths.is_default_root = strcmp(outRoot, canonicalRoot);   % production root (gated)
assert(p.effdim_c == design.effdim_c || ~oos.paths.is_default_root, ...
    'config_empirical_oos:Root', 'A non-baseline effdim_c needs its own outRoot.');
oos.paths.outDir = fullfile(outRoot, ruleFolder, horizon);
oos.paths.windows = fullfile(oos.paths.outDir, 'windows');
oos.paths.scratch = fullfile(oos.paths.outDir, '_scratch');
oos.paths.primaryWindows = fullfile(outRoot, 'primary', horizon, 'windows');

end

function N = local_annualization(horizon)
%% Periods per year used for Sharpe annualization (manuscript convention).
switch horizon
    case 'quarterly', N = 4;
    case 'monthly',   N = 12;
    case 'weekly',    N = 52;
    case 'daily',     N = 252;
end
end

function p = local_abs(p)
%% Absolute, normalized path without a trailing separator.
f = java.io.File(p);
if ~f.isAbsolute(), f = java.io.File(pwd, p); end
p = char(f.getCanonicalPath());
end

function v = get_opt(opts, name, default)
%% Return opts.(name) or a default.
if isfield(opts, name) && ~isempty(opts.(name)), v = opts.(name); else, v = default; end
end
