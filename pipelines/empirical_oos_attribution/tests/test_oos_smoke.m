function R = test_oos_smoke(outRoot)
%% Small OOS validation: output safeguards, full-sample consistency, payoff
% identities, timing, absence of look-ahead, and fit reuse.
%
% outRoot must be a scratch folder outside results/. About 10 fits: the
% monthly full-sample fit (T = 273), monthly windows 1 and 81, one monthly
% look-ahead refit, weekly window 1 and its look-ahead refit, and two late
% daily windows (one with a binding embargo) with one 'python_reference' refit.

assert(nargin == 1 && ~isempty(outRoot), 'Pass a scratch outRoot.');
oos_setup();
opts = struct('outRoot', outRoot, 'mode', 'validation');
R = struct();

% 1. Output safeguards fire before any computation.
root = get_project_root();
R.protected_path_refused = throws(@() config_empirical_oos('monthly', ...
    struct('outRoot', fullfile(root, 'results', 'empirical', 'eff_dim'))), 'config_empirical_oos:ProtectedPath');
R.default_root_refused_in_validation = throws(@() run_empirical_oos_attribution({'monthly'}, ...
    struct('windows', [])), 'run_empirical_oos_attribution:Root');
assert(R.protected_path_refused && R.default_root_refused_in_validation);

% 2. Schedule and information rule (monthly).
oos = config_empirical_oos('monthly', opts);
panel = load_oos_panel(oos);
sched = build_oos_schedule(oos, panel);
oosPy = config_empirical_oos('monthly', setfield(opts, 'information_rule', 'python_reference')); %#ok<SFLD>
schedPy = build_oos_schedule(oosPy, panel);
R.monthly_T = sched.T; R.monthly_K = sched.K; R.monthly_windows = sched.n_windows;
R.monthly_embargo_binds = sched.n_windows_embargo_binds;
R.monthly_train_sets_differ_from_python = sum(~strcmp({sched.windows.train_key}, {schedPy.windows.train_key}));
R.realization_rules = unique(panel.dates_table.realization_rule).';

% 3. Full-sample consistency: the window wrapper on all monthly dates reproduces
% the saved full-sample fit.
winAll = struct('id', 0, 'oos_index', NaN, 'oos_date', panel.dates(end) + days(1), ...
    'train_index', (1:panel.T).');
t0 = tic; msAll = fit_oos_window(oos, panel, winAll); R.bridge_fit_seconds = toc(t0);
P = load(fullfile(root, 'results', 'empirical', 'eff_dim', 'monthly', 'monthly_model_state.mat'), 'model_state');
P = P.model_state;
R.bridge_lambda_rel = abs(msAll.lambda - P.lambda) / P.lambda;
R.bridge_h_rel = norm(msAll.h_star - P.h_star) / norm(P.h_star);
R.bridge_ls_equal = isequal(msAll.ls_endo, P.ls_endo) && isequal(msAll.ls_exo, P.ls_exo);
assert(R.bridge_lambda_rel <= 1e-10 && R.bridge_h_rel <= 1e-10 && R.bridge_ls_equal, 'Bridge failed.');

% 4. First and last monthly windows through the driver; identities.
run_empirical_oos_attribution({'monthly'}, setfield(opts, 'windows', [1 sched.n_windows])); %#ok<SFLD>
[Tm, ~] = assemble_oos_attribution('monthly', setfield(setfield(opts, 'allowPartial', true), 'write', false)); %#ok<SFLD>
R.monthly_identity_max = max(abs([Tm.identity1; Tm.identity2; Tm.identity3]));
R.monthly_identity_rel = R.monthly_identity_max / max(abs(Tm.total));
R.monthly_fit_seconds = Tm.fit_seconds.';
R.monthly_lambda = Tm.lambda.';
assert(R.monthly_identity_rel <= 1e-10, 'Payoff identities fail.');
assert(all(Tm.train_last < Tm.oos_date) && all(Tm.min_slack_days >= 0), 'Timing failed.');

% 5. Look-ahead test: perturbing every row dated >= tau leaves the window fit unchanged.
R.monthly_poison_identical = poison_test(oos, panel, sched.windows(1), opts, 'monthly');
oosW = config_empirical_oos('weekly', opts);
panelW = load_oos_panel(oosW);
schedW = build_oos_schedule(oosW, panelW);
R.weekly_windows = schedW.n_windows; R.weekly_embargo_binds = schedW.n_windows_embargo_binds;
run_empirical_oos_attribution({'weekly'}, setfield(opts, 'windows', 1)); %#ok<SFLD>
R.weekly_poison_identical = poison_test(oosW, panelW, schedW.windows(1), opts, 'weekly');
assert(R.monthly_poison_identical && R.weekly_poison_identical, 'Future data leak into a fit.');

% 6. A 'python_reference' run reuses primary fits where training sets are identical.
o = run_empirical_oos_attribution({'monthly'}, setfield(setfield(opts, ...
    'information_rule', 'python_reference'), 'windows', [1 sched.n_windows])); %#ok<SFLD>
R.reconciliation_new_fits = o.monthly.n_fit; R.reconciliation_reused = o.monthly.n_reused;
assert(R.reconciliation_new_fits == 0 && R.reconciliation_reused == 2, 'Monthly reconciliation should reuse both fits.');

% 7. Late daily windows: one where the embargo binds (an unverified realization
% bound excludes one training date) and the last window. Identities, timing,
% and fit reuse under 'python_reference': the binding window is refitted, the
% other reused.
oosD = config_empirical_oos('daily', opts);
panelD = load_oos_panel(oosD);
schedD = build_oos_schedule(oosD, panelD);
kb = find([schedD.windows.n_excluded_unrealized] > 0);
R.daily_embargo_windows = kb;
kd = [kb(end) schedD.n_windows];
run_empirical_oos_attribution({'daily'}, setfield(opts, 'windows', kd)); %#ok<SFLD>
[Td, ~] = assemble_oos_attribution('daily', setfield(setfield(opts, 'allowPartial', true), 'write', false)); %#ok<SFLD>
R.daily_identity_rel = max(abs([Td.identity1; Td.identity2; Td.identity3])) / max(abs(Td.total));
R.daily_fit_seconds = Td.fit_seconds.';
R.daily_n_train = Td.n_train.';
assert(R.daily_identity_rel <= 1e-10 && all(Td.train_last < Td.oos_date), 'Daily identities or timing fail.');
assert(Td.n_train(1) == kd(1) + schedD.K - 2, 'Binding window should exclude exactly one training date.');
od = run_empirical_oos_attribution({'daily'}, setfield(setfield(opts, ...
    'information_rule', 'python_reference'), 'windows', kd)); %#ok<SFLD>
R.daily_reconciliation_new_fits = od.daily.n_fit; R.daily_reconciliation_reused = od.daily.n_reused;
assert(od.daily.n_fit == 1 && od.daily.n_reused == 1, 'Daily reconciliation: refit 1, reuse 1 expected.');

% 8. Summary statistics on synthetic data: Delta SR of a series with itself is 0.
R.summary_selfcheck = summary_selfcheck();
assert(R.summary_selfcheck, 'Summary self-check failed.');

disp(R);
end

function ok = poison_test(oos, panel, win, opts, hz)
%% Refit one window after corrupting every row dated >= its OOS date; compare exactly.
S = load(fullfile(oos.paths.windows, sprintf('w%04d_%s.mat', win.id, ...
    char(string(win.oos_date, 'yyyyMMdd')))), 'W');
bad = panel.df.date >= win.oos_date;
P = panel;
P.df.VIX(bad) = P.df.VIX(bad) * 1.7 + 9;
P.df.OPT_NET_R_PAYOFF(bad) = -1.3 * P.df.OPT_NET_R_PAYOFF(bad) + 0.01;
P.df.SP500_ret(bad) = 3 * P.df.SP500_ret(bad) + 0.02;
P.df.frwd_moneyness(bad) = P.df.frwd_moneyness(bad) * 1.01;
ms = fit_oos_window(oos, P, win);
ok = isequal(ms.h_star, S.W.fit.h_star) && isequal(ms.lambda, S.W.fit.lambda) && ...
    isequal(ms.ls_endo, S.W.fit.ls_endo) && isequal(ms.ls_exo, S.W.fit.ls_exo) && ...
    isequal([min(ms.exo_grid) max(ms.exo_grid)], S.W.fit.exo_range);
fprintf('poison test %s window %d: identical = %d\n', hz, win.id, ok);
if strcmp(hz, 'weekly')
    trainV = panel.df.VIX(ismember(panel.df.date, panel.dates(win.train_index)));
    assert(isequal(S.W.fit.exo_range, [min(trainV) max(trainV)]), 'Weekly VIX grid is not the training range.');
end
unused = opts; %#ok<NASGU>
end

function ok = summary_selfcheck()
%% summarize_oos_attribution runs on a synthetic table and Delta SR(x, x) = 0.
rng(7);
n = 200;
T = table();
T.oos_date = datetime(2020,1,1) + caldays(7 * (0:n-1)).';
T.fwd_position = 1 + randn(n,1) * 0.2; T.vix = 15 + 5 * rand(n,1);
T.Delta = 0.002 + 0.02 * randn(n,1); T.carry_hold = 1e-4 * ones(n,1); T.rf_hold = 3e-4 * ones(n,1);
for b = 1:5
    T.(sprintf('c%d_put', b)) = 0.001 * randn(n,1);
    T.(sprintf('c%d_call', b)) = 0.001 * randn(n,1);
end
T.forward = T.fwd_position .* T.Delta;
cells = T{:, ~cellfun(@isempty, regexp(T.Properties.VariableNames, '^c\d_(put|call)$', 'once'))};
assert(size(cells, 2) == 10);
T.total = T.forward + sum(cells, 2);
T.identity1 = zeros(n,1); T.identity2 = zeros(n,1); T.identity3 = zeros(n,1);
meta = struct('protocol', struct('annualization', 52), 'initial_window_vix', 15 + 5 * rand(100,1), ...
    'bucket_labels', {{'b1','b2','b3','b4','b5'}});
S = summarize_oos_attribution(T, meta);
T2 = T; T2.forward = T.total; T2.fwd_position = T.total ./ T.Delta;
S2 = summarize_oos_attribution(T2, meta);
ok = abs(S2.delta_sharpe.dSR_ann(1)) < 1e-12 && all(isfinite(S.sharpe.SR_se)) && ...
    abs(S.legs.mean_share(1) + sum(S.legs.mean_share(2:3)) + S.legs.mean_share(4) ...
    + sum(S.legs.mean_share(7:8)) - 1) < 1e-10;
end

function tf = throws(f, id)
%% True if f() throws an error with identifier id.
try
    f(); tf = false;
catch err
    tf = strcmp(err.identifier, id);
    if ~tf, fprintf('unexpected error %s: %s\n', err.identifier, err.message); end
end
end
