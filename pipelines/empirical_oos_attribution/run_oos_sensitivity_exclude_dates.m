function out = run_oos_sensitivity_exclude_dates(horizon, excludeDates, opts)
%% Sensitivity run: the primary OOS exercise with given dates removed from training and evaluation.
%
% out = run_oos_sensitivity_exclude_dates('daily', datetime(2023,10,19), opts)
%
% The data file, the baseline schedule and the baseline window files are not
% changed. The schedule is the primary ('realized_payoff') schedule on the full
% panel, so K, the OOS dates and every other training set are those of the
% baseline; the excluded dates are then (i) removed from every training set and
% (ii) dropped as OOS dates (their windows are not evaluated). A window whose
% training set does not change is copied from the baseline primary window file,
% after checking that its training, code, data and MATLAB fingerprint fields
% match; every other window is refitted with the baseline estimator.
%
% Outputs: <outRoot>/sensitivity_excl_<yyyymmdd...>/<horizon>/windows/, one
% file per evaluated window, named and fingerprinted as in the baseline plus
% the excluded dates and this file's SHA-256. Existing files are loaded after a
% fingerprint check and never overwritten. opts: outRoot (default production
% root), windows (baseline window ids, default all), sdfRoot.

if nargin < 3 || isempty(opts), opts = struct(); end
oos_setup(get_opt(opts, 'sdfRoot', ''));
excludeDates = sort(excludeDates(:));
oos = config_empirical_oos(horizon, opts);
assert(strcmp(oos.protocol.information_rule, 'realized_payoff'), ...
    'run_oos_sensitivity_exclude_dates:Rule', 'The sensitivity is defined on the primary protocol.');
tag = ['sensitivity_excl_' strjoin(cellstr(string(excludeDates, 'yyyyMMdd')), '_')];
outDir = fullfile(oos.paths.outRoot, tag, oos.horizon);
oos.paths.outDir = outDir;
oos.paths.windows = fullfile(outDir, 'windows');
oos.paths.scratch = fullfile(outDir, '_scratch');
if ~isfolder(oos.paths.windows), mkdir(oos.paths.windows); end
if ~isfolder(oos.paths.scratch), mkdir(oos.paths.scratch); end

prov = oos_provenance(oos);
selfSha = oos_sha256(fileread([mfilename('fullpath') '.m']));
panel = load_oos_panel(oos);
sched = build_oos_schedule(oos, panel);
xIdx = find(ismember(panel.dates, excludeDates));
assert(numel(xIdx) == numel(excludeDates), 'run_oos_sensitivity_exclude_dates:Date', ...
    'An excluded date is not a panel date after the thin-date filter.');

protocolHash = oos_sha256([oos.horizon '|' jsonencode(oos.protocol)]);
coreHash = oos_sha256([oos.horizon '|' jsonencode(rmfield(oos.protocol, 'information_rule'))]);
keepW = find(~ismember([sched.windows.oos_index], xIdx));
sel = intersect(get_opt(opts, 'windows', keepW), keepW);

runMeta = struct('mode', 'sensitivity', 'excluded_dates', excludeDates, 'protocol', oos.protocol, ...
    'protocol_hash', protocolHash, 'provenance', prov, 'sensitivity_code_sha256', selfSha, ...
    'K', sched.K, 'T', sched.T, 'n_windows', numel(keepW), 'window_ids', keepW, ...
    'dropped_window_ids', setdiff(1:sched.n_windows, keepW), ...
    'initial_window_vix', panel.dates_table.vix(setdiff(sched.windows(1).train_index, xIdx)), ...
    'paths', oos.paths);
save(fullfile(outDir, sprintf('run_%s.mat', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS')))), 'runMeta');

st = struct('n_fit', 0, 'n_loaded', 0, 'n_copied', 0);
for k = sel(:).'
    win = sched.windows(k);
    base = win;
    win.train_index = setdiff(win.train_index, xIdx);
    win.n_train = numel(win.train_index);
    win.train_key = strjoin(cellstr(string(panel.dates(win.train_index), 'yyyyMMdd')), ',');
    fp = struct('horizon', oos.horizon, 'window_id', win.id, ...
        'oos_date', char(string(win.oos_date, 'yyyy-MM-dd')), ...
        'protocol_hash', protocolHash, 'protocol_core_hash', coreHash, ...
        'code_hash', prov.code_hash, 'data_sha256', prov.data_sha256, ...
        'matlab_version', prov.matlab_version, 'train_hash', oos_sha256(win.train_key), ...
        'excluded_dates', char(strjoin(string(excludeDates, 'yyyy-MM-dd'), ',')), ...
        'sensitivity_code_sha256', selfSha);
    fname = sprintf('w%04d_%s.mat', win.id, char(string(win.oos_date, 'yyyyMMdd')));
    target = fullfile(oos.paths.windows, fname);
    if isfile(target)
        S = load(target, 'W');
        assert(isequal(S.W.fingerprint, fp), 'run_oos_sensitivity_exclude_dates:Fingerprint', ...
            'Existing %s has a different fingerprint; refusing to reuse or overwrite.', target);
        st.n_loaded = st.n_loaded + 1;
        continue
    end
    primaryFile = fullfile(oos.paths.primaryWindows, fname);
    if isequal(win.train_index, base.train_index) && isfile(primaryFile)
        S = load(primaryFile, 'W');
        pf = S.W.fingerprint;
        assert(strcmp(pf.train_hash, fp.train_hash) && strcmp(pf.protocol_hash, protocolHash) ...
            && strcmp(pf.code_hash, fp.code_hash) && strcmp(pf.data_sha256, fp.data_sha256) ...
            && strcmp(pf.matlab_version, fp.matlab_version), ...
            'run_oos_sensitivity_exclude_dates:Baseline', 'Baseline %s does not match.', primaryFile);
        W = S.W;
        W.fingerprint = fp;
        W.reused_from = primaryFile;
        atomic_save(target, W);
        st.n_copied = st.n_copied + 1;
        continue
    end
    W = compute_window(oos, panel, win, fp);
    atomic_save(target, W);
    st.n_fit = st.n_fit + 1;
end
st.windows_dir = oos.paths.windows;
st.outDir = outDir;
st.n_windows = numel(keepW);
out = st;
fprintf('%s [%s]: fitted %d, copied %d, loaded %d of %d windows\n', oos.horizon, tag, ...
    st.n_fit, st.n_copied, st.n_loaded, numel(keepW));
end

function W = compute_window(oos, panel, win, fp)
%% Fit one window and evaluate its payoff components (as the baseline driver).
t0 = tic;
ms = fit_oos_window(oos, panel, win);
tFit = toc(t0);
rows = panel.df.date == win.oos_date;
comp = evaluate_oos_payoff_components(oos, ms, panel.df(rows, :), panel.half_spread(rows));
D = panel.dates_table(win.oos_index, :);
trainDates = panel.dates(win.train_index);
W = struct();
W.fingerprint = fp;
W.window = struct('id', win.id, 'oos_date', win.oos_date, 'n_train', win.n_train, ...
    'train_first', trainDates(1), 'train_last', trainDates(end), ...
    'n_excluded_unrealized', win.n_excluded_unrealized, 'min_slack_days', win.min_slack_days);
W.fit = struct('lambda', ms.lambda, 'T', ms.T, 'h_star', ms.h_star, ...
    'effdim_target', ms.effdim.target, 'effdim_N_at_lambda', ms.effdim.N_at_lambda, ...
    'effdim_residual', ms.effdim.residual, 'effdim_rank', ms.effdim.numerical_rank, ...
    'effdim_rank_upper_bound', ms.effdim.rank_upper_bound, 'ls_endo', ms.ls_endo, ...
    'ls_exo', ms.ls_exo, 'exo_range', [min(ms.exo_grid(:)) max(ms.exo_grid(:))], ...
    'solver_status', ms.solver_status, 'condition_number', ms.condition_number);
W.comp = comp;
W.state = struct('vix', D.vix, 'Delta', D.Delta, 'S', D.S, 'S_next', D.S_next, ...
    'rf_hold', D.rf_hold, 'carry_hold', D.carry_hold, 'holding_days', D.holding_days, ...
    'realization_rule', D.realization_rule, 'n_expiries', D.n_expiries);
W.reused_from = '';
W.timing = struct('fit_seconds', tFit, 'eval_seconds', NaN);
W.created_on = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end

function atomic_save(target, W)
%% Write to a temporary file, then rename; never replace an existing file.
assert(~isfile(target), 'run_oos_sensitivity_exclude_dates:Exists', '%s already exists.', target);
tmp = [target '.tmp_' char(java.util.UUID.randomUUID())];
save(tmp, 'W', '-v7');
[ok, msg] = movefile(tmp, target);
assert(ok, 'run_oos_sensitivity_exclude_dates:Move', 'Could not move %s: %s', tmp, msg);
end

function v = get_opt(opts, name, default)
%% Return opts.(name) or a default.
if isfield(opts, name) && ~isempty(opts.(name)), v = opts.(name); else, v = default; end
end
